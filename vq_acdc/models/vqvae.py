"""
VQ-VAE for cardiac image modelling (SEG and MRI).

Supports:
  - Standard VQ-VAE  (residual=False)
  - Residual VQ-VAE  (residual=True) — stacked codebooks (RQ-VAE)

Architecture:
  Encoder  : strided convolutions + 2 residual blocks → latent [B, D, H/s, W/s]
  VQ layer : vector quantization via vector_quantize_pytorch
  Decoder  : residual blocks + upsampling
               SEG → ConvTranspose2d (learnable upsampling)
               MRI → Bilinear upsample + Conv (avoids checkerboard artifacts)
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, Dict
from vector_quantize_pytorch import VectorQuantize, ResidualVQ


# ── Building blocks ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


# ── Encoder / Decoder factories ───────────────────────────────────────────────

_HIDDEN = {2: [64], 4: [64, 128], 8: [64, 128, 256]}


def _build_encoder(in_ch: int, D: int, downsampling: int) -> nn.Sequential:
    if downsampling not in _HIDDEN:
        raise ValueError(f"downsampling must be 2, 4, or 8 — got {downsampling}")
    hidden = _HIDDEN[downsampling]
    layers = []
    for h in hidden:
        layers += [nn.Conv2d(in_ch, h, 4, stride=2, padding=1), nn.LeakyReLU()]
        in_ch = h
    layers += [nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.LeakyReLU()]
    layers += [ResidualBlock(in_ch), ResidualBlock(in_ch), nn.LeakyReLU()]
    layers += [nn.Conv2d(in_ch, D, 1), nn.LeakyReLU()]
    return nn.Sequential(*layers)


def _build_decoder_seg(D: int, out_ch: int, downsampling: int) -> nn.Sequential:
    """ConvTranspose2d decoder — suitable for discrete segmentation maps."""
    hidden = _HIDDEN[downsampling]
    top = hidden[-1]
    rev = list(reversed(hidden))

    layers = [nn.Conv2d(D, top, 3, padding=1), nn.LeakyReLU()]
    layers += [ResidualBlock(top), ResidualBlock(top), nn.LeakyReLU()]
    for i in range(len(rev) - 1):
        layers += [nn.ConvTranspose2d(rev[i], rev[i + 1], 4, stride=2, padding=1), nn.LeakyReLU()]
    layers += [nn.ConvTranspose2d(rev[-1], out_ch, 4, stride=2, padding=1), nn.ReLU()]
    return nn.Sequential(*layers)


def _build_decoder_mri(D: int, out_ch: int, downsampling: int) -> nn.Sequential:
    """Bilinear upsample + Conv decoder — avoids checkerboard artifacts on MRI."""
    hidden = _HIDDEN[downsampling]
    top = hidden[-1]
    rev = list(reversed(hidden))

    layers = [nn.Conv2d(D, top, 3, padding=1), nn.LeakyReLU()]
    layers += [ResidualBlock(top), ResidualBlock(top), nn.LeakyReLU()]
    for i in range(len(rev) - 1):
        layers += [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(rev[i], rev[i + 1], 3, padding=1), nn.LeakyReLU(),
        ]
    layers += [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(rev[-1], out_ch, 3, padding=1), nn.ReLU(),
    ]
    return nn.Sequential(*layers)


# ── Model ─────────────────────────────────────────────────────────────────────

class VQVAE(nn.Module):
    """
    VQ-VAE for cardiac images.

    Args:
        modality:        'SEG' (segmentation) or 'MRI' (grayscale scan).
        K:               Codebook size (number of embeddings).
        D:               Embedding dimension.
        downsampling:    Spatial downsampling factor — 2, 4, or 8.
        beta:            Commitment loss weight.
        decay:           EMA decay for codebook updates (0 = no EMA).
        residual:        If True, uses Residual VQ (stacked codebooks).
        num_quantizers:  Number of stacked quantizers (residual=True only).
        shared_codebook: Share one codebook across all quantizers (residual only).
    """

    _IN_CHANNELS = {'SEG': 4, 'MRI': 1}

    def __init__(
        self,
        modality: str,
        K: int = 512,
        D: int = 64,
        downsampling: int = 4,
        beta: float = 0.25,
        decay: float = 0.8,
        residual: bool = False,
        num_quantizers: int = 2,
        shared_codebook: bool = False,
        **vq_kwargs,
    ):
        super().__init__()
        if modality not in self._IN_CHANNELS:
            raise ValueError(f"modality must be 'SEG' or 'MRI' — got '{modality}'")

        self.modality = modality
        self.K = K
        self.D = D
        self.residual = residual

        C = self._IN_CHANNELS[modality]
        self.encoder = _build_encoder(C, D, downsampling)

        if residual:
            self.vq = ResidualVQ(
                dim=D, codebook_size=K, num_quantizers=num_quantizers,
                commitment_weight=beta, decay=decay,
                shared_codebook=shared_codebook, accept_image_fmap=True,
                **vq_kwargs,
            )
        else:
            self.vq = VectorQuantize(
                dim=D, codebook_size=K, commitment_weight=beta,
                decay=decay, accept_image_fmap=True, **vq_kwargs,
            )

        build_dec = _build_decoder_seg if modality == 'SEG' else _build_decoder_mri
        self.decoder = build_dec(D, C, downsampling)

    # ── Core forward ──────────────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """Encoder output before quantization: [B, D, H/s, W/s]."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decoder: quantized codes → reconstructed image."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (reconstruction, indices, commitment_loss)."""
        z = self.encode(x)
        z_q, indices, commit_loss = self.vq(z)
        return self.decode(z_q), indices, commit_loss

    # ── Loss ──────────────────────────────────────────────────────────────────

    def loss(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Full forward + loss in one call.

        SEG loss : cross-entropy between logits and one-hot targets.
                   PyTorch >= 1.10 accepts float [B, C, H, W] targets directly.
        MRI loss : MSE between reconstruction and input.

        Returns a dict with keys: 'loss', 'recon', 'commit'.
        """
        recon, _, commit_loss = self(x)

        if self.modality == 'SEG':
            recon_loss = F.cross_entropy(recon, x)
        else:
            recon_loss = F.mse_loss(recon, x)

        if self.residual:
            commit_loss = commit_loss.sum()

        return {
            'loss':   recon_loss + commit_loss,
            'recon':  recon_loss,
            'commit': commit_loss,
        }

    # ── Inference helpers ─────────────────────────────────────────────────────

    def reconstruct(self, x: Tensor) -> Tensor:
        """Returns the reconstruction as probabilities (SEG) or raw values (MRI)."""
        recon, _, _ = self(x)
        return F.softmax(recon, dim=1) if self.modality == 'SEG' else recon

    @torch.no_grad()
    def encode_indices(self, x: Tensor) -> Tensor:
        """Returns quantized codebook indices without gradient tracking."""
        _, indices, _ = self.vq(self.encode(x))
        return indices

    @torch.no_grad()
    def codebook_usage(self, x: Tensor) -> Tensor:
        """
        Histogram of codebook index usage over a batch.
        Returns [K] for standard VQ, or [num_quantizers, K] for residual VQ.
        """
        indices = self.encode_indices(x)
        if self.residual:
            K = self.vq.codebook_sizes[0]
            return torch.stack([
                torch.bincount(indices[..., i].reshape(-1), minlength=K)
                for i in range(indices.shape[-1])
            ])
        return torch.bincount(indices.reshape(-1), minlength=self.vq.codebook_size)
