#!/usr/bin/env python3
"""
Refit: reinitialize the codebook of a trained VQ-VAE using k-means++ and fine-tune.

This dramatically improves codebook utilization by seeding the new (smaller)
codebook from the actual latent distribution of the pre-trained encoder,
instead of relying on random initialization and EMA updates alone.

Usage:
    python refit.py --baseline checkpoints/seg_512.pth \\
                    --out checkpoints/seg_refit256.pth \\
                    --new_K 256 --epochs 50
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import Compose, Resize, InterpolationMode
from torch.utils.data import DataLoader
from sklearn.cluster import kmeans_plusplus
from tqdm import tqdm

from vq_acdc.data import load_dataset, ACDCDataset, OneHotEncode, PercentileClip, MinMaxNormalize
from vq_acdc.models import VQVAE
from vq_acdc.utils.io import save_checkpoint, load_checkpoint, save_metadata, load_metadata
from vq_acdc.utils.training import evaluate, codebook_stats


_DEFAULT_DATA_PATH = "/home/ids/ihamdaoui-21/ACDC/database"


def parse_args():
    p = argparse.ArgumentParser(description="Refit VQ-VAE with k-means++ codebook init")
    p.add_argument("--baseline",     required=True, help="Path to baseline checkpoint (.pth)")
    p.add_argument("--out",          required=True, help="Output checkpoint path (.pth)")
    p.add_argument("--new_K",        type=int, required=True, help="New (smaller) codebook size")
    p.add_argument("--data_path",    default=os.environ.get("ACDC_DATA_PATH", _DEFAULT_DATA_PATH))
    p.add_argument("--img_size",     type=int, default=128)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--epochs",       type=int, default=50)
    return p.parse_args()


def build_loaders(data_path, modality, img_size, batch_size):
    sz = (img_size, img_size)
    if modality == 'SEG':
        tfm = Compose([Resize(sz, interpolation=InterpolationMode.NEAREST), OneHotEncode()])
    else:
        tfm = Compose([Resize(sz, interpolation=InterpolationMode.NEAREST),
                       PercentileClip(), MinMaxNormalize()])

    train_data = load_dataset(os.path.join(data_path, "training"), modality)
    test_data  = load_dataset(os.path.join(data_path, "testing"),  modality)

    train_loader = DataLoader(ACDCDataset(train_data, tfm), batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ACDCDataset(test_data,  tfm), batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def collect_latents(model: VQVAE, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Runs the encoder over the training set and returns all latent vectors."""
    model.eval()
    vecs = []
    for batch in tqdm(loader, desc="Collecting latents"):
        z = model.encode(batch.float().to(device))     # [B, D, H, W]
        B, D, H, W = z.shape
        vecs.append(z.permute(0, 2, 3, 1).reshape(-1, D).cpu().numpy())
    model.train()
    return np.concatenate(vecs, axis=0)                # [N, D]


def seed_codebook(vq, centroids: torch.Tensor, device: torch.device):
    """
    Seeds a VectorQuantize codebook with pre-computed centroids.
    Tries the two most common attribute paths across library versions.
    """
    c = centroids.float().to(device)
    with torch.no_grad():
        if hasattr(vq, '_codebook') and hasattr(vq._codebook, 'embed'):
            # vector_quantize_pytorch >= 1.x
            vq._codebook.embed.data.copy_(c)
        elif hasattr(vq, 'codebook'):
            cb = vq.codebook
            if hasattr(cb, 'weight'):           # nn.Embedding
                cb.weight.data.copy_(c)
            else:                               # raw tensor / buffer
                cb.data.copy_(c)
        else:
            raise AttributeError(
                "Cannot find the VQ codebook attribute. "
                "Inspect model.vq to locate the embedding tensor and update seed_codebook()."
            )


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Load baseline ─────────────────────────────────────────────────────────
    meta_path = args.baseline.replace('.pth', '_meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            "Make sure the baseline was trained with train.py (which auto-saves metadata)."
        )
    meta = load_metadata(meta_path)

    baseline = VQVAE(
        modality=meta['modality'], K=meta['K'], D=meta['D'],
        downsampling=meta['downsampling'], beta=meta['beta'], decay=meta['decay'],
    ).to(device)
    load_checkpoint(args.baseline, baseline, device)
    print(f"Loaded baseline: {args.baseline}  (K={meta['K']}, D={meta['D']})")
    print(f"Refit target:   new_K={args.new_K}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(
        args.data_path, meta['modality'], args.img_size, args.batch_size)

    # ── k-means++ codebook initialization ────────────────────────────────────
    print("Step 1 — collecting encoder latents from training set...")
    latents = collect_latents(baseline, train_loader, device)
    print(f"  {latents.shape[0]:,} latent vectors of dim {latents.shape[1]}")

    print(f"Step 2 — fitting k-means++ with new_K={args.new_K}...")
    centers, _ = kmeans_plusplus(latents, n_clusters=args.new_K, random_state=42)
    print("  done.\n")

    # ── Build refitted model ──────────────────────────────────────────────────
    model = VQVAE(
        modality=meta['modality'], K=args.new_K, D=meta['D'],
        downsampling=meta['downsampling'], beta=meta['beta'], decay=meta['decay'],
    ).to(device)

    model.encoder.load_state_dict(baseline.encoder.state_dict())
    model.decoder.load_state_dict(baseline.decoder.state_dict())
    seed_codebook(model.vq, torch.from_numpy(centers), device)
    print("Step 3 — model ready. Encoder/decoder initialized from baseline, "
          "codebook seeded from k-means++.\n")

    del baseline  # free memory

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    train_losses, val_losses, commit_losses = [], [], []
    best_val, best_epoch = float('inf'), 0

    for epoch in range(args.epochs):
        model.train()
        ep_recon, ep_commit = [], []

        with tqdm(train_loader, desc=f"Refit {epoch + 1:>3}/{args.epochs}") as pbar:
            for batch in pbar:
                x = batch.float().to(device)
                optimizer.zero_grad()
                losses = model.loss(x)
                losses['loss'].backward()
                optimizer.step()
                ep_recon.append(losses['recon'].item())
                ep_commit.append(losses['commit'].item())
                pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

        train_losses.append(float(np.mean(ep_recon)))
        commit_losses.append(float(np.mean(ep_commit)))

        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        marker = ''
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint(args.out, model, epoch,
                            train_losses, val_losses, commit_losses, val_loss)
            marker = '  ✓ saved'

        print(f"  train={train_losses[-1]:.4f}  val={val_loss:.4f}  "
              f"best=epoch{best_epoch + 1}({best_val:.4f}){marker}")

    print(f"\nRefit complete. Best: epoch {best_epoch + 1}, val={best_val:.4f}")
    print("\nCodebook usage on validation set:")
    load_checkpoint(args.out, model, device)
    _, pct = codebook_stats(model, val_loader, device)

    meta_path_out = args.out.replace('.pth', '_meta.json')
    save_metadata(meta_path_out, {
        **meta,
        'K':            args.new_K,
        'baseline':     args.baseline,
        'best_epoch':   best_epoch,
        'best_val':     best_val,
        'codebook_usage': pct,
    })
    print(f"Metadata saved → {meta_path_out}")


if __name__ == "__main__":
    main()
