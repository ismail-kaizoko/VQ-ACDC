#!/usr/bin/env python3
"""
Train a VQ-VAE then refine it with k-means++-seeded codebook (Refit).

Two modes:

  1. Full pipeline  (no --baseline):
       Trains a baseline with --init_K codebook for --init_epochs,
       saves it, then refits with k-means++ to --new_K for --refit_epochs.
       --init_K may equal --new_K (same size, just better code distribution).

  2. Refit only  (--baseline provided):
       Loads an existing checkpoint, skips baseline training, runs refit.

Usage:
    # Full pipeline: 512 → 256
    python train_with_refit.py --modality SEG \\
        --init_K 512 --init_epochs 100 \\
        --new_K 256 --refit_epochs 50 \\
        --out checkpoints/seg_refit256.pth

    # Same size refit (improve utilisation without shrinking)
    python train_with_refit.py --modality SEG \\
        --init_K 256 --init_epochs 100 \\
        --new_K 256 --refit_epochs 50 \\
        --out checkpoints/seg_refit256.pth

    # Refit an existing checkpoint
    python train_with_refit.py --modality SEG \\
        --baseline checkpoints/seg_512.pth \\
        --new_K 256 --refit_epochs 50 \\
        --out checkpoints/seg_refit256.pth

Set ACDC_DATA_PATH env variable to override the default dataset path.
"""

import os
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VQ-VAE training + Refit pipeline")

    # Data
    p.add_argument("--modality",      required=True, choices=['SEG', 'MRI'])
    p.add_argument("--data_path",     default=os.environ.get("ACDC_DATA_PATH", _DEFAULT_DATA_PATH))
    p.add_argument("--img_size",      type=int, default=128)

    # Output
    p.add_argument("--out",           required=True, help="Final refitted checkpoint (.pth)")

    # Shared optimiser / dataloader
    p.add_argument("--batch_size",    type=int, default=16)
    p.add_argument("--lr",            type=float, default=5e-4)

    # Shared model architecture (used for baseline; loaded from metadata when --baseline given)
    p.add_argument("--D",             type=int, default=64, help="Embedding dimension")
    p.add_argument("--downsampling",  type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--beta",          type=float, default=0.25, help="Commitment loss weight")
    p.add_argument("--decay",         type=float, default=0.8, help="EMA decay for codebook")

    # Baseline: provide an existing checkpoint OR let us train one
    p.add_argument("--baseline",      default=None,
                   help="Skip baseline training and load this checkpoint instead")
    p.add_argument("--init_K",        type=int, default=None,
                   help="Codebook size for baseline training (required when --baseline is not set)")
    p.add_argument("--init_epochs",   type=int, default=100,
                   help="Epochs for baseline training")

    # Refit
    p.add_argument("--new_K",         type=int, required=True, help="Codebook size after refit")
    p.add_argument("--refit_epochs",  type=int, default=50)

    return p.parse_args()


def _validate_args(args):
    if args.baseline is None and args.init_K is None:
        raise ValueError("Provide either --baseline (existing checkpoint) "
                         "or --init_K (to train a baseline from scratch).")


# ── Data ──────────────────────────────────────────────────────────────────────

def build_loaders(data_path, modality, img_size, batch_size):
    sz = (img_size, img_size)
    if modality == 'SEG':
        tfm = Compose([Resize(sz, interpolation=InterpolationMode.NEAREST), OneHotEncode()])
    else:
        tfm = Compose([Resize(sz, interpolation=InterpolationMode.NEAREST),
                       PercentileClip(), MinMaxNormalize()])

    train_data = load_dataset(os.path.join(data_path, "training"), modality)
    test_data  = load_dataset(os.path.join(data_path, "testing"),  modality)

    train_loader = DataLoader(ACDCDataset(train_data, tfm), batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ACDCDataset(test_data,  tfm), batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ── Training loop (shared by baseline and refit phases) ───────────────────────

def _run_epochs(model, train_loader, val_loader, device, optimizer, out_path,
                epochs, label) -> tuple:
    """
    Trains `model` for `epochs` epochs, saves the best checkpoint to `out_path`.
    Returns (best_epoch, best_val, train_losses, val_losses, commit_losses).
    """
    train_losses, val_losses, commit_losses = [], [], []
    best_val, best_epoch = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        ep_recon, ep_commit = [], []

        with tqdm(train_loader, desc=f"{label} {epoch + 1:>3}/{epochs}") as pbar:
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
            save_checkpoint(out_path, model, epoch,
                            train_losses, val_losses, commit_losses, val_loss)
            marker = '  ✓ saved'

        print(f"  train={train_losses[-1]:.4f}  val={val_loss:.4f}  "
              f"best=epoch{best_epoch + 1}({best_val:.4f}){marker}")

    return best_epoch, best_val


# ── Refit utilities ───────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_latents(model: VQVAE, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Runs the encoder over the training set and returns all spatial latent vectors."""
    model.eval()
    vecs = []
    for batch in tqdm(loader, desc="  collecting latents"):
        z = model.encode(batch.float().to(device))   # [B, D, H, W]
        B, D, H, W = z.shape
        vecs.append(z.permute(0, 2, 3, 1).reshape(-1, D).cpu().numpy())
    model.train()
    return np.concatenate(vecs, axis=0)              # [N, D]


def _seed_codebook(vq, centroids: torch.Tensor, device: torch.device):
    """
    Seeds a VectorQuantize codebook with pre-computed centroids.
    Handles the two most common attribute layouts across library versions.
    """
    c = centroids.float().to(device)
    with torch.no_grad():
        if hasattr(vq, '_codebook') and hasattr(vq._codebook, 'embed'):
            vq._codebook.embed.data.copy_(c)
        elif hasattr(vq, 'codebook'):
            cb = vq.codebook
            (cb.weight if hasattr(cb, 'weight') else cb).data.copy_(c)
        else:
            raise AttributeError(
                "Cannot locate the VQ codebook tensor. "
                "Inspect model.vq and update _seed_codebook() accordingly."
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _validate_args(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"Device   : {device}")
    print(f"Modality : {args.modality}")
    print(f"Output   : {args.out}\n")

    train_loader, val_loader = build_loaders(
        args.data_path, args.modality, args.img_size, args.batch_size)

    # ── Phase 1: obtain baseline ──────────────────────────────────────────────

    if args.baseline:
        # Load existing checkpoint
        meta_path = args.baseline.replace('.pth', '_meta.json')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Metadata not found: {meta_path}\n"
                "Baseline must have been trained with train.py or train_with_refit.py."
            )
        meta = load_metadata(meta_path)
        baseline_path = args.baseline
        print(f"[Phase 1] Loading baseline: {baseline_path}  (K={meta['K']}, D={meta['D']})")

    else:
        # Train baseline from scratch
        stem = os.path.splitext(args.out)[0]
        baseline_path = f"{stem}_baseline_K{args.init_K}.pth"
        meta = {
            'modality':     args.modality,
            'K':            args.init_K,
            'D':            args.D,
            'downsampling': args.downsampling,
            'beta':         args.beta,
            'decay':        args.decay,
            'residual':     False,
            'num_quantizers': 1,
        }

        print(f"[Phase 1] Training baseline  K={args.init_K}  D={args.D}  "
              f"downsampling={args.downsampling}  epochs={args.init_epochs}")
        print(f"          Checkpoint → {baseline_path}\n")

        baseline_model = VQVAE(
            modality=args.modality, K=args.init_K, D=args.D,
            downsampling=args.downsampling, beta=args.beta, decay=args.decay,
        ).to(device)

        optimizer = optim.AdamW(baseline_model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_epoch, best_val = _run_epochs(
            baseline_model, train_loader, val_loader, device,
            optimizer, baseline_path, args.init_epochs, label="[baseline]",
        )

        load_checkpoint(baseline_path, baseline_model, device)
        print(f"\n[Phase 1] complete. Best epoch {best_epoch + 1}, val={best_val:.4f}")
        print("Codebook usage:")
        _, pct = codebook_stats(baseline_model, val_loader, device)

        meta.update({'best_epoch': best_epoch, 'best_val': best_val, 'codebook_usage': pct})
        save_metadata(baseline_path.replace('.pth', '_meta.json'), meta)
        del baseline_model

    # ── Phase 2: collect latents + k-means++ ─────────────────────────────────

    print(f"\n[Phase 2] Collecting latents from baseline encoder...")
    baseline = VQVAE(
        modality=meta['modality'], K=meta['K'], D=meta['D'],
        downsampling=meta['downsampling'], beta=meta['beta'], decay=meta['decay'],
    ).to(device)
    load_checkpoint(baseline_path, baseline, device)

    latents = _collect_latents(baseline, train_loader, device)
    print(f"  {latents.shape[0]:,} vectors of dim {latents.shape[1]}")

    print(f"  Fitting k-means++ → new_K={args.new_K} ...")
    centers, _ = kmeans_plusplus(latents, n_clusters=args.new_K, random_state=42)
    print("  done.\n")

    # ── Phase 3: build refitted model ─────────────────────────────────────────

    model = VQVAE(
        modality=meta['modality'], K=args.new_K, D=meta['D'],
        downsampling=meta['downsampling'], beta=meta['beta'], decay=meta['decay'],
    ).to(device)

    model.encoder.load_state_dict(baseline.encoder.state_dict())
    model.decoder.load_state_dict(baseline.decoder.state_dict())
    _seed_codebook(model.vq, torch.from_numpy(centers), device)
    del baseline

    print(f"[Phase 3] Refit  new_K={args.new_K}  epochs={args.refit_epochs}")
    print(f"          Checkpoint → {args.out}\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_epoch, best_val = _run_epochs(
        model, train_loader, val_loader, device,
        optimizer, args.out, args.refit_epochs, label="[ refit ]",
    )

    # ── Final evaluation ──────────────────────────────────────────────────────

    print(f"\n[Phase 3] complete. Best epoch {best_epoch + 1}, val={best_val:.4f}")
    print("Codebook usage:")
    load_checkpoint(args.out, model, device)
    _, pct = codebook_stats(model, val_loader, device)

    meta_out = args.out.replace('.pth', '_meta.json')
    save_metadata(meta_out, {
        **meta,
        'K':            args.new_K,
        'baseline':     baseline_path,
        'best_epoch':   best_epoch,
        'best_val':     best_val,
        'codebook_usage': pct,
    })
    print(f"\nMetadata → {meta_out}")


if __name__ == "__main__":
    main()
