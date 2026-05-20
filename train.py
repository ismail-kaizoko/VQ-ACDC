#!/usr/bin/env python3
"""
Train a VQ-VAE on the ACDC cardiac dataset.

Usage examples:
    # Segmentation, K=512 codebook, downsampling ×4
    python train.py --modality SEG --K 512 --D 64 --downsampling 4 \
                    --epochs 100 --out checkpoints/seg_512.pth

    # MRI, with Residual VQ (2 codebooks)
    python train.py --modality MRI --K 256 --D 64 --residual \
                    --num_quantizers 2 --out checkpoints/mri_rq256.pth

Set ACDC_DATA_PATH env variable to override the default dataset path.
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import Compose, Resize, InterpolationMode
from torch.utils.data import DataLoader
from tqdm import tqdm

from vq_acdc.data import load_dataset, ACDCDataset, OneHotEncode, PercentileClip, MinMaxNormalize
from vq_acdc.models import VQVAE
from vq_acdc.utils.io import save_checkpoint, load_checkpoint, save_metadata
from vq_acdc.utils.training import evaluate, codebook_stats


_DEFAULT_DATA_PATH = "/home/ids/ihamdaoui-21/ACDC/database"


def parse_args():
    p = argparse.ArgumentParser(description="Train VQ-VAE on ACDC")

    # Data
    p.add_argument("--modality",     required=True, choices=['SEG', 'MRI'],
                   help="Data modality: 'SEG' for segmentations, 'MRI' for scans")
    p.add_argument("--data_path",    default=os.environ.get("ACDC_DATA_PATH", _DEFAULT_DATA_PATH),
                   help="Root of the ACDC database (contains training/ and testing/)")
    p.add_argument("--img_size",     type=int, default=128, help="Resize images to (img_size × img_size)")

    # Training
    p.add_argument("--out",          required=True, help="Output checkpoint path (.pth)")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=5e-4)

    # Model
    p.add_argument("--K",            type=int, default=512, help="Codebook size")
    p.add_argument("--D",            type=int, default=64,  help="Embedding dimension")
    p.add_argument("--downsampling", type=int, default=4, choices=[2, 4, 8])
    p.add_argument("--beta",         type=float, default=0.25, help="Commitment loss weight")
    p.add_argument("--decay",        type=float, default=0.8,  help="EMA decay for codebook")
    p.add_argument("--residual",     action='store_true', help="Use Residual VQ (RQ-VAE)")
    p.add_argument("--num_quantizers", type=int, default=2, help="Stacked quantizers (residual only)")

    return p.parse_args()


def build_loaders(args):
    sz = (args.img_size, args.img_size)
    if args.modality == 'SEG':
        tfm = Compose([Resize(sz, interpolation=InterpolationMode.NEAREST), OneHotEncode()])
    else:
        tfm = Compose([Resize(sz, interpolation=InterpolationMode.NEAREST),
                       PercentileClip(), MinMaxNormalize()])

    train_data = load_dataset(os.path.join(args.data_path, "training"), args.modality)
    test_data  = load_dataset(os.path.join(args.data_path, "testing"),  args.modality)

    train_loader = DataLoader(ACDCDataset(train_data, tfm), args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ACDCDataset(test_data,  tfm), args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Modality: {args.modality}  |  K={args.K}  D={args.D}  "
          f"downsampling={args.downsampling}  residual={args.residual}")
    print(f"Output: {args.out}\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    train_loader, val_loader = build_loaders(args)

    model = VQVAE(
        modality=args.modality, K=args.K, D=args.D,
        downsampling=args.downsampling, beta=args.beta, decay=args.decay,
        residual=args.residual, num_quantizers=args.num_quantizers,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_losses, val_losses, commit_losses = [], [], []
    best_val, best_epoch = float('inf'), 0

    for epoch in range(args.epochs):
        model.train()
        ep_recon, ep_commit = [], []

        with tqdm(train_loader, desc=f"Epoch {epoch + 1:>3}/{args.epochs}") as pbar:
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

    print(f"\nTraining complete. Best: epoch {best_epoch + 1}, val={best_val:.4f}")
    print("\nCodebook usage on validation set:")
    load_checkpoint(args.out, model, device)
    _, pct = codebook_stats(model, val_loader, device)

    meta_path = args.out.replace('.pth', '_meta.json')
    save_metadata(meta_path, {
        'modality':       args.modality,
        'K':              args.K,
        'D':              args.D,
        'downsampling':   args.downsampling,
        'beta':           args.beta,
        'decay':          args.decay,
        'residual':       args.residual,
        'num_quantizers': args.num_quantizers,
        'best_epoch':     best_epoch,
        'best_val':       best_val,
        'codebook_usage': pct,
    })
    print(f"Metadata saved → {meta_path}")


if __name__ == "__main__":
    main()
