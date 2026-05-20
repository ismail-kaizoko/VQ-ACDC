#!/usr/bin/env python3
"""
Per-slice codebook usage analysis.

The ACDC 3D cardiac volume is a stack of ~12 short-axis 2D planes (apex → base).
For each spatial position k in this stack, this script accumulates a histogram
of codebook indices over all patients that have a plane at position k.

Both ED and ES frames are used — they represent the same anatomical plane at
different cardiac phases, so both contribute statistics for position k.

Patients with fewer planes naturally don't contribute to deeper positions.
Results are saved to a .npz file and visualised with plot_codebook_per_slice().

Usage:
    python analyze_codebook.py --checkpoint checkpoints/seg_refit256.pth

    # both data splits, save figure
    python analyze_codebook.py --checkpoint checkpoints/seg_refit256.pth \\
        --split both --fig results/codebook_per_slice.png
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, InterpolationMode

from vq_acdc.data import load_dataset_per_patient, OneHotEncode, PercentileClip, MinMaxNormalize
from vq_acdc.models import VQVAE
from vq_acdc.utils.io import load_checkpoint, load_metadata
from vq_acdc.utils.viz import plot_codebook_per_slice


_DEFAULT_DATA_PATH = "/home/ids/ihamdaoui-21/ACDC/database"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Per-slice codebook usage analysis")
    p.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint")
    p.add_argument("--data_path",   default=os.environ.get("ACDC_DATA_PATH", _DEFAULT_DATA_PATH))
    p.add_argument("--split",       default="both", choices=["train", "test", "both"])
    p.add_argument("--img_size",    type=int, default=128)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--out",         default=None,
                   help="Save histograms to this .npz path (default: next to checkpoint)")
    p.add_argument("--fig",         default=None,
                   help="Save figure to this path instead of displaying it")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_transform(modality: str, img_size: int):
    sz = (img_size, img_size)
    if modality == 'SEG':
        return Compose([Resize(sz, interpolation=InterpolationMode.NEAREST), OneHotEncode()])
    return Compose([Resize(sz, interpolation=InterpolationMode.NEAREST),
                    PercentileClip(), MinMaxNormalize()])


def _load_patients(data_path: str, split: str, modality: str) -> list:
    patients = []
    if split in ('train', 'both'):
        patients += load_dataset_per_patient(os.path.join(data_path, 'training'), modality)
    if split in ('test', 'both'):
        patients += load_dataset_per_patient(os.path.join(data_path, 'testing'),  modality)
    return patients


# ── Core computation ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_per_slice_histograms(
    model: VQVAE,
    patients: list,
    transform,
    device: torch.device,
    batch_size: int,
) -> tuple:
    """
    For each spatial position k (plane in the 3D short-axis stack), accumulates
    a codebook index histogram over all patients and both cardiac phases (ED, ES).

    patient['slices'][k] = [ed_plane_k, es_plane_k]  (raw [H, W] tensors)

    Returns:
        per_slice_hists : list of np.ndarray [K], one per spatial position.
        patient_counts  : list of int — patients contributing to each position.
    """
    model.eval()
    K = model.K

    # slices_by_pos[k] = flat list of [H, W] tensors at spatial position k
    # (both ED and ES, across all patients with at least k+1 planes)
    slices_by_pos   = defaultdict(list)
    patients_by_pos = defaultdict(set)   # for counting unique patients

    for patient in patients:
        pid = patient['patient']
        for k, planes_at_k in enumerate(patient['slices']):
            # planes_at_k = [ed_plane, es_plane]
            slices_by_pos[k].extend(planes_at_k)
            patients_by_pos[k].add(pid)

    max_pos = max(slices_by_pos) + 1
    per_slice_hists = []
    patient_counts  = []

    for k in range(max_pos):
        raw = slices_by_pos[k]          # list of [H, W] tensors
        n_patients = len(patients_by_pos[k])

        # apply transforms: [H,W] → [1,H,W] → [C,H,W]
        batch = torch.stack([transform(sl.unsqueeze(0)) for sl in raw])  # [N, C, H, W]

        # run in mini-batches
        all_indices = []
        for start in range(0, len(batch), batch_size):
            x = batch[start : start + batch_size].float().to(device)
            idx = model.encode_indices(x)   # [B, h, w]  or  [B, h, w, Q] for residual
            if idx.dim() == 4:              # residual VQ: take first codebook
                idx = idx[..., 0]
            all_indices.append(idx.reshape(-1).cpu())

        hist = torch.bincount(torch.cat(all_indices), minlength=K).numpy()
        per_slice_hists.append(hist)
        patient_counts.append(n_patients)

        active = int((hist > 0).sum())
        print(f"  plane {k:>2}:  {n_patients:>3} patients  |  "
              f"{active:>4}/{K} codes active  ({active * 100.0 / K:.1f}%)")

    return per_slice_hists, patient_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────────
    meta_path = args.checkpoint.replace('.pth', '_meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            "Checkpoint must have been created by train.py or train_with_refit.py."
        )
    meta = load_metadata(meta_path)

    model = VQVAE(
        modality=meta['modality'], K=meta['K'], D=meta['D'],
        downsampling=meta['downsampling'], beta=meta['beta'], decay=meta['decay'],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    print(f"Model    : {args.checkpoint}")
    print(f"Modality : {meta['modality']}  K={meta['K']}  D={meta['D']}")
    print(f"Split    : {args.split}\n")

    # ── load structured dataset ───────────────────────────────────────────────
    print("Loading dataset...")
    patients  = _load_patients(args.data_path, args.split, meta['modality'])
    transform = _build_transform(meta['modality'], args.img_size)
    print(f"  {len(patients)} patients  |  "
          f"planes per patient: {len(patients[0]['slices'])} (first patient)\n")

    # ── compute ───────────────────────────────────────────────────────────────
    print("Computing per-plane codebook histograms...")
    per_slice_hists, patient_counts = compute_per_slice_histograms(
        model, patients, transform, device, args.batch_size)

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = args.out or args.checkpoint.replace('.pth', '_slice_hists.npz')
    np.savez(out_path,
             hists=np.stack(per_slice_hists),
             patient_counts=np.array(patient_counts),
             K=meta['K'])
    print(f"\nHistograms saved → {out_path}")

    # ── plot ──────────────────────────────────────────────────────────────────
    title = (f"Codebook usage per spatial plane  |  "
             f"{meta['modality']}  K={meta['K']}  split={args.split}")
    plot_codebook_per_slice(
        per_slice_hists,
        K=meta['K'],
        patient_counts=patient_counts,
        title=title,
        save_path=args.fig,
    )

if __name__ == "__main__":
    main()
