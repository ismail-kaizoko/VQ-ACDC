#!/usr/bin/env python3
"""
Encode all ACDC patient frames through the VQ-VAE encoder and save index sequences.

Each 2D frame (ED or ES, all spatial positions) is encoded to a [h, w] grid of
codebook indices, then flattened to a 1D sequence of h*w tokens. These sequences
are the direct input for downstream transformer training.

Output layout:
    out_dir/
    ├── metadata.json           — K, h, w, seq_len, modality, checkpoint, ...
    ├── global_hist.npy         — [K] index usage counts across all frames/patients
    ├── training/
    │   ├── patient001.npz      — indices [N_frames, h*w], phases, slice_pos
    │   └── ...
    └── testing/
        └── ...

Usage:
    python encode_latents.py --checkpoint checkpoints/seg_refit256.pth
    python encode_latents.py --checkpoint checkpoints/seg_refit256.pth \\
        --split both --out_dir encoded_latents/seg256
"""

import os
import argparse
import json

import numpy as np
import torch
from collections import defaultdict
from torchvision.transforms import Compose, Resize, InterpolationMode

from vq_acdc.data import load_dataset_per_patient, OneHotEncode, PercentileClip, MinMaxNormalize
from vq_acdc.models import VQVAE
from vq_acdc.utils.io import load_checkpoint, load_metadata
from vq_acdc.utils.viz import plot_index_histograms


_DEFAULT_DATA_PATH = "/home/ids/ihamdaoui-21/ACDC/database"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Encode ACDC frames to VQ-VAE index sequences")
    p.add_argument("--checkpoint",  required=True, help="Path to .pth checkpoint")
    p.add_argument("--data_path",   default=os.environ.get("ACDC_DATA_PATH", _DEFAULT_DATA_PATH))
    p.add_argument("--split",       default="both", choices=["train", "test", "both"])
    p.add_argument("--out_dir",     default="encoded_latents")
    p.add_argument("--img_size",    type=int, default=128)
    p.add_argument("--fig",         default=None, help="Save histogram figure to this path")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_transform(modality, img_size):
    sz = (img_size, img_size)
    if modality == 'SEG':
        return Compose([Resize(sz, interpolation=InterpolationMode.NEAREST), OneHotEncode()])
    return Compose([Resize(sz, interpolation=InterpolationMode.NEAREST),
                    PercentileClip(), MinMaxNormalize()])


# ── Core encoding ─────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_patients(model, patients, transform, device, split_name):
    """
    For each patient, encodes every frame (both ED and ES, all spatial positions)
    through the VQ-VAE encoder and collects the flattened index sequence.

    Returns:
        encoded   : list of dicts, one per patient:
                    {
                      'patient':   str,
                      'indices':   np.ndarray [N_frames, h*w]  (int32),
                      'phases':    np.ndarray [N_frames]        (str: 'ED'/'ES'),
                      'slice_pos': np.ndarray [N_frames]        (int: spatial position)
                    }
        hist      : np.ndarray [K]  global index usage counts for this split
    """
    model.eval()
    K = model.K
    hist = np.zeros(K, dtype=np.int64)
    encoded = []

    for patient in patients:
        pid = patient['patient']
        frame_indices = []
        phases        = []
        slice_pos     = []

        for k, (ed_plane, es_plane) in enumerate(patient['slices']):
            for phase_name, plane in (('ED', ed_plane), ('ES', es_plane)):
                # plane: [H, W] → [1, H, W] → transform → [C, H, W] → [1, C, H, W]
                x = transform(plane.unsqueeze(0)).unsqueeze(0).float().to(device)
                idx = model.encode_indices(x)   # [1, h, w]
                if idx.dim() == 4:              # residual VQ: take first codebook
                    idx = idx[..., 0]
                flat = idx.reshape(-1).cpu().numpy().astype(np.int32)   # [h*w]
                frame_indices.append(flat)
                phases.append(phase_name)
                slice_pos.append(k)

        indices_arr = np.stack(frame_indices)   # [N_frames, h*w]
        hist += np.bincount(indices_arr.reshape(-1), minlength=K)

        encoded.append({
            'patient':   pid,
            'indices':   indices_arr,
            'phases':    np.array(phases),
            'slice_pos': np.array(slice_pos, dtype=np.int32),
        })

        n_frames = len(frame_indices)
        seq_len  = frame_indices[0].shape[0]
        print(f"  [{split_name}] patient {pid} : {n_frames} frames × {seq_len} tokens")

    return encoded, hist


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(hist, K):
    active  = int((hist > 0).sum())
    total   = int(hist.sum())
    probs   = hist / (hist.sum() + 1e-12)
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    top10   = np.argsort(hist)[::-1][:10]

    print(f"\n── Global codebook statistics ───────────────────────")
    print(f"  Total tokens    : {total:,}")
    print(f"  Active codes    : {active} / {K}  ({active * 100.0 / K:.1f}%)")
    print(f"  Shannon entropy : {entropy:.2f} bits  (max {np.log2(K):.2f})")
    print(f"  Top-10 codes    : {list(top10)}")
    print(f"  Top-10 counts   : {[int(hist[i]) for i in top10]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    meta_path = args.checkpoint.replace('.pth', '_meta.json')
    meta = load_metadata(meta_path)
    model = VQVAE(
        modality=meta['modality'], K=meta['K'], D=meta['D'],
        downsampling=meta['downsampling'], beta=meta['beta'], decay=meta['decay'],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    h = w = args.img_size // meta['downsampling']

    print(f"Model    : {args.checkpoint}")
    print(f"Modality : {meta['modality']}  K={meta['K']}  D={meta['D']}")
    print(f"Tokens   : {h}×{w} = {h*w} per frame\n")

    transform = _build_transform(meta['modality'], args.img_size)
    os.makedirs(args.out_dir, exist_ok=True)

    global_hist = np.zeros(meta['K'], dtype=np.int64)

    splits = []
    if args.split in ('train', 'both'):
        splits.append(('training', os.path.join(args.data_path, 'training')))
    if args.split in ('test', 'both'):
        splits.append(('testing',  os.path.join(args.data_path, 'testing')))

    for split_name, split_path in splits:
        patients = load_dataset_per_patient(split_path, meta['modality'])
        print(f"Encoding {len(patients)} patients [{split_name}]...")

        out_split = os.path.join(args.out_dir, split_name)
        os.makedirs(out_split, exist_ok=True)

        encoded, hist = encode_patients(model, patients, transform, device, split_name)
        global_hist += hist

        for record in encoded:
            np.savez_compressed(
                os.path.join(out_split, f"patient{record['patient']}.npz"),
                indices   = record['indices'],     # [N_frames, h*w]  int32
                phases    = record['phases'],       # ['ED', 'ES', ...]
                slice_pos = record['slice_pos'],    # [0, 0, 1, 1, ...]
            )

    # save global histogram and metadata
    np.save(os.path.join(args.out_dir, 'global_hist.npy'), global_hist)

    with open(os.path.join(args.out_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'checkpoint':  args.checkpoint,
            'modality':    meta['modality'],
            'K':           meta['K'],
            'D':           meta['D'],
            'downsampling': meta['downsampling'],
            'img_size':    args.img_size,
            'h': h, 'w': w,
            'seq_len':     h * w,
            'split':       args.split,
        }, f, indent=2)

    print_stats(global_hist, meta['K'])
    print(f"\nSaved → {args.out_dir}/")

    # ── per-frame histograms ──────────────────────────────────────────────────
    # accumulate index counts grouped by spatial position across all patients/splits
    pos_hists    = defaultdict(lambda: np.zeros(meta['K'], dtype=np.int64))
    pos_patients = defaultdict(set)

    for split_name, split_path in splits:
        out_split = os.path.join(args.out_dir, split_name)
        for fname in sorted(os.listdir(out_split)):
            if not fname.endswith('.npz'):
                continue
            pid  = fname.replace('patient', '').replace('.npz', '')
            data = np.load(os.path.join(out_split, fname), allow_pickle=True)
            for frame_idx in range(len(data['indices'])):
                k = int(data['slice_pos'][frame_idx])
                pos_hists[k]    += np.bincount(data['indices'][frame_idx], minlength=meta['K'])
                pos_patients[k].add(pid)

    max_pos       = max(pos_hists) + 1
    frame_hists   = [pos_hists[k]          for k in range(max_pos)]
    patient_counts = [len(pos_patients[k]) for k in range(max_pos)]

    plot_index_histograms(
        frame_hists,
        K=meta['K'],
        patient_counts=patient_counts,
        title=f"Codebook index usage per frame  |  {meta['modality']}  K={meta['K']}",
        save_path=args.fig,
    )


if __name__ == "__main__":
    main()
