#!/usr/bin/env python3
"""
Plot per-frame codebook index usage histograms from an encoded_latents/ directory.

Usage:
    python plot_index_histograms.py --latents_dir encoded_latents/
    python plot_index_histograms.py --latents_dir encoded_latents/ --fig results/hist.png
"""

import os
import argparse
import json
from collections import defaultdict

import numpy as np
from vq_acdc.utils.viz import plot_index_histograms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_dir", default="encoded_latents")
    p.add_argument("--fig",         default=None, help="Save figure here instead of displaying")
    return p.parse_args()


def main():
    args = parse_args()

    with open(os.path.join(args.latents_dir, 'metadata.json')) as f:
        meta = json.load(f)

    K = meta['K']

    pos_hists    = defaultdict(lambda: np.zeros(K, dtype=np.int64))
    pos_patients = defaultdict(set)

    for split in ('training', 'testing'):
        split_dir = os.path.join(args.latents_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith('.npz'):
                continue
            pid  = fname.replace('patient', '').replace('.npz', '')
            data = np.load(os.path.join(split_dir, fname), allow_pickle=True)
            for i in range(len(data['indices'])):
                k = int(data['slice_pos'][i])
                pos_hists[k]    += np.bincount(data['indices'][i], minlength=K)
                pos_patients[k].add(pid)

    max_pos        = max(pos_hists) + 1
    frame_hists    = [pos_hists[k]          for k in range(max_pos)]
    patient_counts = [len(pos_patients[k])  for k in range(max_pos)]

    plot_index_histograms(
        frame_hists,
        K=K,
        patient_counts=patient_counts,
        title=f"Codebook index usage per frame  |  {meta['modality']}  K={K}",
        save_path=args.fig,
    )


if __name__ == "__main__":
    main()
