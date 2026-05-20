"""Visualization helpers for segmentation and MRI batches."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Optional
from torch import Tensor

_SEG_CMAP = ListedColormap(['#000000', '#ff0000', '#00ff00', '#0000ff'])


def show_seg_batch(batch: Tensor, title: str = '', n: int = 8):
    """Displays n samples from a 4-channel one-hot segmentation batch."""
    n = min(n, batch.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(5, 2 * n))
    fig.suptitle(title, fontsize=13)
    for ax in axes.flat:
        ax.set_axis_off()
    for i in range(n):
        for c in range(4):
            axes[i, c].imshow(batch[i, c].cpu(), cmap='gray', vmin=0, vmax=1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def show_mri_batch(batch: Tensor, title: str = '', n: int = 8):
    """Displays n samples from a grayscale MRI batch."""
    n = min(n, batch.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    fig.suptitle(title, fontsize=13)
    axes = [axes] if n == 1 else list(axes)
    for i, ax in enumerate(axes):
        ax.imshow(batch[i].squeeze().cpu(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_errors(true_seg: Tensor, pred_seg: Tensor, title: str = '', n: int = 8):
    """Shows ground truth, prediction, and pixel error map side by side."""
    n = min(n, true_seg.shape[0])

    true_seg = torch.argmax(true_seg,     dim=1).detach().cpu()
    pred_seg = torch.argmax(pred_seg, dim=1).detach().cpu()

    error = (true_seg != pred_seg)*1
    fig, axes = plt.subplots(n, 3, figsize=(8, 3 * n))
    fig.suptitle(title, fontsize=13)
    for i in range(n):
        axes[i, 0].imshow(true_seg[i].cpu(), cmap=_SEG_CMAP)
        axes[i, 1].imshow(pred_seg[i].cpu(), cmap=_SEG_CMAP)
        axes[i, 2].imshow(error[i].cpu(), cmap='magma')
        for ax in axes[i]:
            ax.axis('off')
    for j, label in enumerate(['Ground Truth', 'Prediction', 'Errors']):
        axes[0, j].set_title(label, fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_losses(train: list, val: list, title: str = 'Training curves'):
    plt.figure(figsize=(10, 4))
    plt.plot(train, label='Train')
    plt.plot(val,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(title); plt.yscale('log')
    plt.legend(); plt.grid(); plt.tight_layout(); plt.show()


def plot_index_histograms(
    hists_per_frame: List[np.ndarray],
    K: int,
    patient_counts: Optional[List[int]] = None,
    title: str = 'Codebook index usage per frame',
    save_path: Optional[str] = None,
):
    """
    Bar chart of codebook index usage, one subplot per frame number.

    For each spatial position k, shows a bar plot with K bars:
        x-axis — codebook index  (0 … K-1)
        y-axis — total count across all patients at that frame position

    Args:
        hists_per_frame : list of length n_frames, each np.ndarray [K].
        K               : codebook size.
        patient_counts  : optional list of ints — patients contributing to each frame.
        title           : figure suptitle.
        save_path       : save figure here if given, else display.
    """
    n = len(hists_per_frame)
    n_cols = min(2, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    axes_flat = np.array(axes).reshape(-1) if n > 1 else [axes]
    x = np.arange(K)

    for k, (hist, ax) in enumerate(zip(hists_per_frame, axes_flat)):
        ax.bar(x, hist, width=1.0, color='steelblue', linewidth=0)
        n_pat = f'  n={patient_counts[k]}' if patient_counts is not None else ''
        ax.set_title(f'frame {k}{n_pat}', fontsize=9)
        ax.set_xlim(0, K)
        ax.tick_params(labelsize=7)
        active = int((hist > 0).sum())
        ax.text(0.98, 0.95, f'{active}/{K} active',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=7, color='tomato')

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.supxlabel('Codebook index', fontsize=10)
    fig.supylabel('Count', fontsize=10)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved → {save_path}")
    else:
        plt.show()
