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


def plot_codebook_per_slice(
    per_slice_hists: List[np.ndarray],
    K: int,
    patient_counts: Optional[List[int]] = None,
    title: str = 'Codebook usage per slice position',
    save_path: Optional[str] = None,
):
    """
    Visualises codebook index usage broken down by anatomical slice position.

    Each element of per_slice_hists is a [K] array counting how many times each
    codebook entry was selected across all patients at that slice position.
    Slices are ordered as loaded (typically apex → base or base → apex).

    Args:
        per_slice_hists : list of length n_slices, each a np.ndarray of shape [K].
        K               : total codebook size (for reference lines).
        patient_counts  : optional list of length n_slices — number of patients
                          that contributed to each position (shown as annotations).
        title           : figure title.
        save_path       : if given, saves the figure to this path instead of showing.

    Layout (3 columns):
        Left   — heatmap:  log(1 + count) per slice × codebook index.
        Centre — active codes per slice: how many indices have count > 0.
        Right  — entropy per slice: Shannon entropy of the code distribution
                 (high = uniform use, low = few dominant codes).
    """
    n_slices = len(per_slice_hists)
    matrix = np.stack(per_slice_hists)           # [n_slices, K]

    # ── derived statistics ─────────────────────────────────────────────────────
    active = (matrix > 0).sum(axis=1)            # [n_slices]  int

    # Shannon entropy H = -sum(p * log2(p)), ignoring zero counts
    probs = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p  = np.where(probs > 0, np.log2(probs), 0.0)
    entropy = -(probs * log_p).sum(axis=1)       # [n_slices]

    # ── figure ────────────────────────────────────────────────────────────────
    fig_h = max(5, n_slices * 0.45)
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_h),
                             gridspec_kw={'width_ratios': [3, 1, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    slice_labels = [f"s{i}" for i in range(n_slices)]

    # ── left: usage heatmap ────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(np.log1p(matrix), aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='upper')
    ax.set_xlabel('Codebook index', fontsize=11)
    ax.set_ylabel('Slice position', fontsize=11)
    ax.set_yticks(range(n_slices))
    ax.set_yticklabels(slice_labels, fontsize=8)
    ax.set_title('log(1 + count)  [slice × code]', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── centre: active codes per slice ────────────────────────────────────────
    ax = axes[1]
    bars = ax.barh(range(n_slices), active, color='steelblue', height=0.7)
    ax.axvline(K, color='tomato', linestyle='--', linewidth=1.2,
               label=f'K={K}')
    ax.set_xlabel('Active codes', fontsize=11)
    ax.set_xlim(0, K * 1.05)
    ax.set_yticks(range(n_slices))
    ax.set_yticklabels(slice_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title('Active codes / slice', fontsize=11)
    ax.legend(fontsize=9)

    # annotate patient counts if provided
    if patient_counts is not None:
        for i, (bar, cnt) in enumerate(zip(bars, patient_counts)):
            ax.text(bar.get_width() + K * 0.01, i,
                    f'n={cnt}', va='center', fontsize=7, color='gray')

    # ── right: entropy per slice ───────────────────────────────────────────────
    ax = axes[2]
    ax.barh(range(n_slices), entropy, color='darkorange', height=0.7)
    max_entropy = np.log2(K)
    ax.axvline(max_entropy, color='tomato', linestyle='--', linewidth=1.2,
               label=f'log₂(K)={max_entropy:.1f}')
    ax.set_xlabel('Shannon entropy (bits)', fontsize=11)
    ax.set_xlim(0, max_entropy * 1.05)
    ax.set_yticks(range(n_slices))
    ax.set_yticklabels(slice_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title('Code entropy / slice', fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved → {save_path}")
    else:
        plt.show()


