"""Visualization helpers for segmentation and MRI batches."""

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
