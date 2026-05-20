"""Shared training utilities: evaluation loop and codebook statistics."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .metrics import dice_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Computes mean validation loss over the loader.
    SEG  → Dice loss  (lower is better, same objective as training)
    MRI  → MSE loss
    """
    model.eval()
    losses = []
    for batch in loader:
        x = batch.float().to(device)
        recon, _, _ = model(x)
        if model.modality == 'SEG':
            losses.append(dice_loss(x, recon).item())
        else:
            losses.append(F.mse_loss(recon, x).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def codebook_stats(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple:
    """
    Accumulates codebook usage histogram over the loader and prints a summary.

    Returns:
        (histogram, usage_pct) where usage_pct is the percentage of active codes.
        For residual VQ, histogram is [num_quantizers, K] and usage_pct is the
        last quantizer's percentage.
    """
    model.eval()
    hist = None
    for batch in loader:
        h = model.codebook_usage(batch.float().to(device))
        hist = h if hist is None else hist + h.to(hist.device)
    model.train()

    hist_np = hist.cpu().numpy()

    if model.residual:
        pct = 0.0
        for i, h in enumerate(hist_np):
            used = int((h > 0).sum())
            pct  = used * 100.0 / model.K
            print(f"  Codebook {i + 1}: {used}/{model.K} used ({pct:.1f}%)")
    else:
        used = int((hist_np > 0).sum())
        pct  = used * 100.0 / model.K
        print(f"  Codebook: {used}/{model.K} used ({pct:.1f}%)")

    return hist_np, pct
