"""Checkpoint save / load and metadata helpers."""

import json
import torch
from torch import nn


def save_checkpoint(path: str, model: nn.Module, epoch: int,
                    train_losses: list, val_losses: list,
                    commit_losses: list, score: float):
    vq = model.vq
    if model.residual:
        codebook = vq.codebooks
        K = vq.codebook_sizes[0]
        D = vq.layers[0].dim
    else:
        codebook = vq.codebook
        K = vq.codebook_size
        D = vq.dim

    torch.save({
        'epoch':         epoch,
        'K':             K,
        'D':             D,
        'model_state_dict': model.state_dict(),
        'score':         score,
        'train_losses':  train_losses,
        'val_losses':    val_losses,
        'commit_losses': commit_losses,
        'codebook':      codebook,
    }, path)


def load_checkpoint(path: str, model: nn.Module,
                    device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt


def save_metadata(path: str, info: dict):
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)


def load_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
