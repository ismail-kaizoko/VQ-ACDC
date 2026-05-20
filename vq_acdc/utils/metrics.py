import torch
import torch.nn.functional as F
from torch import Tensor


def dice_score(targets: Tensor, logits: Tensor, smooth: float = 1e-6) -> Tensor:
    """
    Multi-class Dice score averaged over batch and channels.

    Args:
        targets : one-hot ground truth  [B, C, H, W]
        logits  : raw model output      [B, C, H, W]
        smooth  : numerical stability constant

    Returns:
        Scalar mean Dice score in [0, 1].
    """
    probs = F.softmax(logits, dim=1)
    preds = F.one_hot(probs.argmax(dim=1), num_classes=targets.shape[1])
    preds = preds.permute(0, 3, 1, 2).float()   # [B, C, H, W]

    p = preds.contiguous().view(preds.shape[0], preds.shape[1], -1)
    t = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)

    intersection = (p * t).sum(dim=2)
    union = p.sum(dim=2) + t.sum(dim=2)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean()


def dice_loss(targets: Tensor, logits: Tensor) -> Tensor:
    return 1.0 - dice_score(targets, logits)
