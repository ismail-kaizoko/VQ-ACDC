"""
ACDC cardiac dataset loader and transforms.

Supports two modalities:
  - 'SEG': 2D segmentation label maps (4 classes: background, RV, MYO, LV)
  - 'MRI': 2D grayscale cardiac MRI slices

For each patient, ED (end-diastole) and ES (end-systole) frames are extracted,
cropped to the heart ROI using the segmentation bounding box, and returned as
a flat list of 2D slices.
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchio as tio
import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)


# ── Bounding box helpers ───────────────────────────────────────────────────────

def _get_bbox(seg: torch.Tensor) -> torch.Tensor:
    """Tightest axis-aligned box over all non-zero pixels across D slices."""
    boxes = []
    for mask in (seg > 0).float():
        y, x = torch.where(mask)
        if len(y) == 0:
            continue
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    t = torch.tensor(boxes)
    return torch.tensor([t[:, 0].min(), t[:, 1].min(), t[:, 2].max(), t[:, 3].max()])


def _to_square(bbox: list) -> list:
    """Pads the shorter side of the bounding box to make it square."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    d = abs(w - h)
    lo, hi = d // 2, d - d // 2
    if w > h:
        y1 -= lo; y2 += hi
    else:
        x1 -= lo; x2 += hi
    return [x1, y1, x2, y2]


def _crop_to_roi(seg: torch.Tensor, img: torch.Tensor = None, margin: int = 10) -> torch.Tensor:
    """
    Crops `img` (or `seg` if img is None) to the heart ROI derived from `seg`.
    `seg` is [D, H, W], output is [D, H', W'].
    """
    x1, y1, x2, y2 = _to_square(_get_bbox(seg).tolist())
    x1, y1 = int(x1) - margin, int(y1) - margin
    x2, y2 = int(x2) + margin, int(y2) + margin
    target = seg if img is None else img
    return TF.crop(target, y1, x1, y2 - y1, x2 - x1)


# ── Patient loading ────────────────────────────────────────────────────────────

def _load_patient(path: str, modality: str) -> list:
    """Returns a list of 2D tensors for one patient (ED + ES frames)."""
    if modality not in ('SEG', 'MRI'):
        raise ValueError(f"modality must be 'SEG' or 'MRI', got '{modality}'")

    files = os.listdir(path)
    gt_files = [f for f in files if f.endswith('_gt.nii.gz')]
    frames = sorted(set(f.split('_')[1].split('.')[0] for f in gt_files))
    ed_frame, es_frame = frames[0], frames[-1]

    pid = path[-3:]
    ed_gt = tio.LabelMap(os.path.join(path, f"patient{pid}_{ed_frame}_gt.nii.gz"))
    es_gt = tio.LabelMap(os.path.join(path, f"patient{pid}_{es_frame}_gt.nii.gz"))

    def _to_slices(gt_vol, img_vol=None):
        seg = gt_vol.data.squeeze(0).permute(2, 0, 1)   # [D, H, W]
        img = img_vol.data.squeeze(0).permute(2, 0, 1).float() if img_vol else None
        return list(_crop_to_roi(seg, img).unbind(0))    # list of [H, W]

    if modality == 'SEG':
        return _to_slices(ed_gt) + _to_slices(es_gt)

    # MRI: crop using the segmentation mask but return MRI intensities
    ed_mri = tio.LabelMap(os.path.join(path, f"patient{pid}_{ed_frame}.nii.gz"))
    es_mri = tio.LabelMap(os.path.join(path, f"patient{pid}_{es_frame}.nii.gz"))
    return _to_slices(ed_gt, ed_mri) + _to_slices(es_gt, es_mri)


def load_dataset(path: str, modality: str = 'SEG') -> list:
    """Loads all 2D slices from an ACDC split directory (training/ or testing/)."""
    patients = sorted(p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p)))
    slices = []
    for p in patients:
        slices.extend(_load_patient(os.path.join(path, p), modality))
    return slices


# ── Dataset ───────────────────────────────────────────────────────────────────

class ACDCDataset(Dataset):
    def __init__(self, slices: list, transform=None):
        self.slices = slices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.slices[idx].unsqueeze(0)          # [1, H, W]
        return self.transform(item) if self.transform else item


# ── Transforms ────────────────────────────────────────────────────────────────

class OneHotEncode:
    """Converts a [1, H, W] label map to a [C, H, W] one-hot float tensor."""
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x.squeeze(0).long(), self.num_classes).permute(2, 0, 1).float()


class PercentileClip:
    """Clips pixel intensities to the [lo, hi] percentile range."""
    def __init__(self, lo: float = 1.0, hi: float = 99.0):
        self.lo, self.hi = lo / 100.0, hi / 100.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lo = torch.quantile(x.reshape(-1), self.lo)
        hi = torch.quantile(x.reshape(-1), self.hi)
        return torch.clamp(x, lo.item(), hi.item())


class MinMaxNormalize:
    """Linearly scales a tensor to [out_min, out_max]."""
    def __init__(self, out_min: float = 0.0, out_max: float = 1.0):
        self.out_min, self.out_max = out_min, out_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8) * (self.out_max - self.out_min) + self.out_min
