# VQ-ACDC

VQ-VAE applied to the [ACDC cardiac dataset](https://acdc.creatis.insa-lyon.fr/).
Learns a discrete latent representation of 2D cardiac images — either segmentation maps (4 classes: background, RV, myocardium, LV) or raw MRI grayscale slices.

---

## What this project does

The model compresses 2D cardiac images into a discrete codebook of learned vectors (Vector Quantization), then reconstructs them. The key challenge is **codebook collapse**: most codes go unused and only a few dominate. This project tackles it with **Refit** — after an initial training phase, the encoder's latent distribution is collected and a new codebook is seeded via k-means++, then the model is fine-tuned with this better-distributed codebook.

---

## What is implemented

### Data (`vq_acdc/data/acdc.py`)
Loads the ACDC dataset for both ED (end-diastole) and ES (end-systole) cardiac phases. Crops each patient's volume to the heart ROI using the segmentation bounding box, then unpacks the 3D volume into 2D slices. Supports two loading modes: flat (for training) and structured per-patient per-spatial-plane (for analysis). Transforms: one-hot encoding for segmentation, percentile clipping + min-max normalization for MRI.

### Model (`vq_acdc/models/vqvae.py`)
Convolutional VQ-VAE with configurable downsampling factor (×2, ×4, ×8). Supports standard VQ and Residual VQ (stacked codebooks). The encoder and decoder mirror each other; the MRI decoder uses bilinear upsampling to avoid checkerboard artifacts. EMA updates for codebook stability.

### Training (`train.py`)
Trains a VQ-VAE from scratch with a train/val split. Saves the best checkpoint (by validation loss) and a companion JSON metadata file.

### Refit pipeline (`train_with_refit.py`)
Two modes: (1) train a baseline from scratch then refit, or (2) refit an existing checkpoint. The refit step collects encoder latents on the training set, runs k-means++ to find new centroids, seeds the new codebook, and fine-tunes. The new codebook size can be smaller than or equal to the original.

### Codebook analysis (`analyze_codebook.py`)
Computes per-spatial-plane codebook usage histograms: for each of the ~12 short-axis planes composing the 3D cardiac volume, accumulates which codebook entries are selected across all patients (both ED and ES frames). Produces a 3-panel figure: usage heatmap, active codes per plane, and Shannon entropy per plane.

### Utilities (`vq_acdc/utils/`)
- **metrics**: Dice score and Dice loss for segmentation evaluation.
- **training**: Validation loop and codebook utilisation statistics.
- **viz**: Batch visualisation (segmentation and MRI), error maps, training curves, per-plane codebook figure.
- **io**: Checkpoint save/load and JSON metadata helpers.

---

## Results

| Model | Base | Dice ↑ | Codebook usage ↑ |
|-------|------|--------|-----------------|
| 400   | 300  | 94.0%  | 56.25%          |
| 402   | 301  | 96.40% | 61.70%          |
| 404   | 302  | 96.44% | 89.84%          |
