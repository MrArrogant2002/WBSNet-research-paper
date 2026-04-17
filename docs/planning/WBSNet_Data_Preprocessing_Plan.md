# WBSNet Data Preprocessing Plan
> **Purpose:** This document guides an agent to create `data_preprocessing.ipynb` — a Kaggle notebook that preprocesses all four WBSNet datasets, generates boundary ground truths, and saves a final structured dataset ready for training.

---

## 1. OVERVIEW

### Goal
Take raw downloaded datasets → preprocess → save one unified structured directory → download as zip → upload to Kaggle Datasets → use as input in all future training notebooks.

### Input Datasets (uploaded to Kaggle as separate datasets)
| Dataset | Raw Structure | Role in WBSNet |
|---|---|---|
| Kvasir-SEG | `images/`, `masks/` | Train + Val (polyp) |
| CVC-ClinicDB | `Original/`, `Ground Truth/` | Train + Val (polyp) |
| CVC-ColonDB | `images/`, `masks/` | Test only (cross-dataset generalization) |
| ISIC 2018 Task 1 | `ISIC2018_Task1-2_Training_Input/`, `ISIC2018_Task1_Training_GroundTruth/`, `ISIC2018_Task1_Test_Input/`, `ISIC2018_Task1_Test_GroundTruth/` | Train + Test (skin lesion) |

---

## 2. FINAL OUTPUT STRUCTURE

The notebook must produce exactly this directory structure:

```
WBSNet_Dataset/
├── kvasir/
│   ├── train/
│   │   ├── images/        ← .png, 352×352, RGB
│   │   ├── masks/         ← .png, 352×352, binary {0,255}
│   │   └── boundaries/    ← .png, 352×352, binary {0,255} (Sobel GT)
│   └── val/
│       ├── images/
│       ├── masks/
│       └── boundaries/
│
├── cvc_clinicdb/
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── boundaries/
│   └── val/
│       ├── images/
│       ├── masks/
│       └── boundaries/
│
├── cvc_colondb/
│   └── test/
│       ├── images/
│       ├── masks/
│       └── boundaries/
│
├── isic2018/
│   ├── train/
│   │   ├── images/
│   │   ├── masks/
│   │   └── boundaries/
│   ├── val/
│   │   ├── images/
│   │   ├── masks/
│   │   └── boundaries/
│   └── test/
│       ├── images/
│       ├── masks/
│       └── boundaries/
│
└── dataset_info.json      ← metadata: split sizes, image stats
```

---

## 3. PREPROCESSING STEPS (Per Dataset)

### Step 1: Image Resize
- Resize all images to **352 × 352** using `cv2.resize` with `INTER_LINEAR` interpolation.
- Resize all masks to **352 × 352** using `cv2.resize` with `INTER_NEAREST` interpolation (preserves binary values).

### Step 2: Mask Binarization
- Convert masks to binary: pixel values → **0 or 255**.
- Threshold: any pixel > 127 → 255, else 0.
- Save as single-channel PNG (grayscale).

### Step 3: Image Normalization (Save as-is)
- Save images as **uint8 RGB PNG** (do NOT normalize to float — normalization happens in the DataLoader during training).

### Step 4: Boundary Ground Truth Generation
For every mask, generate a boundary map using Sobel edge detection:

```python
import cv2
import numpy as np

def generate_boundary_gt(mask_path, save_path, dilate=True):
    """
    Generate boundary GT from binary mask using Sobel operator.
    
    Args:
        mask_path: path to binary mask PNG (values 0 or 255)
        save_path: path to save boundary PNG
        dilate: whether to dilate boundary by 1 iteration (recommended)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_float = (mask / 255.0).astype(np.float32)
    
    sobelx = cv2.Sobel(mask_float, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask_float, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobelx**2 + sobely**2)
    edge = (edge > 0).astype(np.uint8) * 255
    
    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
    
    cv2.imwrite(save_path, edge)
```

**Important:** Apply this to ALL splits (train, val, test) — boundary GT is needed for supervision during training AND for HD95 evaluation during testing.

### Step 5: Train/Val Split
Apply splits only to datasets that don't have predefined splits:

| Dataset | Split Strategy |
|---|---|
| Kvasir-SEG (1000 images) | 880 train / 120 val — **use fixed random seed=42, sorted filenames** |
| CVC-ClinicDB (612 images) | 550 train / 62 val — **use fixed random seed=42, sorted filenames** |
| CVC-ColonDB (380 images) | 380 test only — no split |
| ISIC 2018 | Use official train (2594) / test (1000) split. Create val by taking last 260 of train (10%) with seed=42 |

---

## 4. DATASET-SPECIFIC NOTES

### Kvasir-SEG
- Raw structure: `kvasir-seg/images/*.jpg`, `kvasir-seg/masks/*.jpg`
- Images and masks share the same filename (e.g., `cju0qkwl9lqxq0801l0adwq1n.jpg`)
- Convert JPG → PNG during save
- Total: 1000 image-mask pairs

### CVC-ClinicDB
- Raw structure varies by source. Common structures:
  - Option A: `Original/*.png`, `Ground Truth/*.png`
  - Option B: `PNG/Original/`, `PNG/Ground Truth/`
- Check actual folder names at runtime and handle both
- Images are PNG, masks are PNG
- Total: 612 image-mask pairs
- Filenames are numeric (e.g., `1.png`, `2.png`)

### CVC-ColonDB
- Raw structure: `images/*.png`, `masks/*.png`
- Total: 380 image-mask pairs
- Used ONLY as test set — no train split needed

### ISIC 2018 Task 1
- Training images: `ISIC2018_Task1-2_Training_Input/ISIC_*.jpg`
- Training masks: `ISIC2018_Task1_Training_GroundTruth/ISIC_*_segmentation.png`
- Test images: `ISIC2018_Task1-2_Test_Input/ISIC_*.jpg`
- Test masks: `ISIC2018_Task1_Test_GroundTruth/ISIC_*_segmentation.png`
- Mask filename pattern: image `ISIC_0024306.jpg` → mask `ISIC_0024306_segmentation.png`
- Masks are already binary (0 or 255)
- Total train: 2594, test: 1000

---

## 5. NOTEBOOK STRUCTURE

The agent must create `data_preprocessing.ipynb` with these cells in order:

### Cell 1: Install & Imports
```python
# Install required packages
!pip install opencv-python-headless tqdm

import os
import cv2
import numpy as np
import json
import shutil
from tqdm import tqdm
import random
from pathlib import Path
```

### Cell 2: Configuration
```python
# ── EDIT THESE PATHS TO MATCH YOUR KAGGLE INPUT PATHS ──
CONFIG = {
    "kvasir_root":      "/kaggle/input/kvasir-seg/kvasir-seg",
    "clinicdb_root":    "/kaggle/input/cvc-clinicdb",        # adjust if needed
    "colondb_root":     "/kaggle/input/cvc-colondb/CVC-ColonDB",
    "isic_train_img":   "/kaggle/input/isic-2018-task1/ISIC2018_Task1-2_Training_Input",
    "isic_train_mask":  "/kaggle/input/isic-2018-task1/ISIC2018_Task1_Training_GroundTruth",
    "isic_test_img":    "/kaggle/input/isic-2018-task1/ISIC2018_Task1-2_Test_Input",
    "isic_test_mask":   "/kaggle/input/isic-2018-task1/ISIC2018_Task1_Test_GroundTruth",
    "output_root":      "/kaggle/working/WBSNet_Dataset",
    "image_size":       352,
    "seed":             42,
    # Kvasir split
    "kvasir_train":     880,
    "kvasir_val":       120,
    # CVC-ClinicDB split
    "clinicdb_train":   550,
    "clinicdb_val":     62,
    # ISIC val fraction
    "isic_val_frac":    0.10,
}
```

### Cell 3: Utility Functions
```python
# resize_image(), resize_mask(), binarize_mask(), generate_boundary_gt()
# save_image(), make_dirs(), get_sorted_files()
```

### Cell 4: Process Kvasir-SEG
```python
# Load all image-mask pairs
# Sort by filename for reproducibility
# Split: first 880 → train, last 120 → val
# For each split: resize → binarize → save image/mask/boundary
```

### Cell 5: Process CVC-ClinicDB
```python
# Auto-detect folder structure (handle both naming variants)
# Split: first 550 → train, last 62 → val
# For each split: resize → binarize → save image/mask/boundary
```

### Cell 6: Process CVC-ColonDB
```python
# All 380 → test only
# resize → binarize → save image/mask/boundary
```

### Cell 7: Process ISIC 2018
```python
# Train: 2594 total → shuffle with seed=42 → first 90% train, last 10% val
# Test: 1000 official test images
# Match image to mask using filename pattern (strip _segmentation suffix)
# resize → binarize → save image/mask/boundary
```

### Cell 8: Generate dataset_info.json
```python
# Count files in each split directory
# Record: dataset name, split, n_images, image_size, boundary_dilated
# Save as WBSNet_Dataset/dataset_info.json
```

### Cell 9: Verification
```python
# For each dataset/split:
#   - Print count of images, masks, boundaries
#   - Assert counts match
#   - Load 1 random sample, display image + mask + boundary side by side
#   - Print min/max pixel values of mask (must be 0 and 255)
#   - Print min/max pixel values of boundary (must be 0 and 255)
```

### Cell 10: Zip and Save
```python
import shutil
shutil.make_archive(
    '/kaggle/working/WBSNet_Dataset',  # output zip name
    'zip',
    '/kaggle/working',                  # root dir
    'WBSNet_Dataset'                    # directory to zip
)
print("Done! Download WBSNet_Dataset.zip from Kaggle output.")
```

---

## 6. VERIFICATION CHECKLIST

After running the notebook, verify:

- [ ] Total files per dataset match expected counts (see table below)
- [ ] All images are 352×352×3 uint8 RGB PNG
- [ ] All masks are 352×352 uint8 grayscale PNG with only values {0, 255}
- [ ] All boundaries are 352×352 uint8 grayscale PNG with only values {0, 255}
- [ ] image count == mask count == boundary count for every split
- [ ] `dataset_info.json` exists and is valid JSON
- [ ] Zip file is created at `/kaggle/working/WBSNet_Dataset.zip`

### Expected File Counts

| Dataset | Split | Images | Masks | Boundaries |
|---|---|---|---|---|
| kvasir | train | 880 | 880 | 880 |
| kvasir | val | 120 | 120 | 120 |
| cvc_clinicdb | train | 550 | 550 | 550 |
| cvc_clinicdb | val | 62 | 62 | 62 |
| cvc_colondb | test | 380 | 380 | 380 |
| isic2018 | train | ~2334 | ~2334 | ~2334 |
| isic2018 | val | ~260 | ~260 | ~260 |
| isic2018 | test | 1000 | 1000 | 1000 |

---

## 7. AFTER DOWNLOADING

1. Download `WBSNet_Dataset.zip` from Kaggle output
2. Upload it to **Kaggle Datasets** (New Dataset → upload zip)
3. Name it: `wbsnet-processed-dataset`
4. In all future training notebooks, attach it as input → available at `/kaggle/input/wbsnet-processed-dataset/WBSNet_Dataset/`

---

## 8. IMPORTANT NOTES FOR AGENT

- **Do not normalize images to float** — keep as uint8. Normalization (mean/std ImageNet) is done in the PyTorch DataLoader.
- **Use `cv2.INTER_NEAREST` for masks** — never use linear interpolation on binary masks (creates intermediate gray values).
- **Use `cv2.INTER_LINEAR` for images** — standard for RGB images.
- **Fixed seed=42 everywhere** — critical for reproducibility across experiments.
- **Sort filenames before splitting** — ensures the same split every run regardless of filesystem order.
- **Boundary dilation=1 iteration** — makes boundaries 3px wide, more stable for supervision at downsampled resolutions.
- **Save all files as PNG** — even if input is JPG (avoids lossy compression on masks).
- **Handle missing mask files gracefully** — print a warning and skip, don't crash.
