# WBSNet: Complete Architecture Design & Implementation Plan

---

## 1. THE BIG PICTURE — What Are We Building?

**WBSNet** (Wavelet Boundary Skip Network) is a medical image segmentation model that improves how U-Net transfers information from its encoder to its decoder.

### The Problem We're Solving

In a standard U-Net, **skip connections** directly copy encoder features to the decoder. This causes two issues:

1. **Semantic Gap**: Encoder features are low-level (edges, textures) while decoder expects high-level (object understanding). Directly merging them confuses the network.
2. **Boundary Blurring**: During downsampling, fine boundary details are lost. Standard skip connections can't recover them because they pass raw features without distinguishing what's important.

### Our Solution — The WBS Module

Instead of naively passing encoder features through skip connections, we:

1. **Decompose** the encoder feature into frequency components using Wavelet Transform
2. **Enhance** the low-frequency (structure) with channel attention
3. **Sharpen** the high-frequency (boundaries) with boundary-supervised spatial attention
4. **Reconstruct** the enhanced feature and pass it to the decoder

This is like having a smart filter that says: *"Keep the structure clean, make the boundaries sharper, and only pass what the decoder actually needs."*

---

## 2. ARCHITECTURE OVERVIEW

```
Input Image (H × W × 3)
        │
        ▼
Stem + ReLU: H/2 × W/2 × 64  ──── WBS Module 1 ───────────────┐
        │                                                      │
Layer1: H/4 × W/4 × 64    ──── WBS Module 2 ───────────────┐   │
        │                                                   │   │
Layer2: H/8 × W/8 × 128   ──── WBS Module 3 ────────────┐   │   │
        │                                                │   │   │
Layer3: H/16 × W/16 × 256 ──── WBS Module 4 ────────┐    │   │   │
        │                                            │    │   │   │
Layer4 / Bottleneck: H/32 × W/32 × 512              │    │   │   │
        │                                            │    │   │   │
        ▼                                            ▼    ▼   ▼   ▼
┌───────────────────────────────────────────────────────────────────┐
│                            DECODER                               │
│                                                                  │
│ Dec4: Upsample(2×) + Concat(WBS4) + 2×Conv  H/16 × W/16 × 256   │
│ Dec3: Upsample(2×) + Concat(WBS3) + 2×Conv  H/8 × W/8 × 128     │
│ Dec2: Upsample(2×) + Concat(WBS2) + 2×Conv  H/4 × W/4 × 64      │
│ Dec1: Upsample(2×) + Concat(WBS1) + 2×Conv  H/2 × W/2 × 32      │
│ Final Upsample(2×) → H × W × 32                                 │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
Segmentation Head: Conv 1×1 (logits) → logits_seg


LOSS = L_seg(logits_seg, y) + λ · L_bnd({M_bnd^(k)}, edge(y))
     = BCEWithLogitsLoss(logits_seg, y)
       + Dice(sigmoid(logits_seg), y)
       + 0.5 · mean_k BCEWithLogitsLoss(M_bnd^(k), resize(edge(y), k))
```

**Locked topology**:
- The four WBS-refined skip pathways come from `stem/relu`, `layer1`, `layer2`, and `layer3`.
- `layer4` is used only as the bottleneck input and is **not** also used as a skip.
- Because the skip inputs are `176`, `88`, `44`, and `22`, all WBS modules operate on even spatial dimensions and do not need DWT padding in the final design.

---

## 3. WBS MODULE — THE CORE CONTRIBUTION (Detailed Design)

This is the **heart of the paper**. Each WBS module replaces one standard skip connection.

### 3.1 Input

- Encoder feature: `F_enc` with shape `[B, C, H, W]`

### 3.2 Step-by-Step Pipeline

```
F_enc [B, C, H, W]
    │
    ▼
┌──────────────────────────────┐
│   2D Haar Discrete Wavelet   │
│   Transform (DWT)            │
│                              │
│   Decomposes each channel    │
│   into 4 sub-bands:          │
│   LL, LH, HL, HH            │
│   Each: [B, C, H/2, W/2]    │
└──────────┬───────────────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐  ┌──────────────┐
│   LL    │  │ LH, HL, HH   │
│ (Low    │  │ (High         │
│ Freq)   │  │  Frequency)   │
│Structure│  │ Edges/Bounds  │
└────┬────┘  └──────┬────────┘
     │               │
     ▼               ▼
┌─────────┐  ┌───────────────┐
│  LFSA   │  │    HFBA       │
│ Channel │  │ Spatial Attn  │
│ Attn    │  │ + Boundary GT │
│ (SE)    │  │ Supervision   │
└────┬────┘  └───────┬───────┘
     │               │
     ▼               ▼
   LL'             HF'
     │               │
     └───────┬───────┘
             │
             ▼
┌──────────────────────────────┐
│   2D Haar Inverse DWT        │
│   (IDWT)                     │
│                              │
│   Reconstructs from          │
│   LL' + HF' → F_skip'       │
│   Shape: [B, C, H, W]       │
└──────────────────────────────┘
             │
             ▼
    F_skip' → Concatenated with decoder feature
```

### 3.3 Sub-Components

#### A) 2D Haar DWT (No learnable parameters — pure math)

The Haar wavelet is the simplest wavelet. For a 2D feature map, it produces:

- **LL** (low-low): Approximation — captures smooth structure (average of 2×2 blocks)
- **LH** (low-high): Horizontal edges
- **HL** (high-low): Vertical edges  
- **HH** (high-high): Diagonal edges

**Why Haar?** It's computationally free (just additions and subtractions), perfectly invertible, and aligns naturally with boundary detection. For a conference paper, Haar is sufficient. You can test Daubechies-2 in ablation.

**Implementation**: Use `pytorch_wavelets` library or implement manually:

```
LL = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) / 2
LH = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) / 2
HL = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) / 2
HH = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) / 2
```

#### B) LFSA: Low-Frequency Semantic Attention

**Purpose**: Enhance the structural (low-freq) features by recalibrating channel importance. Some channels capture more relevant semantic info than others.

**Architecture** (Squeeze-and-Excitation style):

```
LL [B, C, H/2, W/2]
    │
    ▼
Global Average Pooling → [B, C, 1, 1]
    │
    ▼
FC Layer 1: C → C/r  (r=16, reduction ratio)
    │
    ▼
ReLU
    │
    ▼
FC Layer 2: C/r → C
    │
    ▼
Sigmoid → Channel weights w [B, C, 1, 1]
    │
    ▼
LL' = LL × w  (channel-wise multiplication)
```

**Parameters**: 2 × (C × C/16) = C²/8 per module

#### C) HFBA: High-Frequency Boundary Attention

**Purpose**: Selectively enhance edge/boundary features in the high-frequency sub-bands. Supervised by actual boundary ground truth during training.

**Architecture** (lightweight final version):

```
LH [B,C,H/2,W/2], HL [B,C,H/2,W/2], HH [B,C,H/2,W/2]
    │
    ▼
Concatenate along channel dim → HF [B, 3C, H/2, W/2]
    │
    ▼
Depthwise Conv 3×3 (groups=3C) → Pointwise Conv 1×1 (3C → C)
    │
    ▼
BN → ReLU → HF_reduced [B, C, H/2, W/2]
    │
    ▼
Conv 1×1 (C → 1) → M_bnd_logits
    │
    ▼
sigmoid(M_bnd_logits) → A_bnd [B, 1, H/2, W/2]
    │
    │   During TRAINING: M_bnd_logits is supervised by Boundary GT
    │   L_bnd^(k) = BCEWithLogitsLoss(M_bnd_logits, resize(Sobel(y_mask)))
    │
    ▼
HF' = HF_reduced × A_bnd  (spatial-wise multiplication)
    │
    ▼
Split HF' back into 3 parts for IDWT:
LH' = HF'
HL' = HF'
HH' = HF'
```

**Important design decision**: Instead of keeping LH/HL/HH separate (which wastes parameters), we concatenate, process jointly, and the single output HF' is used for all three high-freq positions in IDWT. This is simpler and works because the attention map is spatial (where are the boundaries?), not directional.

**Alternative (for ablation)**: Keep LH/HL/HH separate with 3 independent Conv1×1 heads — slightly more parameters but direction-aware.

**Parameters**: In the final design, HFBA uses depthwise separable convolution plus a logits head, reducing the overall WBS overhead to about **1.1M parameters total** across all four skip modules.

#### D) 2D Haar IDWT (Inverse — No learnable parameters)

Reconstructs the feature map from the enhanced sub-bands:
```
F_skip' = IDWT(LL', LH', HL', HH')
Shape: [B, C, H, W]  — same as original encoder feature
```

This is the mathematical inverse of the DWT, perfectly reconstructing the spatial resolution.

### 3.4 Parameter Overhead Summary

| Module Level | C | LFSA Params | HFBA Params | Total |
|---|---|---|---|---|
| WBS 1 (Stem/ReLU) | 64 | 512 | ~13K | ~13.5K |
| WBS 2 (Layer1) | 64 | 512 | ~13K | ~13.5K |
| WBS 3 (Layer2) | 128 | 2K | ~50K | ~52K |
| WBS 4 (Layer3) | 256 | 8K | ~200K | ~208K |
| **Total** | | | | **~1.1M** |

**Final choice**: Use the depthwise separable HFBA variant from the start. This is the locked implementation for the paper and keeps WBSNet lightweight relative to the ResNet-34 baseline.

### 3.5 Depthwise Separable HFBA (Lightweight Version — RECOMMENDED)

```
HF [B, 3C, H/2, W/2]
    │
    ▼
Depthwise Conv 3×3 (groups=3C) → BN → ReLU
    │
    ▼
Pointwise Conv 1×1 (3C → C) → BN → ReLU  
    │
    ├──► HF_reduced [B, C, H/2, W/2]
    │
    ▼
Conv 1×1 (C → 1) → M_bnd_logits
    │
    ▼
HF' = HF_reduced × sigmoid(M_bnd_logits)
```

**This reduces HFBA params by ~9× while maintaining accuracy.**

---

## 4. LOSS FUNCTION DESIGN

### 4.1 Segmentation Loss

```
L_seg = BCEWithLogitsLoss(logits_seg, y)
      + Dice_Loss(sigmoid(logits_seg), y)
```

where:
- `BCEWithLogitsLoss` = numerically stable binary cross-entropy on logits
- `Dice_Loss` = 1 - (2 × |ŷ ∩ y| + ε) / (|ŷ| + |y| + ε)

### 4.2 Boundary Loss

```
L_bnd = (1/4) × Σ BCEWithLogitsLoss(M_bnd_i, edge_GT_i)
```

where:
- `M_bnd_i` = logits output from the `i`-th WBS module before sigmoid
- `edge_GT_i` = Sobel edge map resized to match the spatial size of `M_bnd_i`
- `edge_GT` = Sobel edge detection applied to the binary segmentation mask `y`

There is **no separate auxiliary boundary head** in the final model. Boundary supervision happens only inside the WBS modules.

### 4.3 Total Loss

```
L_total = L_seg + λ × L_bnd
```

- **λ = 0.5** (start with this; tune in [0.1, 0.3, 0.5, 1.0] during ablation)

### 4.4 Boundary Ground Truth Generation (Preprocessing)

```python
import cv2
import numpy as np

def generate_boundary_gt(mask):
    """Generate boundary GT from segmentation mask using Sobel."""
    # mask: binary mask [H, W] with values {0, 1}
    sobelx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobelx**2 + sobely**2)
    edge = (edge > 0).astype(np.float32)  # binary edge
    # Optional: dilate to make boundary thicker (2-3 pixels)
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)
    return edge
```

---

## 5. COMPLETE IMPLEMENTATION PLAN

### Phase 1: Setup (Days 1-3)

**Task** | **Details** | **Time**
---|---|---
Environment setup | PyTorch 2.0+, CUDA, pytorch_wavelets, albumentations | Day 1
Download datasets | Kvasir-SEG, CVC-ClinicDB, ISIC 2018 (all publicly available) | Day 1
Data preprocessing | Resize to 352×352, generate boundary GTs, train/val/test splits | Day 2
Baseline training | Train vanilla U-Net (ResNet-34 encoder) on each dataset | Day 3

### Phase 2: Core Implementation (Days 4-10)

**Task** | **Details** | **Time**
---|---|---
Implement DWT/IDWT | Haar wavelet forward/inverse (or use pytorch_wavelets) | Day 4
Implement LFSA module | SE-style channel attention for LL sub-band | Day 4
Implement HFBA module | Depthwise separable conv + spatial attention + boundary supervision | Day 5
Implement WBS module | Combine DWT → LFSA/HFBA → IDWT into one nn.Module | Day 5-6
Integrate into U-Net | Replace the 4 decoder skip pathways (stem/relu, layer1, layer2, layer3) with WBS modules | Day 6
Implement loss function | Combined segmentation + boundary loss | Day 7
Debug & verify | Check shapes, gradients, memory usage | Day 7-8
First training run | Train WBSNet on Kvasir-SEG (smallest dataset) | Day 8-10

### Phase 3: Experiments (Days 11-20)

**Task** | **Details** | **Time**
---|---|---
Main experiments | Train on all 3-4 datasets, compare with baselines | Days 11-15
Ablation studies | 7 ablation variants (see Section 6) | Days 15-18
Qualitative results | Generate visual comparisons, boundary quality images | Day 19
Statistical analysis | Compute mean ± std over 3 runs; significance tests | Day 20

### Phase 4: Paper Writing (Days 21-30)

**Task** | **Details** | **Time**
---|---|---
Draft all sections | Using the outline we created; LaTeX IEEE format | Days 21-26
Create figures | Architecture diagram, qualitative comparison, ablation charts | Days 27-28
References & polish | BibTeX, proofreading, formatting | Days 29-30

---

## 6. EXPERIMENT PLAN

### 6.1 Datasets

| Dataset | Domain | Train | Test | Image Size | Evaluation |
|---|---|---|---|---|---|
| Kvasir-SEG | Polyp | 880 | 120 | 352×352 | Dice, IoU, Precision, Recall |
| CVC-ClinicDB | Polyp | 550 | 62 | 352×352 | Dice, IoU |
| CVC-ColonDB | Polyp | — | 380 | 352×352 | Cross-dataset generalization |
| ISIC 2018 | Skin lesion | 2594 | 1000 | 352×352 | Dice, IoU, Acc |

### 6.2 Baselines to Compare

| Method | Why Include | Source Code |
|---|---|---|
| U-Net (ResNet-34) | Standard baseline | torchvision / segmentation_models_pytorch |
| U-Net++ | Dense skip connection baseline | segmentation_models_pytorch |
| Attention U-Net | Attention gate baseline | segmentation_models_pytorch |
| PraNet | Strong polyp-specific baseline | Official GitHub |
| U-Net v2 | Best skip connection competitor | Official GitHub |
| TransUNet | Transformer baseline | Official GitHub |
| MEGANet | Edge-guided attention competitor | Official GitHub |

### 6.3 Ablation Study Design

| ID | Variant | What It Tests |
|---|---|---|
| A1 | U-Net + Standard Skip (Baseline) | No WBS module |
| A2 | U-Net + WBS (Full Model) | Complete WBSNet |
| A3 | U-Net + WBS without HFBA (LFSA only) | Is boundary attention needed? |
| A4 | U-Net + WBS without LFSA (HFBA only) | Is channel attention on LL needed? |
| A5 | U-Net + WBS without boundary supervision | Is boundary GT supervision needed? |
| A6 | U-Net + WBS without wavelet (attention on raw skip features) | Is wavelet decomposition needed? |
| A7 | U-Net + WBS with Daubechies-2 wavelet | Does wavelet family matter? |

**A6 is the most critical ablation** — it proves wavelet decomposition itself matters, not just "adding more attention."

### 6.4 Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 (encoder), 1e-3 (decoder + WBS) |
| LR scheduler | CosineAnnealing (T_max=epochs) |
| Batch size | 16 |
| Epochs | 200 (Kvasir/CVC), 100 (ISIC 2018) |
| Image size | 352 × 352 |
| Augmentation | HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter, RandomResizedCrop |
| Boundary loss weight λ | 0.5 |
| SE reduction ratio r | 16 |
| Encoder pretrained | ImageNet (ResNet-34) |

### 6.5 Evaluation Metrics

| Metric | Formula | What It Measures |
|---|---|---|
| Dice (DSC) | 2×TP / (2×TP + FP + FN) | Overall segmentation overlap |
| IoU (Jaccard) | TP / (TP + FP + FN) | Intersection over union |
| Precision | TP / (TP + FP) | False positive rate |
| Recall (Sensitivity) | TP / (TP + FN) | False negative rate |
| HD95 | 95th percentile Hausdorff Distance | Boundary accuracy |

**HD95 is critical for our paper** since we claim boundary improvement. Make sure to report it.

---

## 7. KEY CODE STRUCTURE

```
WBSNet/
├── models/
│   ├── wbsnet.py          # Main model
│   ├── wbs_module.py      # WBS Module (DWT + LFSA + HFBA + IDWT)
│   ├── lfsa.py            # Low-Frequency Semantic Attention
│   ├── hfba.py            # High-Frequency Boundary Attention
│   ├── wavelet.py         # DWT and IDWT implementations
│   └── decoder.py         # Decoder blocks
├── datasets/
│   ├── polyp_dataset.py   # Kvasir-SEG, CVC-ClinicDB loader
│   ├── isic_dataset.py    # ISIC 2018 loader
│   └── transforms.py      # Data augmentation
├── losses/
│   ├── seg_loss.py        # BCEWithLogits + Dice
│   └── boundary_loss.py   # Mean BCEWithLogits over WBS boundary logits
├── utils/
│   ├── metrics.py         # Dice, IoU, HD95 calculation
│   ├── boundary_gt.py     # Sobel edge GT generation
│   └── visualize.py       # Qualitative result visualization
├── train.py               # Training script
├── test.py                # Evaluation script
├── configs/
│   ├── kvasir.yaml        # Dataset-specific config
│   └── isic.yaml
└── requirements.txt
```

---

## 8. CRITICAL TIPS FOR SUCCESS

### What Reviewers Will Look For

1. **Ablation completeness**: A6 (no wavelet) is mandatory. If WBS without wavelet performs similarly, your paper loses its core claim.

2. **HD95 metric**: Since you claim boundary improvement, you MUST show Hausdorff Distance improvement. If you only show Dice, reviewers will say "where's the boundary evidence?"

3. **Visualization**: Show side-by-side predictions on challenging boundary cases (fuzzy polyp edges, hair-occluded lesion borders). Draw a red box around the boundary region and zoom in.

4. **Parameter/FLOPs comparison**: Show a scatter plot of Dice vs. Parameters for all baselines. Your model should be in the top-right (high Dice, reasonable params).

5. **Cross-dataset generalization**: Train on Kvasir-SEG, test on CVC-ColonDB (zero-shot). This shows your method generalizes, not just overfits.

### Common Pitfalls to Avoid

- **Don't forget to detach boundary GT during backprop** — the Sobel edge is precomputed, not a gradient-tracked operation.
- **Final skip topology**: The 4 WBS modules operate on `176`, `88`, `44`, and `22` feature maps from `stem/relu`, `layer1`, `layer2`, and `layer3`. `layer4` is bottleneck-only and not a skip.
- **Wavelet size constraint**: WBS inputs must be even for Haar DWT. With the locked `352×352` setup, all four skip-source feature maps are even-sized, so no DWT padding is needed.
- **Memory**: DWT doubles the number of feature maps temporarily. Monitor GPU memory.
- **Fair comparison**: Use the SAME encoder (ResNet-34) for ALL baselines. Don't compare your ResNet-34 model against a PVTv2-B2 model.

---

## 9. TIMELINE SUMMARY (1-2 Months)

```
Week 1: Setup + Baseline + Core Implementation
Week 2: WBS Module Complete + First Training
Week 3: Main Experiments (all datasets)
Week 4: Ablation Studies + Qualitative Results  
Week 5: Paper Writing (Sections 1-3)
Week 6: Paper Writing (Sections 4-5) + Figures + Polish
Week 7: Review + Revise + Submit (buffer week)
```

---

## 10. QUICK-START: First 3 Things To Do Today

1. **Install**: `pip install segmentation-models-pytorch pytorch-wavelets albumentations`
2. **Download**: Kvasir-SEG dataset from https://datasets.simula.no/kvasir-seg/
3. **Run**: Train a vanilla U-Net baseline to establish your bottom-line performance

Once you have the baseline Dice score, we know exactly how much improvement the WBS module needs to deliver for a publishable result (typically +1-3% Dice is sufficient for mid-tier venues).
