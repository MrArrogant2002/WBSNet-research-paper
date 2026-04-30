# WBSNet Paper Outline
**Title:** WBSNet: Wavelet Boundary Skip Network for Medical Image Segmentation
**Venue:** IEEE journal — target IEEE Transactions on Medical Imaging (TMI) or IEEE Access
**Template:** `\documentclass[journal]{IEEEtran}` (~12–18 pages)

---

## Paper Thesis
Standard U-Net skip connections conflate low-frequency semantic features with high-frequency edge details; inserting a DWT-based WBS module into each skip connection — with dedicated LFSA attention on the LL subband and HFBA attention + boundary supervision on the LH/HL/HH subbands — yields sharper, more accurate segmentation boundaries on medical images.

---

## Contributions (verbatim, for Introduction bullet list)
1. **WBS Module** — plug-in skip refinement block: DWT decomposition → dual-branch attention → IDWT reconstruction; no external wavelet library at runtime.
2. **LFSA** — SE-style channel attention on the LL subband for semantic recalibration.
3. **HFBA** — high-frequency fusion → per-level boundary logits + per-subband attention gates (gate applied independently to each of LH/HL/HH, preserving directional edge information).
4. **Multi-Level Boundary Supervision** — auxiliary BCE losses on per-level boundary logits upsampled to GT resolution (preserves 1-pixel edge signal).
5. **Comprehensive Evaluation** — state-of-the-art results on Kvasir-SEG, CVC-ClinicDB, ISIC-2018; cross-dataset generalisation on CVC-ColonDB; seven-variant ablation (A1–A7) with statistical significance testing across three seeds.

---

## Section-by-Section Plan

### Abstract (150–250 words)
| Beat | Content | Fill status |
|------|---------|-------------|
| Context | Clinical importance of polyp / skin-lesion segmentation | Draft ready |
| Problem | Skip-connection frequency entanglement → blurred boundaries | Draft ready |
| Method | WBSNet: DWT + LFSA + HFBA + multi-level boundary supervision | Draft ready |
| Results | Dice on Kvasir / ClinicDB / ISIC / ColonDB (4 numbers), improvement over U-Net, p-value | **TODO after experiments** |
| Significance | Pure-PyTorch DWT; 7-variant ablation | Draft ready |

---

### I. Introduction
| Beat | Content | Required citations |
|------|---------|-------------------|
| 1 | Clinical motivation: colorectal cancer / melanoma, automated diagnosis | `jha2020kvasir`, `bernal2015cvc`, `codella2019isic` |
| 2 | U-Net dominance; skip connection variants | `ronneberger2015unet`, `zhou2019unetpp`, `oktay2018attnunet`, `wang2024udtransnet`, `tamosc2025` |
| 3 | Gap: no existing method applies frequency decomposition within the skip path | `mallat1989wavelet`, `wang2023wranet`, `sffnet2024` |
| 4 | Our approach: WBSNet overview | — (self-referential) |
| 5 | Contribution bullets (verbatim from above) | — |
| 6 | Paper roadmap | — |

**Figure planned:** None in Introduction.

---

### II. Related Work

#### A. U-Net Variants and Skip Connection Refinement
- UNet++ nested dense skips; Attention U-Net gated skips; U-Net v2 saliency; R2U++ recurrent residual
- Transformer skips: TransUNet, Swin-UNet, UDTransNet, TA-MoSC, SK-VM++
- **Closing limitation:** none performs frequency decomposition within the skip path
- **Required citations:** `zhou2019unetpp`, `oktay2018attnunet`, `peng2023unetv2`, `alom2022r2upp`, `chen2021transunet`, `cao2021swinunet`, `wang2024udtransnet`, `tamosc2025`, `wu2025skvmpp`, `rethinkskips2024`, `isensee2021nnu`

#### B. Boundary-Aware Segmentation
- PraNet, BUNet, BRNet, BMANet, MEGANet, BCF-UNet, BGGL-Net (spatial boundary cues in polyp/skin segmentation)
- Boundary loss formulation
- **Closing limitation:** boundary cues derived via spatial convolutions — sensitive to noise; no frequency-domain grounding
- **Required citations:** `fan2020pranet`, `bunet2023`, `brnet2024`, `bmanet2025`, `meganet2024`, `bcfunet2024`, `bgglnet2025`, `kervadec2019boundaryloss`

#### C. Frequency-Domain Deep Learning
- Mallat wavelet theory; DWT for shift invariance; FNet Fourier mixing
- Medical: WRANet, EASNet, WA-NET, FMDDC; Remote sensing: SFFNet, WFE; Dense prediction: FreqFusion
- **Closing limitation:** wavelets in encoder or global fusion — not in the skip path of a U-Net decoder pipeline
- **Required citations:** `mallat1989wavelet`, `zhang2019makingshiftinvariant`, `rao2022fnet`, `wang2023wranet`, `easnet2025`, `wanet2025`, `dai2025fmddc`, `sffnet2024`, `wfe2023`, `li2024freqfusion`

#### D. Attention Mechanisms in Segmentation
- SE Networks; channel and spatial attention; PMFSNet, nnFormer, FCBFormer
- **Distinction:** LFSA/HFBA apply attention to frequency-decomposed subbands, not full-spectrum features
- **Required citations:** `hu2018senet`, `pmfsnet2024`, `zhou2023nnformer`, `fcbformer2022`

---

### III. WBSNet (Method)

#### 3.1 Problem Formulation
- Notation: x ∈ ℝ^{3×H×W}, y ∈ {0,1}^{H×W}, b = boundary mask
- Network output: ŷ (segmentation logit) + {b̂^(k)}_{k=1}^4 (boundary logits)
- Total loss: L = L_seg + λ L_bnd, λ=0.5
- **No citations needed** (notation section)

#### 3.2 Overall Architecture
- ResNet-34 encoder (ImageNet pretrained): 5 feature tensors, channels {64,64,128,256,512}
- 4 WBS modules on skip connections (stem, layer1, layer2, layer3)
- 4 DecoderBlocks: 2× bilinear upsample + concat + 2× Conv3×3+BN+ReLU, channels [256,128,64,32]
- Head: Conv(32→1) + 2× bilinear upsample
- **Figure 1:** WBSNet architecture overview (see TODO in paper.tex)
- **Required citations:** `he2016resnet`

#### 3.3 WBS Module
- DWT: grouped stride-2 convolution → LL, LH, HL, HH (each C × H/2 × W/2)
- Dual branch: LL → LFSA, {LH,HL,HH} → HFBA
- IDWT: grouped stride-1/2 transposed convolution → F̂ ∈ ℝ^{C×H×W}
- Output: (F̂, b̂) — boundary logit for auxiliary loss
- Fallback: RawAttentionSkip when use_wavelet=False (ablation A6)
- **Figure 2:** WBS module internals (see TODO in paper.tex)
- **Required citations:** `mallat1989wavelet`

#### 3.4 LFSA
- Formula: LFSA(F_LL) = F_LL ⊙ σ(W₂ δ(W₁ GAP(F_LL)))
- Reduction ratio r=16, min bottleneck=4
- **Required citations:** `hu2018senet`

#### 3.5 HFBA
- Concat LH+HL+HH → DWConv3×3+BN+ReLU → PWConv1×1+BN+ReLU → boundary logit b̂
- Gate g = σ(b̂); applied independently to each of LH, HL, HH (design rationale: preserves directional edge info)
- **No external citations** (novel module)

#### 3.6 Loss Function
- L_seg = L_BCE(ŷ, y) + L_Dice(ŷ, y)
- L_bnd = (1/K) Σ L_BCE(↑b̂^(k), b), upsampling preserves 1-pixel edge signal
- L = L_seg + 0.5 L_bnd
- Equations already in paper.tex (eqs. 1–3)
- **Required citations:** `kervadec2019boundaryloss`

#### 3.7 Training Protocol
- AdamW, LR_enc=1e-4, LR_dec=1e-3, WD=1e-4, grad clip norm 1.0
- CosineAnnealingLR, T_max=200 epochs (100 for ISIC)
- AMP; batch=16; input 352×352; seeds 3407/3408/3409
- **Required citations:** `loshchilov2019adamw`, `loshchilov2017cosine`

---

### IV. Experiments

#### 4.1 Datasets
| Dataset | Size | Split | Purpose |
|---------|------|-------|---------|
| Kvasir-SEG | 1000 | 80/10/10 | Primary polyp benchmark |
| CVC-ClinicDB | 612 | 80/10/10 | Secondary polyp benchmark |
| ISIC-2018 | 2594 | 80/10/10 | Skin lesion (cross-modality) |
| CVC-ColonDB | 380 | all | Cross-dataset generalisation only |

**Required citations:** `jha2020kvasir`, `bernal2015cvc`, `codella2019isic`, `tajbakhsh2015colondb`

#### 4.2 Evaluation Metrics
- Dice, IoU, Precision, Recall, Accuracy, Specificity (from global TP/FP/FN/TN)
- HD95 (95th-percentile Hausdorff distance) — boundary-quality metric; disabled in training, enabled in final eval
- Threshold: 0.5
- **Required citations:** `aydin2021hd95`

#### 4.3 Comparison with SOTA (Tables I & II)
- **Table I (polyp):** Kvasir-SEG + CVC-ClinicDB — Dice/IoU/HD95 for 12 baselines + WBSNet
- **Table II (skin):** ISIC-2018 — Dice/IoU/HD95 for 6 baselines + WBSNet
- Narrative: highlight HD95 improvement as key boundary-quality evidence
- **Status: TODO — fill from outputs/aggregated after experiments**

#### 4.4 Ablation Study (Table III)
- Variants A1–A7 on Kvasir-SEG, Dice/IoU/HD95 mean±std over 3 seeds
- Paired t-test (A2 vs each variant), p-values reported
- **Status: TODO — fill from outputs/aggregated**

| ID | Config | DWT | LFSA | HFBA | Bnd.Sup | Wavelet |
|----|--------|-----|------|------|---------|---------|
| A1 | identity_unet | ✗ | ✗ | ✗ | ✗ | — |
| A2 | kvasir_wbsnet (full) | ✓ | ✓ | ✓ | ✓ | haar |
| A3 | lfsa_only | ✓ | ✓ | ✗ | ✗ | haar |
| A4 | hfba_only | ✓ | ✗ | ✓ | ✓ | haar |
| A5 | no_boundary_supervision | ✓ | ✓ | ✓ | ✗ | haar |
| A6 | no_wavelet | ✗ | ✓ | ✓ | ✓ | — |
| A7 | db2_wavelet | ✓ | ✓ | ✓ | ✓ | db2 |

#### 4.5 Cross-Dataset Generalisation (Table IV)
- Train Kvasir-SEG, eval on CVC-ColonDB (all 380); zero fine-tuning
- WBSNet vs U-Net baseline; Dice/IoU/HD95 mean±std; p-value
- **Status: TODO**

#### 4.6 Model Complexity (Table V)
- Params (M), FLOPs (GFLOPs), Dice for WBSNet vs U-Net, TransUNet, Swin-UNet
- Run: `python scripts/model_complexity.py`

#### 4.7 Qualitative Analysis (Figure 3)
- 6 examples: 3 Kvasir-SEG + 1 CVC-ClinicDB + 2 ISIC-2018
- Columns: Input | GT | U-Net | WBSNet | WBSNet boundary map
- Add zoomed insets at challenging boundaries
- Generated by `scripts/make_paper_figures.py`

#### 4.8 Lambda Sensitivity (Figure 4)
- λ ∈ {0.0, 0.1, 0.2, 0.5, 1.0, 2.0}, Dice + HD95 vs λ on Kvasir-SEG val
- Mark chosen λ=0.5; show mean±std bands over 3 seeds

---

### V. Discussion
| Beat | Content |
|------|---------|
| Frequency separation | A6 vs A2: DWT contribution beyond raw attention |
| Boundary supervision | A5 vs A2: HD95 gap quantifies auxiliary loss effect |
| Wavelet filter | A7 vs A2: when db2 helps vs Haar suffices |
| Cross-dataset | Why frequency disentanglement improves transfer |
| Limitations | Even-dim padding; binary only; inference overhead; ResNet-34 ceiling |
| Threats | Single random split (mitigated by 3 seeds); reproduced baseline numbers |

---

### VI. Conclusion
| Beat | Content |
|------|---------|
| Restate | WBS module: DWT + LFSA + HFBA + boundary supervision; pure-PyTorch |
| Evidence | Headline Dice/IoU/HD95 on 3 datasets + ColonDB generalisation + ablation significance |
| Future work | Learnable DWT; multi-class boundary heads; 3D extension; SAM/DINOv2 encoder |

---

## Figures and Tables Summary

| # | Type | Label | Status |
|---|------|-------|--------|
| Fig. 1 | Architecture diagram | `fig:architecture` | PNG exists (`figures/wbsnet_architecture.png`) — verify quality |
| Fig. 2 | WBS module block diagram | `fig:wbs_module` | **TODO** — create TikZ or draw diagram |
| Fig. 3 | Qualitative prediction grid | `fig:qualitative` | **TODO** — generate after training |
| Fig. 4 | Lambda sensitivity curve | `fig:lambda` | **TODO** — run lambda sweep |
| Table I | Polyp SOTA comparison | `tab:sota_polyp` | **TODO** — fill from experiments |
| Table II | Skin SOTA comparison | `tab:sota_skin` | **TODO** — fill from experiments |
| Table III | Ablation A1–A7 | `tab:ablation` | **TODO** — fill from ablation runs |
| Table IV | Cross-dataset generalisation | `tab:generalisation` | **TODO** — fill from generalization runs |
| Table V | Model complexity | `tab:complexity` | **TODO** — run model_complexity.py |

---

## Citation Checklist (all keys in references.bib)
| Cite key | Entry type | Used in |
|----------|-----------|---------|
| `ronneberger2015unet` | conference | Intro, Related, caption |
| `zhou2019unetpp` | journal | Related, Table I |
| `oktay2018attnunet` | conference | Related, Table I |
| `peng2023unetv2` | conference | Related |
| `wang2024udtransnet` | journal | Intro, Related |
| `tamosc2025` | misc | Intro, Related |
| `wu2025skvmpp` | journal | Related |
| `alom2022r2upp` | journal | Related |
| `rethinkskips2024` | misc | Related |
| `mallat1989wavelet` | journal | Intro, Related, Method |
| `hu2018senet` | conference | Intro, Method |
| `wang2023wranet` | journal | Related |
| `sffnet2024` | journal | Related |
| `wfe2023` | journal | Related |
| `wanet2025` | journal | Related |
| `easnet2025` | journal | Related |
| `li2024freqfusion` | journal | Related |
| `dai2025fmddc` | journal | Related |
| `zhang2019makingshiftinvariant` | conference | Related |
| `rao2022fnet` | conference | Related |
| `fan2020pranet` | conference | Related, Table I |
| `bunet2023` | journal | Related |
| `brnet2024` | journal | Related |
| `bmanet2025` | journal | Related, Table I |
| `meganet2024` | conference | Related, Table I |
| `bcfunet2024` | journal | Related, Table I |
| `bgglnet2025` | journal | Related, Table I |
| `kervadec2019boundaryloss` | conference | Related, Method |
| `chen2021transunet` | misc | Related, Table I |
| `cao2021swinunet` | conference | Related, Table I, II |
| `fcbformer2022` | conference | Related, Table I |
| `rao2022fnet` | conference | Related |
| `zhou2023nnformer` | journal | Related |
| `pmfsnet2024` | journal | Related |
| `he2016resnet` | conference | Method |
| `isensee2021nnu` | journal | Related |
| `ruan2024vmunet` | journal | Related, Table II |
| `skinmamba2024` | misc | Related, Table II |
| `jha2020kvasir` | conference | Intro, Experiments |
| `bernal2015cvc` | journal | Intro, Experiments |
| `tajbakhsh2015colondb` | journal | Experiments |
| `codella2019isic` | misc | Intro, Experiments |
| `loshchilov2019adamw` | conference | Method |
| `loshchilov2017cosine` | conference | Method |
| `aydin2021hd95` | journal | Experiments |

---

## Next Steps (ordered by priority)
1. **Run experiments** — `python scripts/run_ablation_suite.py --seeds 3407 3408 3409` on Kvasir-SEG; `python train.py` on ClinicDB and ISIC-2018
2. **Aggregate results** — `python aggregate_results.py --root outputs --output outputs/aggregated`
3. **Statistical tests** — `python scripts/significance_tests.py --root outputs --reference A1_identity_unet`
4. **Model complexity** — `python scripts/model_complexity.py`
5. **Generate figures** — `python scripts/make_paper_figures.py`
6. **Fill tables** — replace all TODO values in paper.tex with actual numbers
7. **Draft prose** — run `/paper-draft introduction`, `/paper-draft method`, etc. per section
8. **WBS module diagram** — create TikZ diagram for Figure 2 (highest priority figure)
9. **Polish** — run `/paper-polish` once full draft is complete
