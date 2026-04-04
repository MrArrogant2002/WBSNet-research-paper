# WBSNet Research Proposal — LaTeX Source Code

> **Instructions:** Copy the entire LaTeX code below into a `.tex` file and compile with `pdflatex` + `bibtex`. All BibTeX entries are included inline at the bottom.

---

```latex
% ============================================================
% PREAMBLE
% ============================================================
\documentclass[12pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{natbib}
\usepackage{algorithm, algpseudocode}
\usepackage{multirow, makecell}
\usepackage{xcolor}
\usepackage{pifont}
\usepackage{enumitem}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{array}
\usepackage{tabularx}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\newcommand{\method}{WBSNet}
\newcommand{\module}{WBS}

\begin{document}

% ============================================================
% FRONT MATTER
% ============================================================
\begin{center}
    {\LARGE \textbf{WBSNet: Wavelet-Guided Boundary-Aware Skip Connections\\[4pt] for Medical Image Segmentation}}\\[20pt]
    {\large \textbf{Research Proposal \& Architecture Document}}\\[10pt]
    {\large Prepared for: Guide Discussion \& Hardware Planning}\\[10pt]
    \today
\end{center}

\vspace{10pt}
\hrule
\vspace{15pt}

\tableofcontents
\newpage

% ============================================================
% SECTION 1: EXECUTIVE SUMMARY
% ============================================================
\section{Executive Summary}

This document presents \textbf{\method{}} (Wavelet Boundary Skip Network), a novel medical image segmentation architecture that addresses two fundamental weaknesses of U-Net's skip connections: the \textit{semantic gap} between encoder and decoder features, and \textit{boundary information loss} during downsampling. Our core contribution is the \textbf{WBS Module} (Wavelet Boundary Skip), a plug-and-play module that replaces standard skip connections by decomposing encoder features into frequency sub-bands using the Haar wavelet transform, applying targeted attention to each sub-band, and reconstructing enhanced features for the decoder.

The project targets publication at a medical image analysis venue (e.g., IEEE ISBI, MICCAI, or IEEE TMI) and requires GPU-based training infrastructure. This document provides the complete technical design, literature positioning, experiment plan, and hardware requirements for the research.

% ============================================================
% SECTION 2: PROBLEM STATEMENT & MOTIVATION
% ============================================================
\section{Problem Statement \& Motivation}

\subsection{Why Medical Image Segmentation Matters}

Medical image segmentation is the task of assigning a class label to every pixel in a medical image — for example, delineating a polyp in a colonoscopy frame or marking the boundary of a skin lesion in a dermoscopy image. Accurate segmentation is critical for:

\begin{itemize}[noitemsep]
    \item \textbf{Clinical diagnosis:} Precise boundary delineation affects staging (e.g., lesion size in skin cancer), treatment planning, and surgical navigation.
    \item \textbf{Automated screening:} Large-scale population screening (e.g., colorectal polyp detection) requires reliable pixel-level predictions.
    \item \textbf{Quantitative analysis:} Downstream measurements (volume, shape descriptors) depend on segmentation accuracy, especially at boundaries.
\end{itemize}

\subsection{U-Net and the Skip Connection Problem}

U-Net~\citep{ronneberger2015unet} is the de facto standard for medical image segmentation. Its encoder-decoder architecture uses \textit{skip connections} to transfer spatial information from the encoder to the decoder. However, two well-documented problems exist:

\begin{enumerate}[noitemsep]
    \item \textbf{Semantic gap:} Encoder features at shallow layers capture low-level patterns (edges, textures), while the decoder expects high-level semantic representations. Directly concatenating these creates a feature distribution mismatch that the decoder's limited capacity cannot fully resolve~\citep{wang2024udtransnet, peng2023unetv2}.
    
    \item \textbf{Boundary information loss:} Progressive downsampling in the encoder erodes fine-grained boundary details. Standard skip connections pass raw encoder features without distinguishing boundary-critical information from noise, leading to blurred predictions at object edges~\citep{bui2023meganet, brnet2024}.
\end{enumerate}

\textbf{Key insight:} Recent analytical work~\citep{rethinking2024skips} has shown that shallow skip connections often carry domain-specific noise that hurts cross-domain generalization. This motivates a \textit{selective filtering} approach rather than passing all features blindly.

\subsection{Why Existing Solutions Fall Short}

Table~\ref{tab:gap_analysis} summarizes the landscape and identifies the specific gap our work fills.

\begin{table}[H]
\centering
\caption{Gap analysis: How \method{} differs from the closest related works.}
\label{tab:gap_analysis}
\small
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.2cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}p{2.2cm} >{\centering\arraybackslash}p{2.5cm}}
\toprule
\textbf{Method} & \textbf{Wavelet in Skips?} & \textbf{Boundary Supervision?} & \textbf{Unified Module?} & \textbf{All Skip Levels?} \\
\midrule
U-Net v2~\citep{peng2023unetv2} & \ding{55} & \ding{55} & -- & \ding{51} \\
UDTransNet~\citep{wang2024udtransnet} & \ding{55} & \ding{55} & \ding{51} & \ding{51} \\
WA-NET~\citep{wanet2025} & Separate & Separate & \ding{55} & Partial \\
HCViT-Net~\citep{hcvitnet2025} & \ding{51} & \ding{55} & Partial & Only 1 level \\
MEGANet~\citep{bui2023meganet} & \ding{55} & Laplacian & \ding{51} & \ding{51} \\
\midrule
\textbf{\method{} (Ours)} & \textbf{\ding{51}} & \textbf{\ding{51}} & \textbf{\ding{51}} & \textbf{\ding{51}} \\
\bottomrule
\end{tabularx}
\end{table}

\textbf{The gap:} To the best of our knowledge, no existing method unifies wavelet-based frequency decomposition with boundary-supervised spatial attention \textit{inside} the skip pathways at \textit{all four} encoder-decoder fusion levels of a standard ResNet-U-Net topology.

% ============================================================
% SECTION 3: PROPOSED METHOD
% ============================================================
\section{Proposed Method: \method{}}

\subsection{Architecture Overview}

\method{} follows a standard ResNet-U-Net encoder-decoder paradigm with a ResNet-34~\citep{he2016resnet} encoder pretrained on ImageNet. The key modification is that the \textbf{four skip pathways used by the decoder} are all replaced with our proposed \module{} modules. Concretely, we use:

\begin{equation}
\{S_1, S_2, S_3, S_4\} = \{\text{stem/relu}, \text{layer1}, \text{layer2}, \text{layer3}\},
\quad
B = \text{layer4}
\end{equation}

\begin{equation}
\text{logits}_{\text{seg}} =
\text{Head}\Big(
\text{Up}_{2\times}
\big(
\text{Decoder}(B, \text{WBS}(S_1), \text{WBS}(S_2), \text{WBS}(S_3), \text{WBS}(S_4))
\big)
\Big)
\end{equation}

where $S_k$ denotes a skip-source feature and $B$ is the deepest bottleneck representation. Importantly, \textbf{layer4 is used only as the bottleneck input}, not as an additional skip feature.

% Architecture figure placeholder
\begin{figure}[H]
    \centering
    \fbox{\parbox{0.9\textwidth}{\centering \vspace{40pt} \textit{[Insert WBSNet\_Architecture.png here]} \\ \texttt{\small \textbackslash includegraphics[width=\textbackslash linewidth]\{figures/WBSNet\_Architecture.png\}} \vspace{40pt}}}
    \caption{Overall architecture of \method{}. The four skip sources are stem/relu, layer1, layer2, and layer3. Each passes through a \module{} module (yellow), while layer4 serves only as the bottleneck input for the decoder.}
    \label{fig:architecture}
\end{figure}

\subsubsection{Encoder}
We tap the standard ResNet-34 feature hierarchy at the following spatial scales:
\begin{itemize}[noitemsep]
    \item $S_1 = \text{stem/relu} \in \mathbb{R}^{B \times 64 \times 176 \times 176}$
    \item $S_2 = \text{layer1} \in \mathbb{R}^{B \times 64 \times 88 \times 88}$
    \item $S_3 = \text{layer2} \in \mathbb{R}^{B \times 128 \times 44 \times 44}$
    \item $S_4 = \text{layer3} \in \mathbb{R}^{B \times 256 \times 22 \times 22}$
    \item $B = \text{layer4} \in \mathbb{R}^{B \times 512 \times 11 \times 11}$
\end{itemize}

Thus, the decoder receives four WBS-refined skip tensors at resolutions $176$, $88$, $44$, and $22$, while the deepest $11 \times 11$ feature is reserved for the bottleneck.

\subsubsection{Bottleneck}
The output of \texttt{layer4} serves as the bottleneck representation $B \in \mathbb{R}^{B \times 512 \times 11 \times 11}$. No skip connection is taken from this stage.

\subsubsection{Decoder}
The decoder contains four stages:
\begin{itemize}[noitemsep]
    \item \textbf{Dec4:} $11 \rightarrow 22$, concatenate with $\text{WBS}(S_4)$, output $22 \times 22 \times 256$
    \item \textbf{Dec3:} $22 \rightarrow 44$, concatenate with $\text{WBS}(S_3)$, output $44 \times 44 \times 128$
    \item \textbf{Dec2:} $44 \rightarrow 88$, concatenate with $\text{WBS}(S_2)$, output $88 \times 88 \times 64$
    \item \textbf{Dec1:} $88 \rightarrow 176$, concatenate with $\text{WBS}(S_1)$, output $176 \times 176 \times 32$
\end{itemize}

After Dec1, a final $2\times$ bilinear upsampling produces a $352 \times 352 \times 32$ feature map before the segmentation head.

\subsubsection{Output Heads}
\begin{itemize}[noitemsep]
    \item \textbf{Segmentation head:} final $2\times$ upsampling $\rightarrow$ Conv $1 \times 1$, producing $\text{logits}_{\text{seg}} \in \mathbb{R}^{H \times W}$.
\end{itemize}

All WBS inputs in the final design have even spatial dimensions ($176$, $88$, $44$, $22$), so no DWT padding is required.

\subsection{The WBS Module — Core Contribution}

Each \module{} module takes an encoder feature $\mathbf{F}_{\text{enc}} \in \mathbb{R}^{B \times C \times H \times W}$ and outputs an enhanced feature $\mathbf{F}'_{\text{skip}} \in \mathbb{R}^{B \times C \times H \times W}$ of the \textit{same shape}. The pipeline consists of four steps.

\subsubsection{Step 1: 2D Haar Discrete Wavelet Transform (DWT)}

The Haar DWT decomposes each channel of $\mathbf{F}_{\text{enc}}$ into four frequency sub-bands:

\begin{align}
    \text{LL}[i,j] &= \tfrac{1}{2}\big(x[2i,2j] + x[2i,2j\!+\!1] + x[2i\!+\!1,2j] + x[2i\!+\!1,2j\!+\!1]\big) \label{eq:ll}\\
    \text{LH}[i,j] &= \tfrac{1}{2}\big(x[2i,2j] - x[2i,2j\!+\!1] + x[2i\!+\!1,2j] - x[2i\!+\!1,2j\!+\!1]\big) \label{eq:lh}\\
    \text{HL}[i,j] &= \tfrac{1}{2}\big(x[2i,2j] + x[2i,2j\!+\!1] - x[2i\!+\!1,2j] - x[2i\!+\!1,2j\!+\!1]\big) \label{eq:hl}\\
    \text{HH}[i,j] &= \tfrac{1}{2}\big(x[2i,2j] - x[2i,2j\!+\!1] - x[2i\!+\!1,2j] + x[2i\!+\!1,2j\!+\!1]\big) \label{eq:hh}
\end{align}

\begin{itemize}[noitemsep]
    \item $\text{LL} \in \mathbb{R}^{B \times C \times \frac{H}{2} \times \frac{W}{2}}$: Low-frequency approximation capturing \textit{smooth structure and semantics}.
    \item $\text{LH}, \text{HL}, \text{HH} \in \mathbb{R}^{B \times C \times \frac{H}{2} \times \frac{W}{2}}$: High-frequency details capturing \textit{horizontal, vertical, and diagonal edges} respectively.
\end{itemize}

\textbf{Why Haar?} The Haar wavelet is computationally free (only additions and subtractions), perfectly invertible, and naturally aligns with boundary detection. It introduces \textbf{zero learnable parameters}.

\subsubsection{Step 2: LFSA — Low-Frequency Semantic Attention}

LFSA applies channel-wise recalibration (Squeeze-and-Excitation style~\citep{hu2018senet}) to the LL sub-band, enhancing semantically important channels:

\begin{equation}
    \text{LL}' = \text{LL} \odot \sigma\Big(\mathbf{W}_2 \cdot \text{ReLU}\big(\mathbf{W}_1 \cdot \text{GAP}(\text{LL})\big)\Big)
    \label{eq:lfsa}
\end{equation}

where $\text{GAP}(\cdot)$ is global average pooling, $\mathbf{W}_1 \in \mathbb{R}^{C/r \times C}$ and $\mathbf{W}_2 \in \mathbb{R}^{C \times C/r}$ are fully connected layers with reduction ratio $r=16$, $\sigma$ is the sigmoid function, and $\odot$ denotes channel-wise multiplication.

\textbf{Intuition:} Not all channels in the low-frequency sub-band are equally informative. LFSA learns to amplify channels carrying strong semantic signals and suppress noisy ones.

\subsubsection{Step 3: HFBA — High-Frequency Boundary Attention}

HFBA generates a spatial attention map that highlights boundary-relevant regions in the high-frequency sub-bands:

\begin{align}
    \mathbf{HF} &= \text{Concat}(\text{LH}, \text{HL}, \text{HH}) \in \mathbb{R}^{B \times 3C \times \frac{H}{2} \times \frac{W}{2}} \\
    \tilde{\mathbf{HF}} &= \text{ReLU}\Big(\text{BN}\big(\text{PWConv}_{1\times1}(\text{DWConv}_{3\times3}(\mathbf{HF}))\big)\Big) \in \mathbb{R}^{B \times C \times \frac{H}{2} \times \frac{W}{2}} \\
    \mathbf{M}_{\text{bnd}}^{(k)} &= \text{Conv}_{1\times1}(\tilde{\mathbf{HF}}) \in \mathbb{R}^{B \times 1 \times \frac{H}{2} \times \frac{W}{2}} \label{eq:mbnd} \\
    \mathbf{A}_{\text{bnd}}^{(k)} &= \sigma\big(\mathbf{M}_{\text{bnd}}^{(k)}\big) \\
    \mathbf{HF}' &= \tilde{\mathbf{HF}} \odot \mathbf{A}_{\text{bnd}}^{(k)}
\end{align}

where DWConv and PWConv denote depthwise and pointwise separable convolutions respectively (lightweight variant).

\textbf{Critical design:} $\mathbf{M}_{\text{bnd}}^{(k)}$ denotes the \textit{logits} produced by the $k$-th WBS module. A sigmoid is applied only to obtain the attention weights $\mathbf{A}_{\text{bnd}}^{(k)}$ used for feature modulation. During training, the logits are supervised by boundary ground truth derived from Sobel edge detection on the segmentation mask:
\begin{equation}
    \mathcal{L}_{\text{bnd}}^{(k)} = \text{BCEWithLogitsLoss}\big(\mathbf{M}_{\text{bnd}}^{(k)}, \; \text{Resize}(\text{Sobel}(\mathbf{y}), \text{size}(\mathbf{M}_{\text{bnd}}^{(k)}))\big)
    \label{eq:lbnd}
\end{equation}

This explicit boundary supervision forces the attention map to focus on actual anatomical boundaries rather than arbitrary high-frequency noise.

\subsubsection{Step 4: Inverse DWT (IDWT)}

The enhanced sub-bands are recombined via the inverse Haar wavelet transform:
\begin{equation}
    \mathbf{F}'_{\text{skip}} = \text{IDWT}(\text{LL}', \text{LH}', \text{HL}', \text{HH}') \in \mathbb{R}^{B \times C \times H \times W}
\end{equation}

Following the lightweight design, the same attended tensor is reused for all three high-frequency slots:
\[
\text{LH}' = \text{HL}' = \text{HH}' = \mathbf{HF}'.
\]

This produces a feature map at the \textit{original spatial resolution}, ready for concatenation with the upsampled decoder feature.

\subsection{Loss Function}

The total training loss combines segmentation quality and boundary fidelity:

\begin{equation}
    \mathcal{L}_{\text{seg}} = \text{BCEWithLogitsLoss}(\text{logits}_{\text{seg}}, y) + \text{Dice}\big(\sigma(\text{logits}_{\text{seg}}), y\big)
\end{equation}

\begin{equation}
    \mathcal{L}_{\text{bnd}} = \frac{1}{4} \sum_{k=1}^{4} \text{BCEWithLogitsLoss}\Big(\mathbf{M}_{\text{bnd}}^{(k)}, \text{Resize}(\text{Sobel}(y), \text{size}(\mathbf{M}_{\text{bnd}}^{(k)}))\Big)
\end{equation}

\begin{equation}
    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda \cdot \mathcal{L}_{\text{bnd}}
    \label{eq:total_loss}
\end{equation}

where $\lambda = 0.5$ balances the two objectives (tuned during ablation).

% ============================================================
% SECTION 4: NOVELTY & CONTRIBUTIONS
% ============================================================
\section{Novelty \& Contributions}

Our primary contributions are:

\begin{enumerate}
    \item \textbf{WBS Module:} A novel, plug-and-play skip connection enhancement module that unifies wavelet-based frequency decomposition with boundary-supervised spatial attention. To the best of our knowledge, this is the first lightweight design to do so across all four skip pathways of a standard ResNet-U-Net.
    
    \item \textbf{Dual-path attention in frequency domain:} LFSA for semantic channel recalibration of the low-frequency sub-band, and HFBA for boundary-aware spatial attention on the high-frequency sub-bands — each path addresses a distinct aspect of the semantic gap problem.
    
    \item \textbf{Boundary GT supervision of attention maps:} Unlike prior methods that learn attention in an unsupervised manner, we directly supervise the high-frequency attention logits with Sobel-derived boundary ground truth, providing an explicit learning signal for boundary awareness.
    
    \item \textbf{Comprehensive evaluation:} Experiments on polyp segmentation (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB) and skin lesion segmentation (ISIC 2018) benchmarks, with ablation studies isolating each component's contribution.
\end{enumerate}

% ============================================================
% SECTION 5: EXPERIMENT DESIGN
% ============================================================
\section{Experiment Design}

\subsection{Datasets}

\begin{table}[H]
\centering
\caption{Datasets for evaluation.}
\label{tab:datasets}
\begin{tabular}{lllccl}
\toprule
\textbf{Dataset} & \textbf{Domain} & \textbf{Modality} & \textbf{Train} & \textbf{Test} & \textbf{Resolution} \\
\midrule
Kvasir-SEG & Gastrointestinal polyp & Colonoscopy & 880 & 120 & $352 \times 352$ \\
CVC-ClinicDB & Gastrointestinal polyp & Colonoscopy & 550 & 62 & $352 \times 352$ \\
CVC-ColonDB & Gastrointestinal polyp & Colonoscopy & -- & 380 & $352 \times 352$ \\
ISIC 2018 & Skin lesion & Dermoscopy & 2594 & 1000 & $352 \times 352$ \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Cross-dataset generalization:} We train on Kvasir-SEG and evaluate zero-shot on CVC-ColonDB to test whether the WBS module improves generalization (not just in-distribution performance).

\subsection{Baselines}

\begin{table}[H]
\centering
\caption{Baseline methods for comparison (all using ResNet-34 backbone for fair comparison).}
\label{tab:baselines}
\small
\begin{tabular}{lll}
\toprule
\textbf{Method} & \textbf{Category} & \textbf{Why Included} \\
\midrule
U-Net~\citep{ronneberger2015unet} & Standard baseline & Foundation architecture \\
U-Net++~\citep{zhou2018unetpp} & Dense skip connections & Skip connection baseline \\
Attention U-Net~\citep{oktay2018attunet} & Attention gates & Attention-based skip filtering \\
PraNet~\citep{fan2020pranet} & Polyp-specific & Strong domain-specific model \\
U-Net v2~\citep{peng2023unetv2} & Enhanced skip connections & Direct skip connection competitor \\
TransUNet~\citep{chen2021transunet} & Transformer encoder & Transformer baseline \\
MEGANet~\citep{bui2023meganet} & Edge-guided attention & Edge-aware competitor \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Evaluation Metrics}

\begin{table}[H]
\centering
\caption{Evaluation metrics.}
\label{tab:metrics}
\begin{tabular}{lll}
\toprule
\textbf{Metric} & \textbf{Formula} & \textbf{What It Measures} \\
\midrule
Dice (DSC) & $\frac{2 \cdot TP}{2 \cdot TP + FP + FN}$ & Overall segmentation overlap \\
IoU (Jaccard) & $\frac{TP}{TP + FP + FN}$ & Intersection over union \\
Precision & $\frac{TP}{TP + FP}$ & False positive rate \\
Recall & $\frac{TP}{TP + FN}$ & False negative rate \\
HD95 & 95th \%-ile Hausdorff & \textbf{Boundary accuracy} (critical for our claims) \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Ablation Study}

\begin{table}[H]
\centering
\caption{Ablation variants to isolate each contribution.}
\label{tab:ablation}
\begin{tabular}{clp{7cm}}
\toprule
\textbf{ID} & \textbf{Variant} & \textbf{What It Tests} \\
\midrule
A1 & U-Net + Standard Skip & Baseline (no WBS module) \\
A2 & U-Net + WBS (Full) & Complete \method{} \\
A3 & WBS without HFBA & Is boundary attention needed? \\
A4 & WBS without LFSA & Is channel attention on LL needed? \\
A5 & WBS without boundary supervision & Is explicit boundary GT supervision needed? \\
A6 & Attention on raw skips (no wavelet) & \textbf{Is wavelet decomposition itself needed?} \\
A7 & WBS with Daubechies-2 wavelet & Does wavelet family matter? \\
\bottomrule
\end{tabular}
\end{table}

\textbf{A6 is the most critical ablation.} If removing the wavelet decomposition shows similar performance, the core novelty claim is weakened. This ablation proves that \textit{operating in the frequency domain} is fundamentally different from just adding more attention to raw features.

\subsection{Training Configuration}

\begin{table}[H]
\centering
\caption{Training hyperparameters.}
\label{tab:hyperparams}
\begin{tabular}{ll}
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
Optimizer & AdamW \\
Learning rate (encoder / decoder+WBS) & $1 \times 10^{-4}$ / $1 \times 10^{-3}$ \\
LR scheduler & CosineAnnealing ($T_{\max}$ = epochs) \\
Batch size & 16 \\
Epochs & 200 (polyp datasets), 100 (ISIC 2018) \\
Image size & $352 \times 352$ \\
Augmentation & HFlip, VFlip, Rotate90, ColorJitter, RandomResizedCrop \\
Boundary loss weight $\lambda$ & 0.5 \\
SE reduction ratio $r$ & 16 \\
Encoder pretrained & ImageNet (ResNet-34) \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
% SECTION 6: PARAMETER & COMPUTE ANALYSIS
% ============================================================
\section{Parameter \& Computational Analysis}

\subsection{Parameter Overhead of WBS Modules}

The WBS module adds parameters only through the LFSA and HFBA sub-modules (DWT/IDWT are parameter-free). Using the recommended \textbf{depthwise separable} HFBA variant:

\begin{table}[H]
\centering
\caption{Parameter overhead per WBS module (depthwise separable HFBA).}
\label{tab:params}
\begin{tabular}{ccccc}
\toprule
\textbf{Stage} & $C$ & \textbf{LFSA Params} & \textbf{HFBA Params} & \textbf{Total} \\
\midrule
WBS 1 & 64 & 512 & $\sim$13K & $\sim$13.5K \\
WBS 2 & 128 & 2K & $\sim$50K & $\sim$52K \\
WBS 3 & 256 & 8K & $\sim$200K & $\sim$208K \\
WBS 4 & 512 & 33K & $\sim$790K & $\sim$823K \\
\midrule
\multicolumn{4}{r}{\textbf{Total additional parameters}} & $\sim$\textbf{1.1M} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Context:} ResNet-34 has $\sim$21.8M parameters. The WBS modules add $\sim$5\% overhead, which is negligible compared to transformer-based alternatives (e.g., UDTransNet adds $\sim$8M for its DAT modules).

\subsection{Memory \& FLOPs Estimation}

\begin{itemize}[noitemsep]
    \item \textbf{DWT doubles feature maps temporarily}: At each WBS module, the DWT produces 4 sub-bands at half resolution. Net memory per module: $\sim$2$\times$ the encoder feature at that stage.
    \item \textbf{Peak GPU memory} (batch size 16, $352 \times 352$ input, ResNet-34 encoder): Estimated $\sim$\textbf{8--10 GB} based on similar architectures.
    \item \textbf{Training throughput}: Expected $\sim$15--25 images/second on a single GPU (comparable to standard U-Net with attention gates).
\end{itemize}

% ============================================================
% SECTION 7: HARDWARE REQUIREMENTS
% ============================================================
\section{Hardware Requirements}
\label{sec:hardware}

This section provides detailed hardware specifications needed to train \method{}, covering minimum requirements, recommended setup, and estimated training times.

\subsection{GPU Requirements}

\begin{table}[H]
\centering
\caption{GPU requirements for training \method{}.}
\label{tab:gpu}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X}
\toprule
\textbf{Specification} & \textbf{Minimum} & \textbf{Recommended} & \textbf{Ideal} \\
\midrule
\textbf{GPU Model} & NVIDIA RTX 3060 (12GB) & NVIDIA RTX 3090 / A5000 (24GB) & NVIDIA A100 (40/80GB) \\
\textbf{VRAM} & 12 GB & 24 GB & 40+ GB \\
\textbf{Batch Size} & 4--8 & 16 & 32+ \\
\textbf{Mixed Precision} & Required (FP16) & Recommended & Optional \\
\textbf{Multi-GPU} & Not needed & Optional (faster) & 2--4$\times$ for ablations \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Complete System Requirements}

\begin{table}[H]
\centering
\caption{Full system specifications.}
\label{tab:system}
\begin{tabular}{lll}
\toprule
\textbf{Component} & \textbf{Minimum} & \textbf{Recommended} \\
\midrule
\textbf{GPU} & 1$\times$ RTX 3060 (12GB) & 1$\times$ RTX 3090/4090 (24GB) \\
\textbf{CPU} & 8-core (Intel i7 / AMD Ryzen 7) & 16-core \\
\textbf{RAM} & 16 GB & 32 GB \\
\textbf{Storage} & 100 GB SSD & 500 GB NVMe SSD \\
\textbf{CUDA Version} & $\geq$ 11.7 & $\geq$ 12.0 \\
\textbf{PyTorch} & $\geq$ 2.0 & $\geq$ 2.1 \\
\textbf{Python} & $\geq$ 3.8 & $\geq$ 3.10 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Training Time Estimates}

\begin{table}[H]
\centering
\caption{Estimated training time per dataset (single run, single GPU).}
\label{tab:time}
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{Epochs} & \textbf{RTX 3060 (12GB)} & \textbf{RTX 3090 (24GB)} \\
\midrule
Kvasir-SEG (880 train) & 200 & $\sim$4--5 hours & $\sim$2--3 hours \\
CVC-ClinicDB (550 train) & 200 & $\sim$3--4 hours & $\sim$1.5--2 hours \\
ISIC 2018 (2594 train) & 100 & $\sim$6--8 hours & $\sim$3--4 hours \\
\midrule
\multicolumn{2}{l}{\textbf{All main experiments (4 datasets, 1 run each)}} & $\sim$15--20 hours & $\sim$8--10 hours \\
\multicolumn{2}{l}{\textbf{Ablation studies (7 variants $\times$ 1 dataset)}} & $\sim$28--35 hours & $\sim$14--21 hours \\
\multicolumn{2}{l}{\textbf{Statistical runs (3 runs per experiment)}} & $\times$3 multiplier & $\times$3 multiplier \\
\midrule
\multicolumn{2}{l}{\textbf{TOTAL ESTIMATED GPU TIME}} & \textbf{$\sim$130--165 hours} & \textbf{$\sim$66--93 hours} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Cloud GPU Alternatives}

If local GPUs are unavailable, the following cloud options are viable:

\begin{table}[H]
\centering
\caption{Cloud GPU pricing estimates (approximate, subject to change).}
\label{tab:cloud}
\begin{tabular}{llcl}
\toprule
\textbf{Platform} & \textbf{GPU} & \textbf{Cost/Hour (USD)} & \textbf{Est. Total Cost} \\
\midrule
Google Colab Pro+ & A100 (40GB) & $\sim$\$0.50--1.00 & \$50--100 \\
Kaggle (free tier) & T4/P100 (16GB) & Free (30h/week) & Free (slow) \\
Lambda Cloud & A100 (80GB) & $\sim$\$1.10 & \$75--110 \\
Vast.ai & RTX 3090 & $\sim$\$0.20--0.40 & \$15--40 \\
RunPod & RTX 4090 & $\sim$\$0.40--0.70 & \$30--70 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Software Dependencies}

\begin{table}[H]
\centering
\caption{Key Python packages.}
\label{tab:software}
\begin{tabular}{lll}
\toprule
\textbf{Package} & \textbf{Version} & \textbf{Purpose} \\
\midrule
\texttt{torch} & $\geq$ 2.0 & Deep learning framework \\
\texttt{torchvision} & $\geq$ 0.15 & Pretrained encoders \\
\texttt{segmentation-models-pytorch} & $\geq$ 0.3.3 & Baseline models \\
\texttt{pytorch-wavelets} & $\geq$ 1.3 & DWT/IDWT implementation \\
\texttt{albumentations} & $\geq$ 1.3 & Data augmentation \\
\texttt{opencv-python} & $\geq$ 4.8 & Sobel edge, image processing \\
\texttt{scipy} & $\geq$ 1.10 & HD95 metric computation \\
\texttt{wandb} or \texttt{tensorboard} & latest & Experiment tracking \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
% SECTION 8: RELATED WORK LANDSCAPE
% ============================================================
\section{Related Work Landscape}

Our work sits at the intersection of three research directions:

\subsection{Skip Connection Design}

The semantic gap in U-Net skip connections has been addressed by multiple approaches: U-Net++~\citep{zhou2018unetpp} adds dense nested skip pathways, U-Net v2~\citep{peng2023unetv2} uses Hadamard product-based semantic infusion, UDTransNet~\citep{wang2024udtransnet} employs dual-attention transformers, and SK-VM++~\citep{skvm2025} uses Mamba-based refinement. However, none of these operate in the frequency domain or explicitly target boundary information.

\subsection{Wavelet / Frequency Methods in Segmentation}

Wavelet transforms have been used in medical image analysis for feature extraction~\citep{mfrunet2025}, noise reduction~\citep{wranet2023}, and feature fusion~\citep{li2024freqfusion}. SFFNet~\citep{sffnet2024} combines Haar wavelet decomposition with dual-cross attention for remote sensing. WaveFormer~\citep{waveformer2023} reformulates self-attention in the frequency domain. However, to the best of our knowledge, \textbf{no prior work applies wavelet decomposition specifically inside all four skip pathways of a standard ResNet-U-Net} with boundary supervision.

\subsection{Boundary-Aware Segmentation}

Boundary enhancement has been explored via Laplacian operators (MEGANet~\citep{bui2023meganet}), dedicated boundary branches (BRNet~\citep{brnet2024}, BMANet~\citep{bmanet2025}), uncertainty modeling (BUNet~\citep{bunet2023}), and wavelet-based auxiliary modules (WA-NET~\citep{wanet2025}, SkinMamba~\citep{skinmamba2024}). The closest work, HCViT-Net~\citep{hcvitnet2025}, uses wavelet attention in skip connections but only at a single resolution level and without boundary supervision.

\textbf{Our differentiation:} To the best of our knowledge, \method{} is the first lightweight design to unify wavelet decomposition, dual-path attention, and boundary GT supervision in a single module applied at all four skip pathways of a standard ResNet-U-Net.

% ============================================================
% SECTION 9: PROJECT TIMELINE
% ============================================================
\section{Project Timeline}

\begin{table}[H]
\centering
\caption{7-week project timeline.}
\label{tab:timeline}
\begin{tabular}{clp{8cm}}
\toprule
\textbf{Week} & \textbf{Phase} & \textbf{Tasks} \\
\midrule
1 & Setup \& Baseline & Environment setup, download datasets, preprocess boundary GTs, train vanilla U-Net baseline \\
2 & Core Implementation & Implement DWT/IDWT, LFSA, HFBA, complete WBS module, integrate into U-Net \\
3 & Main Experiments & Train \method{} on all 4 datasets, compare with all baselines \\
4 & Ablation Studies & Run all 7 ablation variants, generate qualitative results \\
5 & Paper Writing I & Draft Introduction, Related Work, Methodology sections \\
6 & Paper Writing II & Draft Experiments, Conclusion, create all figures and tables \\
7 & Polish \& Submit & Proofreading, formatting, reference verification, submission \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
% SECTION 10: EXPECTED OUTCOMES
% ============================================================
\section{Expected Outcomes}

Based on the performance of related methods in the literature, we expect:

\begin{itemize}
    \item \textbf{Dice improvement:} +1--3\% over vanilla U-Net (ResNet-34) and competitive with or exceeding U-Net v2, Attention U-Net, and MEGANet on polyp and skin lesion benchmarks.
    \item \textbf{HD95 improvement:} Significant reduction (lower = better) due to explicit boundary supervision — this is our strongest expected signal.
    \item \textbf{Cross-dataset generalization:} Improved zero-shot performance on CVC-ColonDB when trained on Kvasir-SEG, demonstrating that wavelet-filtered skip features are less prone to dataset-specific overfitting.
    \item \textbf{Lightweight overhead:} $\sim$1.1M additional parameters ($\sim$5\% of ResNet-34), making the module practical for clinical deployment.
\end{itemize}

% ============================================================
% SECTION 11: KEY RISKS & MITIGATIONS
% ============================================================
\section{Key Risks \& Mitigations}

\begin{table}[H]
\centering
\caption{Risk assessment.}
\label{tab:risks}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{4cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X}
\toprule
\textbf{Risk} & \textbf{Impact} & \textbf{Mitigation} \\
\midrule
Ablation A6 (no wavelet) shows similar performance & Core novelty weakened & Test multiple attention baselines; if wavelet doesn't help, pivot to stronger wavelet families \\
\midrule
GPU memory overflow at batch 16 & Slower training & Use mixed precision (FP16); reduce batch to 8 with gradient accumulation \\
\midrule
Marginal Dice improvement & Paper less compelling & Focus on HD95 and qualitative boundary quality; target boundary-focused venues \\
\midrule
Concurrent competing publication & Reduced novelty & Differentiate via comprehensive ablation and plug-and-play demonstration on multiple backbones \\
\bottomrule
\end{tabularx}
\end{table}

% ============================================================
% SECTION 12: CODE STRUCTURE
% ============================================================
\section{Code Structure}

\begin{verbatim}
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
│   └── boundary_loss.py   # Per-module boundary BCEWithLogits
├── utils/
│   ├── metrics.py         # Dice, IoU, HD95 calculation
│   ├── boundary_gt.py     # Sobel edge GT generation
│   └── visualize.py       # Qualitative result visualization
├── train.py               # Training script
├── test.py                # Evaluation script
└── configs/
    ├── kvasir.yaml        # Dataset-specific config
    └── isic.yaml
\end{verbatim}

% ============================================================
% REFERENCES
% ============================================================
\section*{References}

\begingroup
\renewcommand{\section}[2]{}
\bibliographystyle{abbrvnat}
\bibliography{references}
\endgroup

% ============================================================
% BIBTEX ENTRIES — Copy to references.bib
% ============================================================
% Save the following as references.bib in the same directory.

\end{document}
```

---

## BibTeX File (`references.bib`)

Save the following content as `references.bib` in the same directory as the `.tex` file:

```bibtex
@inproceedings{ronneberger2015unet,
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  booktitle = {MICCAI},
  year      = {2015},
  pages     = {234--241}
}

@inproceedings{he2016resnet,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {CVPR},
  year      = {2016},
  pages     = {770--778}
}

@inproceedings{hu2018senet,
  author    = {Hu, Jie and Shen, Li and Sun, Gang},
  title     = {Squeeze-and-Excitation Networks},
  booktitle = {CVPR},
  year      = {2018},
  pages     = {7132--7141}
}

@inproceedings{zhou2018unetpp,
  author    = {Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  title     = {UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  booktitle = {DLMIA/ML-CDS Workshop, MICCAI},
  year      = {2018},
  pages     = {3--11}
}

@inproceedings{oktay2018attunet,
  author    = {Oktay, Ozan and Schlemper, Jo and Folgoc, Loic Le and Lee, Matthew and Heinrich, Mattias and Misawa, Kazunari and Mori, Kensaku and McDonagh, Steven and Hammerla, Nils Y and Kainz, Bernhard and others},
  title     = {Attention U-Net: Learning Where to Look for the Pancreas},
  booktitle = {MIDL},
  year      = {2018}
}

@inproceedings{fan2020pranet,
  author    = {Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
  title     = {PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
  booktitle = {MICCAI},
  year      = {2020},
  pages     = {263--273}
}

@article{chen2021transunet,
  author  = {Chen, Jieneng and Lu, Yongyi and Yu, Qiuhui and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  title   = {TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  journal = {arXiv preprint arXiv:2102.04306},
  year    = {2021}
}

@article{peng2023unetv2,
  author  = {Peng, Yaopeng and Sonber, Mike and others},
  title   = {U-Net v2: Rethinking the Skip Connections of U-Net for Medical Image Segmentation},
  journal = {arXiv preprint},
  year    = {2023}
}

@article{wang2024udtransnet,
  author  = {Wang, Haonan and others},
  title   = {UDTransNet: Narrowing Semantic Gaps in U-Net with Learnable Skip Connections via Dual Attention Transformer},
  journal = {Neural Networks},
  year    = {2024},
  volume  = {172}
}

@inproceedings{bui2023meganet,
  author    = {Bui, Nhat-Tan and Tran, Dinh-Hieu and Hoang, Xuan-Bac and Nguyen, Quang-Thuc and Phan, Minh-Triet},
  title     = {MEGANet: Multi-Scale Edge-Guided Attention Network for Weak Boundary Polyp Segmentation},
  booktitle = {IEEE WACV},
  year      = {2024},
  pages     = {7985--7994}
}

@article{li2024freqfusion,
  author  = {Li, Linwei and Zheng, Minghao and others},
  title   = {FreqFusion: Frequency-Aware Feature Fusion for Dense Image Prediction},
  journal = {IEEE TPAMI},
  year    = {2024}
}

@article{wranet2023,
  author  = {Authors},
  title   = {WRANet: Wavelet Integrated Residual Attention U-Net Network for Medical Image Segmentation},
  journal = {Complex \& Intelligent Systems},
  year    = {2023}
}

@article{mfrunet2025,
  author  = {Authors},
  title   = {MFR-UNet: Medical Image Segmentation with Fused Multi-Scale Feature Refinement Using Wavelet Transform Convolution},
  journal = {IET Systems Biology},
  year    = {2025}
}

@article{sffnet2024,
  author  = {Authors},
  title   = {SFFNet: Wavelet-Based Spatial and Frequency Domain Fusion Network},
  journal = {IEEE Trans. Geoscience and Remote Sensing},
  year    = {2024}
}

@article{waveformer2023,
  author  = {Authors},
  title   = {WaveFormer: Unlocking Fine-Grained Details with Wavelet-based High-Frequency Enhancement},
  journal = {arXiv preprint},
  year    = {2023}
}

@article{wanet2025,
  author  = {Authors},
  title   = {WA-NET: Boundary-Aware Skin Lesion Segmentation via Frequency-Spatial Fusion},
  journal = {Scientific Reports},
  year    = {2025}
}

@article{hcvitnet2025,
  author  = {Authors},
  title   = {HCViT-Net: Hybrid CNN-ViT with Wavelet-Guided Attention Refinement for Skin Lesion Segmentation},
  journal = {J. Applied Clinical Medical Physics},
  year    = {2025}
}

@article{skinmamba2024,
  author  = {Zhu, Siyuan and others},
  title   = {SkinMamba: Cross-Scale Global State Modeling with Frequency Boundary Guidance for Dermoscopic Image Segmentation},
  journal = {arXiv preprint},
  year    = {2024}
}

@article{brnet2024,
  author  = {Authors},
  title   = {BRNet: Boundary Refinement Network for Polyp Segmentation},
  journal = {IEEE Signal Processing Letters},
  year    = {2024}
}

@article{bmanet2025,
  author  = {Zhu, Wenhao and others},
  title   = {BMANet: Boundary-guided Multi-level Attention Network for Polyp Segmentation},
  journal = {Biomed. Signal Processing and Control},
  year    = {2025}
}

@article{bunet2023,
  author  = {Authors},
  title   = {BUNet: Boundary Uncertainty Aware Network for Automated Polyp Segmentation},
  journal = {Neural Networks},
  year    = {2023}
}

@article{skvm2025,
  author  = {Wu, Renkai and others},
  title   = {SK-VM++: Mamba Assists Skip-Connections for Medical Image Segmentation},
  journal = {Biomed. Signal Processing and Control},
  year    = {2025}
}

@article{rethinking2024skips,
  author  = {Authors},
  title   = {Rethinking U-net Skip Connections for Biomedical Image Segmentation},
  journal = {arXiv preprint},
  year    = {2024}
}
```

---

## Compilation Instructions

```bash
# 1. Save the LaTeX code as WBSNet_Proposal.tex
# 2. Save the BibTeX block as references.bib (same folder)
# 3. Place WBSNet_Architecture.png in a figures/ subfolder
# 4. Compile:
pdflatex WBSNet_Proposal.tex
bibtex WBSNet_Proposal
pdflatex WBSNet_Proposal.tex
pdflatex WBSNet_Proposal.tex
```
