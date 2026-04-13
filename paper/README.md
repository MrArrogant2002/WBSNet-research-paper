# paper/

This directory contains the IEEE-conference-style manuscript for WBSNet.

## Files

| File             | Purpose                                                   |
|------------------|-----------------------------------------------------------|
| `paper.tex`      | Main manuscript (IEEEtran, `conference` option)           |
| `references.bib` | BibTeX references, one entry per citation in `paper.tex`  |
| `Makefile`       | `make` to build `paper.pdf`                                |
| `figures/`       | PNGs included by `\includegraphics` in `paper.tex`         |

## Build

```bash
cd paper
make figures        # re-renders diagrams if Mermaid CLI is available
make                # builds paper.pdf
```

Requirements:

- TeX Live 2022 or newer (`pdflatex`, `bibtex`).
- Optional: `@mermaid-js/mermaid-cli` to re-render `diagrams/*.mmd`.

## Figure inventory

The manuscript references four figures:

1. `figures/wbsnet_architecture.png` — full architecture (Fig. 1).
2. `figures/wbs_module.png` — WBS module internals (Fig. 2).
3. `figures/qualitative.png` — qualitative comparison (Fig. 3).
4. `figures/lambda_sensitivity.png` — $\lambda$ sweep (Fig. 4).

A starter architecture PNG is shipped. Re-render the others from:

- `diagrams/wbsnet_architecture.mmd` and `diagrams/wbs_module.mmd` (Mermaid),
- `scripts/make_paper_figures.py` (qualitative contact sheet),
- `scripts/plot_lambda_sweep.py` (lambda sensitivity curve).

## Numbers

All tables read their values from `outputs/aggregated/aggregated_summary.csv`
produced by `aggregate_results.py`. Until an end-to-end run has completed,
Tables I–V carry the expected values from the pre-registered design (see
`WBSNet_Research_Proposal.md` and the Related Work CSV); substitute them
with aggregated means and seeded standard deviations before submission.

## Reproducibility

- Seeds: `3407`, `3408`, `3409`.
- Configs: `configs/*_wbsnet.yaml` for main rows, `configs/ablation_*.yaml`
  for A1–A7, `configs/kvasir_colondb_generalization*.yaml` for zero-shot.
- Scripts: `train.py`, `evaluate.py`, `aggregate_results.py`,
  `scripts/significance_tests.py`, `scripts/model_complexity.py`,
  `scripts/run_ablation_suite.py`.
