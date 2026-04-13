# paper/figures

Images referenced from `paper/paper.tex`. Swap any of these with rendered
outputs from `diagrams/` (Mermaid or TikZ) or from
`scripts/make_paper_figures.py`.

| File                          | Referenced by                      | Source                                                   |
|-------------------------------|------------------------------------|----------------------------------------------------------|
| `wbsnet_architecture.png`     | Fig. 1 (architecture)              | `diagrams/wbsnet_architecture.tikz` or `WBSNet_Architecture.png` |
| `wbs_module.png`              | Fig. 2 (WBS module)                | `diagrams/wbs_module.tikz`                                |
| `qualitative.png`             | Fig. 3 (qualitative)               | `scripts/make_paper_figures.py --input-dir outputs/.../predictions` |
| `lambda_sensitivity.png`      | Fig. 4 (lambda sweep)              | matplotlib script on `aggregate_results.py` output        |

Until you re-render the latter three, LaTeX will show a missing-figure box
for them; the paper still compiles.
