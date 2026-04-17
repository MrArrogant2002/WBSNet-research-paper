# WBSNet Diagrams

This folder contains the source for all architecture figures used in the
paper. We provide both Mermaid (quick preview in GitHub, VS Code, Obsidian)
and TikZ (publication-quality LaTeX) versions of every figure.

## Files

| File                           | Purpose                                               |
|--------------------------------|-------------------------------------------------------|
| `wbsnet_architecture.mmd`      | Mermaid: full WBSNet encoder, WBS skips, decoder, loss |
| `wbs_module.mmd`               | Mermaid: inside of a single WBS module                |
| `wbsnet_architecture.tikz`     | TikZ: camera-ready version of the full architecture   |
| `wbs_module.tikz`              | TikZ: camera-ready version of the WBS module          |

The canonical drawing `docs/WBSNet_Architecture.drawio` +
`docs/WBSNet_Architecture.png` is the paper's Fig. 1 source; the TikZ and
Mermaid files above give you a pure-text alternative that survives
LaTeX-only or VCS-only pipelines.

## Rendering

### Mermaid

```bash
# One-off, command line:
npm install -g @mermaid-js/mermaid-cli
mmdc -i wbsnet_architecture.mmd -o wbsnet_architecture.png -w 1600
mmdc -i wbs_module.mmd          -o wbs_module.png          -w 1200
```

### TikZ

Drop either `.tikz` file into a LaTeX document:

```latex
\documentclass{article}
\usepackage{tikz}
\usepackage[margin=1cm]{geometry}
\begin{document}
\input{wbsnet_architecture.tikz}
\end{document}
```

To embed directly in `paper.tex`, replace the matching
`\includegraphics{figures/wbsnet_architecture.png}` line with
`\input{../diagrams/wbsnet_architecture.tikz}`.

### Paper PNGs

`paper/figures/*.png` are the rendered assets referenced by `paper/paper.tex`.
If you change a diagram source, re-export and copy the PNG into
`paper/figures/` so the paper builds out of the box.
