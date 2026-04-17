# Build paper.pdf
PAPER = paper
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
BIB   = bibtex

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	-$(BIB) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

#
# `make figures` renders whatever figures are buildable from current inputs.
# Five of the paper's six figures require training outputs (aggregated CSVs
# and saved qualitative panels) that live under ../outputs/ after running the
# ablation suite on a GPU machine. On a laptop without those outputs this
# target builds the two architecture diagrams (if `mmdc` is installed) and
# skips the rest with a warning.
#
# Expected inputs (produced by scripts/run_ablation_suite.py + aggregate):
#   ../outputs/lambda_sweep.csv     -> figures/lambda_sensitivity.png
#   ../outputs/.../predictions/     -> figures/qualitative.png
#   ../outputs/subband_cache.npz    -> figures/subband_visualization.png
#   ../outputs/.../metrics.csv      -> figures/training_curves.png
#
figures:
	@mkdir -p figures
	@command -v mmdc >/dev/null 2>&1 && \
		mmdc -i ../diagrams/wbsnet_architecture.mmd -o figures/wbsnet_architecture.png -w 1600 \
		|| echo "mmdc not installed; skipping wbsnet_architecture.png"
	@command -v mmdc >/dev/null 2>&1 && \
		mmdc -i ../diagrams/wbs_module.mmd          -o figures/wbs_module.png          -w 1200 \
		|| echo "mmdc not installed; skipping wbs_module.png"
	@if [ -f ../outputs/lambda_sweep.csv ]; then \
		python ../scripts/plot_lambda_sweep.py --input ../outputs/lambda_sweep.csv \
			--output figures/lambda_sensitivity.png; \
	else echo "../outputs/lambda_sweep.csv missing; skipping lambda_sensitivity.png"; fi
	@if ls ../outputs/*/predictions/*_paper_panel.png >/dev/null 2>&1; then \
		python ../scripts/make_paper_figures.py \
			--input-dir $$(dirname $$(ls ../outputs/*/predictions/*_paper_panel.png | head -n1)) \
			--output figures/qualitative.png; \
	else echo "No qualitative panels found; skipping qualitative.png"; fi

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all figures clean distclean
