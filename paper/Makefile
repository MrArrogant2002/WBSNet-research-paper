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

figures:
	@mkdir -p figures
	@command -v mmdc >/dev/null 2>&1 && \
		mmdc -i ../diagrams/wbsnet_architecture.mmd -o figures/wbsnet_architecture.png -w 1600 || true
	@command -v mmdc >/dev/null 2>&1 && \
		mmdc -i ../diagrams/wbs_module.mmd          -o figures/wbs_module.png          -w 1200 || true

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all figures clean distclean
