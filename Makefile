.PHONY: all install test benchmarks figures paper clean help

CADET_CLI ?= $(shell which cadet-cli 2>/dev/null)
DATA_DIR  = artifacts/nl_nilt_benchmarks
FIG_DIR   = artifacts/figures
PAPER_DIR = paper

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: benchmarks figures paper ## Run everything: benchmarks -> figures -> paper

install: ## Install package with all dependencies
	pip install -e ".[all]"

test: ## Run test suite
	pytest tests/ -q

# ---------- Benchmarks ----------

benchmarks: benchmark-chen-hsu benchmark-nlnilt ## Run all benchmarks

benchmark-chen-hsu: ## Reproduce Chen & Hsu (1989) Table 1
	python scripts/chen_hsu_reproduction.py

benchmark-nlnilt: ## Run NL-NILT benchmarks (needs CADET)
ifeq ($(CADET_CLI),)
	@echo "CADET not found. Running NL-NILT only (no reference comparison)."
	@echo "Set CADET_CLI=/path/to/cadet-cli for full reproduction."
	python scripts/run_nl_nilt_benchmarks.py --cadet-cli "" --skip-cadet
else
	python scripts/run_nl_nilt_benchmarks.py --cadet-cli $(CADET_CLI)
endif

benchmark-nlnilt-no-cadet: ## Run NL-NILT benchmarks without CADET
	python scripts/run_nl_nilt_benchmarks.py --cadet-cli "" --skip-cadet

# ---------- Figures ----------

figures: ## Generate all paper figures (from precomputed data)
	python notebooks/nl_nilt_paper_figures.py --data-dir $(DATA_DIR) --output-dir $(FIG_DIR)

# ---------- Paper ----------

paper: $(PAPER_DIR)/nl_nilt_paper.pdf ## Compile the paper

$(PAPER_DIR)/nl_nilt_paper.pdf: $(PAPER_DIR)/nl_nilt_paper.tex $(FIG_DIR)/*.pdf
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode nl_nilt_paper.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode nl_nilt_paper.tex

# ---------- Clean ----------

clean: ## Remove generated artifacts
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out
	rm -f $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg $(PAPER_DIR)/*.pdf
	rm -rf __pycache__ cadet_lab/__pycache__ cadet_lab/nilt/__pycache__
	rm -rf .pytest_cache tests/__pycache__

clean-all: clean ## Remove all generated data and figures too
	rm -f $(DATA_DIR)/*.json $(DATA_DIR)/*.npz $(DATA_DIR)/*.h5
	rm -f $(FIG_DIR)/*.pdf $(FIG_DIR)/*.png
