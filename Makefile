# Common tasks. Everything here runs without a GPU or a checkpoint.

PYTHON ?= python3

.PHONY: help test gateway figures numbers smoke clean

help:
	@echo "make test      the test suite"
	@echo "make gateway   build the Rust transport gateway"
	@echo "make figures   redraw every figure from the data that ships with it"
	@echo "make numbers   recompute the paper's stated numbers from the figure data"
	@echo "make smoke     run all five methods end to end on the tiny model"

test:
	$(PYTHON) -m pytest tests -q

gateway:
	cd gateway && cargo build --release && cargo test --quiet

figures:
	@for script in figures/*.py; do \
		case "$$script" in */diagram.py|*/tree_overlap.py) continue;; esac; \
		echo "== $$script"; \
		$(PYTHON) "$$script" --save >/dev/null 2>&1 || $(PYTHON) "$$script" >/dev/null || echo "   failed"; \
	done
	@ls figures/*.pdf | wc -l | xargs echo "figures written:"

numbers:
	$(PYTHON) analysis/check_paper_numbers.py

smoke:
	$(PYTHON) experiments/evaluation/overall.py --in-flight 4 --clients 2 --requests 2 \
		--max-output-tokens 12 --out results/smoke/overall.json
	$(PYTHON) analysis/stats.py results/smoke/overall.json

clean:
	rm -rf figures/*.pdf figures/*.png .shm results __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
