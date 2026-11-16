# Run every target from inside phase-2/.
PY  := ../.venv/bin/python
RUN := $(PY) -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600

.PHONY: setup all phase1 notebooks test report clean

# one-time: virtual environment in the repository root with the Phase 2 packages
setup:
	python3.11 -m venv ../.venv
	$(PY) -m pip install -r requirements.txt

# full Phase 2 pipeline (about 35 min on a laptop CPU)
all: phase1 notebooks test

# Phase 1 outputs that Phase 2 reads (features, LR/XGBoost models, predictions).
# The executed copies go to build/, so the graded Phase 1 notebooks are not rewritten.
phase1:
	$(RUN) --output-dir=build ../notebooks/02_feature_engineering.ipynb
	$(RUN) --output-dir=build ../notebooks/03_model_training.ipynb

# notebooks 06-10 in order; each one reads what the previous one saved
notebooks:
	$(RUN) --inplace notebooks/06_dl_data_pipeline.ipynb
	$(RUN) --inplace notebooks/07_embedding_mlp_and_temporal.ipynb
	$(RUN) --inplace notebooks/08_hybrid_models.ipynb
	$(RUN) --inplace notebooks/09_ablation_studies.ipynb
	$(RUN) --inplace notebooks/10_evaluation_calibration.ipynb

# protocol checks: chronological split, train-only scaling, validation-only choices,
# report numbers = result files, no future days in the GRU history
test:
	$(PY) -m unittest discover -s tests -v

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -rf build
	cd report && latexmk -c main.tex
