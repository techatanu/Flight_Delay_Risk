# Flight Delay Prediction (Phase 2: Deep Learning)

Phase 2 of Project 21. Same question as Phase 1: on the evening before departure, will a flight leaving New York arrive 15+ minutes late? Now we test whether deep learning helps.

Everything is kept identical to Phase 1 so that the comparison is fair: the same 35 features, the same decision time, the same split (train Jan-Sep, validation Oct-Nov, test Dec) and the same metrics. The Phase 1 LR and XGBoost numbers are read from the Phase 1 output files, not retrained.

## Models

- **Embedding MLP** (PyTorch). Carrier, origin and destination get learned embedding vectors (8, 3 and 10 numbers), concatenated with the 32 numeric features, then two ReLU layers (64, 32) with dropout.
- **Temporal model** (MLP + GRU). Adds a GRU that reads the 14 days of system, airport and carrier delay statistics *before* the flight date, in time order. A leakage test in notebook 06 checks that the flight day and later days are never used.
- **Leaf hybrid** (XGBoost + network). Every flight lands in one leaf of each of the 308 Phase 1 XGBoost trees; each leaf gets a learned vector and their mean is an extra input of the temporal network.
- **Blend.** Weighted average of the temporal network and XGBoost, weight chosen on validation.

Every hyperparameter, the number of epochs (early stopping), the calibration method and the final model are chosen on validation. December is only scored.

## Results (December test set)

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Best recall at 50% precision* |
|---|---|---|---|---|
| Logistic regression (Phase 1) | 0.297 | 0.461 | 0.640 | 0.276 |
| XGBoost (Phase 1) | 0.276 | 0.468 | 0.645 | 0.292 |
| Embedding MLP | 0.293 | 0.497 | 0.661 | 0.403 |
| **Temporal (MLP + GRU)**, final | **0.298** | **0.503** | **0.670** | **0.441** |
| Leaf hybrid | 0.294 | 0.498 | 0.669 | 0.423 |
| Blend | 0.288 | 0.486 | 0.657 | 0.356 |

Network rows are seed 42. \*Read off the December precision-recall curve, so it shows what a threshold could reach, not what the deployed rule achieves (see the calibration point below).

What we found:
- Both networks beat XGBoost on December. Resampling whole days (2,000 bootstraps), the temporal model's gain over XGBoost is +0.035 with a 95% interval of [+0.006, +0.058]: above zero, but its lower end is close to zero.
- The gain is at high precision: the best recall at 50% precision on the December curve goes from 0.292 to 0.441. With the threshold fixed on validation, recall goes from 0.035 to 0.050 with raw probabilities (0.062 to 0.124 with rolling recalibration, see below).
- The GRU's extra gain over the plain MLP is small and within seed noise (temporal model 0.499 ± 0.010 over three seeds, MLP 0.497 ± 0.001).
- Ablations (notebook 09): without the 19 Phase 1 history features *and* without the GRU, December PR-AUC drops to 0.437. With the GRU reading the raw 14-day history it recovers to 0.484, so the GRU learns most of what the hand-made history features encode. Early stopping matters; dropout and weight decay barely change anything. Learned embeddings bring no measurable gain over one-hot inputs.
- Both hybrids beat XGBoost but not the temporal network.
- Calibration: fitting a calibrator on calm Oct-Nov (17.8% delayed) makes December (33.5%) worse, as in Phase 1. A rolling 28-day Platt recalibration fixes it, and at the validation-chosen threshold the temporal model then catches about twice as many delayed flights as XGBoost at similar precision.

All numbers come from `results/metrics_phase2.csv`, `results/bootstrap_pr_auc.csv` and `results/ablation_summary.csv`. The report reads them through `results/tables/*.tex`.

## Folder structure

```
phase-2/
  notebooks/
    06_dl_data_pipeline.ipynb            network inputs, 14-day history sequences + leakage test, XGBoost leaf ids
    07_embedding_mlp_and_temporal.ipynb  PyTorch embedding MLP and GRU model, hyperparameter search, 3 seeds
    08_hybrid_models.ipynb               leaf hybrid and blend, choice of the final Phase 2 model
    09_ablation_studies.ipynb            11 ablation variants x 3 seeds
    10_evaluation_calibration.ipynb      comparison with Phase 1, calibration, operating point, bootstrap, report tables
  tests/
    test_pipeline.py                     protocol checks (split, train-only scaling, validation-only choices,
                                         report numbers = result files, no future days in the GRU history)
  Makefile     runs the whole pipeline and the tests
  results/     metrics (csv/json), figures (pdf), LaTeX tables (written by the notebooks)
  report/      IEEE-format report (LaTeX + PDF)
  slides/      presentation (PowerPoint)
```

`data/` and `models/` are written by the notebooks and are not in the repo.

## How to run

Needs Python 3.11 and the Phase 1 outputs.

1. From the repository root, run the Phase 1 notebooks first (at least `02_feature_engineering` and `03_model_training`). Phase 2 reads `data/processed/features.pkl`, the Phase 1 prediction files and `models/xgb.json` from there.
2. Install the Phase 2 requirements (Phase 1 packages plus PyTorch):
   ```bash
   pip install -r phase-2/requirements.txt
   ```
3. Open `phase-2/notebooks` in Jupyter and run 06 to 10 in order, top to bottom (about 30 minutes on a laptop CPU, mostly notebooks 07 and 09).

Or run everything from inside `phase-2/` with the Makefile (expects the virtual environment at `../.venv`):

```bash
make setup    # once: creates ../.venv and installs requirements.txt
make all      # Phase 1 inputs (notebooks 02-03, executed copies go to build/), then 06-10, then the tests
make test     # protocol checks only, about 3 seconds
make report   # report/main.pdf
```

`make all` regenerates the Phase 1 files in `../data`, `../models` and `../results` (the Phase 1 notebooks themselves are not rewritten); on our machine they come out identical to the committed ones.

`make test` checks that the split is chronological and disjoint, that scaling statistics come from training rows only, that every selected setting is the validation optimum, that the report numbers equal the result files, and that rebuilding the GRU history from data cut off at a date leaves earlier days unchanged.

Note for macOS: XGBoost and PyTorch each load their own OpenMP library and crash when used in the same Python process, so XGBoost is only used in notebook 06 and PyTorch only in 07-09.

## Limitations

- One year of NYC departures and a single test month (31 independent days), so the confidence intervals are wide.
- No weather forecasts, which limits any evening-before prediction.
- The temporal model is sensitive to the random seed; a deployed version should average several seeds.
