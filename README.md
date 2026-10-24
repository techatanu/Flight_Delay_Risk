# Flight Delay Prediction (Phase 1: ML)

Predict, the night before departure, whether a flight leaving New York will arrive 15+ minutes late, and use the risk score for an offline what-if rescheduling analysis.

Course project for Advanced Machine Learning, Project 21 (Flight Delay Prediction and Rescheduling). Phase 1 covers the classical ML part: EDA, feature engineering, logistic regression and XGBoost, calibration, and the rescheduling analysis.

## Setup

- **Data:** Kaggle "Flights" dataset (copy of `nycflights13`), 336,776 departures from EWR/JFK/LGA in 2013. Put the file at `data_raw/flights.csv`.
- **Target:** `arr_delay >= 15` minutes (BTS definition). Cancelled or diverted flights have no label. They are dropped as examples but still counted in the scheduled-volume features.
- **Decision time:** the end of the previous day. Features use the published schedule plus outcomes from strictly earlier dates. A leakage test in notebook 02 checks this.
- **Split (chronological):** train Jan-Sep, validation Oct-Nov, test Dec. Every choice (hyperparameters, calibration method, rolling window, threshold) is made on validation. December is only scored.

## Results (December test set)

| Model | PR-AUC | ROC-AUC | Brier | ECE |
|---|---|---|---|---|
| Prior rate (constant) | 0.335 | 0.500 | 0.230 | 0.086 |
| Route history baseline | 0.381 | 0.554 | 0.230 | 0.095 |
| Logistic regression | 0.461 | 0.640 | 0.222 | 0.103 |
| XGBoost | 0.468 | 0.645 | 0.221 | 0.102 |

What we found:
- On validation, LR is slightly better (PR-AUC 0.297 vs 0.276). XGBoost only becomes competitive with very large leaves (`min_child_weight=2000`). Otherwise it memorises individual training days through the day-level history features.
- December is the worst month of the year (33.5% delayed) and Oct-Nov among the calmest (17.8%). A Platt calibrator fitted on Oct-Nov therefore under-predicts December: ECE goes from 0.073 to 0.102 for XGBoost. A rolling Platt recalibration on the previous 28 days brings it down to 0.047.
- With a threshold chosen on validation for precision >= 0.5, December precision holds (0.66 for XGBoost) but recall is only 0.035.
- Rescheduling: of 1,979 flights with risk >= 0.5, only 133 have a move of up to ±2 h that cuts predicted risk by >= 0.05. Most of those moves are earlier in the day.

All numbers come from `results/metrics.csv`, `results/calibration.csv` and `results/reschedule_summary.csv`. The report and slides read them through `results/tables/*.tex`.

## Project structure

```
notebooks/
  01_eda.ipynb                     data quality, target, seasonality, time of day, persistence
  02_feature_engineering.ipynb     past-only features + leakage test
  03_model_training.ipynb          LR + XGBoost tuning, calibration, rolling recalibration, ablation
  04_evaluation_calibration.ipynb  metrics, reliability, operating point, importance, LaTeX tables
  05_rescheduling.ipynb            what-if rescheduling analysis
results/      metrics, figures, tables (written by the notebooks)
report/       IEEE-format report (LaTeX + PDF)
slides/       Beamer presentation (LaTeX + PDF)
```

The notebooks must be run in order. Each one saves what the next one needs to `data/processed/` and `models/`. Those folders are not in the repo because they can be regenerated.

## How to run

Needs Python 3.11, plus a LaTeX installation for the report and slides.

```bash
make setup        # create .venv and install requirements
make all          # run notebooks 01-05 in order (~1.5 min on a laptop)
make report       # report/main.pdf
make slides       # slides/slides.pdf
```

You can also open the notebooks in Jupyter and run them top to bottom, 01 to 05.

## Limitations

- No weather forecasts are available. Day-level weather explains much of the variance, so day-ahead accuracy is limited.
- The data is one year of NYC departures only. Inbound legs of an aircraft's rotation are not visible.
- The rescheduling analysis is associational. It ignores crews, aircraft rotations and passenger connections.
