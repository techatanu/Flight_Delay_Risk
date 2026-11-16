"""Checks that the Phase 2 results respect the evaluation protocol.

Run from phase-2/ with `make test` (or `python -m unittest discover -s tests -v`)
after the notebooks have been executed. Tests that need generated files
(data/processed, models) are skipped when those files do not exist yet.

The pickles loaded here are files written by our own notebooks, never downloads.
"""
import ast
import json
import os
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2 = Path(__file__).resolve().parents[1]
ROOT = PHASE2.parent
RESULTS = PHASE2 / "results"
P1_DATA = ROOT / "data" / "processed"
P2_DATA = PHASE2 / "data" / "processed"
TARGET_PRECISION = 0.5

needs_p1 = unittest.skipUnless((P1_DATA / "features.pkl").exists(), "run the Phase 1 notebooks first")
needs_p2 = unittest.skipUnless((P2_DATA / "dl_preprocessing.json").exists(), "run notebooks 06-10 first")


def labelled_splits():
    df = pd.read_pickle(P1_DATA / "features.pkl")
    df = df[~df["no_arrival"]]
    return {s: df[df["split"] == s] for s in ("train", "val", "test")}


def threshold_for_precision(y, p, target):
    """Same rule as notebooks 04 and 10: lowest threshold with precision >= target."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, p)
    ok = np.where(precision[:-1] >= target)[0]
    return float(thresholds[ok[np.argmax(recall[ok])]])


class SplitTests(unittest.TestCase):
    @needs_p1
    def test_splits_are_chronological_and_disjoint(self):
        s = labelled_splits()
        self.assertLess(s["train"]["date"].max(), s["val"]["date"].min())
        self.assertLess(s["val"]["date"].max(), s["test"]["date"].min())
        ids = [set(d["id"]) for d in s.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    @needs_p1
    def test_no_post_departure_columns_are_inputs(self):
        with open(P1_DATA / "feature_columns.json") as f:
            cols = json.load(f)
        inputs = set(cols["cat"]) | set(cols["num"])
        forbidden = {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time", "delayed", "no_arrival"}
        self.assertFalse(inputs & forbidden)


class PreprocessingTests(unittest.TestCase):
    @needs_p1
    @needs_p2
    def test_scaling_statistics_come_from_training_rows_only(self):
        with open(P2_DATA / "dl_preprocessing.json") as f:
            prep = json.load(f)
        train = labelled_splits()["train"][prep["num_cols"]]
        medians = train.median()
        filled = train.fillna(medians)
        np.testing.assert_allclose(medians.to_numpy(), pd.Series(prep["medians"])[prep["num_cols"]].to_numpy())
        np.testing.assert_allclose(filled.mean().to_numpy(), pd.Series(prep["means"])[prep["num_cols"]].to_numpy())
        train_carriers = set(labelled_splits()["train"]["carrier"])
        self.assertEqual(set(prep["vocab"]["carrier"]), train_carriers)


class SelectionTests(unittest.TestCase):
    """Every choice must be the validation optimum, never a December one."""

    def test_network_configs_are_validation_argmax(self):
        tuning = pd.read_csv(RESULTS / "dl_tuning.csv")
        with open(RESULTS / "dl_selected.json") as f:
            selected = json.load(f)
        for kind in ("MLP", "Temporal"):
            best = tuning.loc[tuning[tuning["model"] == kind]["val_pr_auc"].idxmax()]
            cfg = selected[kind]
            self.assertEqual(list(ast.literal_eval(best["hidden"])), cfg["hidden"])
            for key in ("dropout", "weight_decay", "lr"):
                self.assertAlmostEqual(best[key], cfg[key])

    def test_hybrid_choices_are_validation_optima(self):
        with open(RESULTS / "hybrid_selected.json") as f:
            selected = json.load(f)
        leaf = pd.read_csv(RESULTS / "hybrid_leaf_tuning.csv")
        best = leaf.loc[leaf["val_pr_auc"].idxmax()]
        for key in ("leaf_dim", "lr", "leaf_dropout"):
            self.assertAlmostEqual(best[key], selected["leaf_hybrid"][key])
        blend = pd.read_csv(RESULTS / "hybrid_blend_weights.csv")
        self.assertAlmostEqual(blend.loc[blend["val_log_loss"].idxmin(), "w_temporal"],
                               selected["blend_weight_temporal"])

    def test_final_model_has_best_validation_score(self):
        choice = pd.read_csv(RESULTS / "phase2_model_choice.csv", index_col=0)
        phase2 = choice.drop(index=[i for i in choice.index if "Phase 1" in i])
        with open(RESULTS / "hybrid_selected.json") as f:
            final = json.load(f)["final_model"]
        self.assertEqual(phase2["val_pr_auc"].idxmax(), final)

    @needs_p1
    @needs_p2
    def test_operating_threshold_is_set_on_validation(self):
        y_val = pd.read_pickle(P1_DATA / "predictions_val.pkl")["delayed"].to_numpy()
        p_val = pd.read_pickle(P2_DATA / "dl_predictions_val.pkl")["p_temporal"].to_numpy()
        metrics = pd.read_csv(RESULTS / "metrics_phase2.csv")
        row = metrics[(metrics["model"] == "Temporal (MLP+GRU)") & (metrics["probabilities"] == "raw")].iloc[0]
        self.assertAlmostEqual(threshold_for_precision(y_val, p_val, TARGET_PRECISION), row["threshold"], places=6)


class ReportSyncTests(unittest.TestCase):
    """The report reads numbers.tex; those macros must equal the result files."""

    def setUp(self):
        text = (RESULTS / "tables" / "numbers.tex").read_text()
        self.macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text))

    def test_main_metrics_match_csv(self):
        metrics = pd.read_csv(RESULTS / "metrics_phase2.csv")
        raw = metrics[metrics["probabilities"] == "raw"].set_index("model")
        for model, key in (("XGBoost", "Xgboost"), ("Embedding MLP", "EmbeddingMlp"),
                           ("Temporal (MLP+GRU)", "Temporal"), ("Leaf hybrid", "LeafHybrid")):
            self.assertEqual(self.macros[f"{key}TestPR"], f"{raw.loc[model, 'test_pr_auc']:.3f}")
            self.assertEqual(self.macros[f"{key}RecallPfifty"], f"{raw.loc[model, 'test_recall_at_p50']:.3f}")

    def test_bootstrap_interval_matches_csv(self):
        boot = pd.read_csv(RESULTS / "bootstrap_pr_auc.csv").set_index("comparison")
        r = boot.loc["Temporal (MLP+GRU) - XGBoost"]
        self.assertEqual(self.macros["TemporalCILow"], f"{r['ci_low']:+.3f}")
        self.assertEqual(self.macros["TemporalCIHigh"], f"{r['ci_high']:+.3f}")

    def test_report_has_no_placeholders(self):
        tex = (PHASE2 / "report" / "main.tex").read_text()
        for marker in ("Member 1", "to be filled", "TODO", "XXX"):
            self.assertNotIn(marker, tex)


class SequenceLeakageTests(unittest.TestCase):
    """Rebuilding the daily history table from data cut off at day D must not change days <= D.

    This complements the scramble test in notebook 06: a table that silently used later
    rows (not only later outcomes) would fail here.
    """

    @needs_p1
    def test_daily_table_is_unchanged_by_truncation(self):
        nb = json.loads((PHASE2 / "notebooks" / "06_dl_data_pipeline.ipynb").read_text())
        ns, cwd = {}, os.getcwd()
        os.chdir(PHASE2 / "notebooks")
        try:
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if "plt.show()" in source:
                    continue
                exec(source, ns)
                if "def build_daily" in source:
                    break
        finally:
            os.chdir(cwd)
        df, build = ns["df"], ns["build_daily"]
        full = build(df)
        for cut in ("2013-03-15", "2013-11-30"):
            k = (pd.Timestamp(cut) - pd.Timestamp("2013-01-01")).days + 1
            part = build(df[df["date"] <= cut])
            np.testing.assert_allclose(part[:, :, :k], full[:, :, :k], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
