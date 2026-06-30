"""
allOptimizedThreshold.py
=========================
Runs all seven already-tuned model pipelines (Baseline, SVM, Random Forest,
LightGBM, XGBoost, Ensemble Voting, Ensemble Stacking) back to back and
produces ONE combined Precision-Recall (AUPRC) comparison plot, in the same
visual style as `save_roc_pr_curves` in randomforestOptimizedThreshhold.py.

HOW THIS WORKS
--------------
Each of your six `EvasionModel`-based scripts (svm/xgb/lgbm/rf/voting/
stacking) already exposes:

    model_runner = EvasionModel()
    clf, calibrated_clf, metrics = model_runner.run_<model>(csv_path)

and an identical `prepare_data(csv_path, calib_size=...)` method built on
`GroupShuffleSplit(..., random_state=42)`. Because every script uses the
SAME csv_path + SAME random_state, calling `prepare_data` again from any
of them reproduces the exact same train/calib/test split — so we can pull
a fresh `X_test, y_test` from each module and score that module's own
calibrated model on it, without touching or duplicating your modeling
code.

The baseline script doesn't use the `EvasionModel` class (it's a flat
script using a different dataset/target prep), so it's wrapped here in a
small `run_baseline()` function that mirrors its existing `__main__`
logic 1:1 — no modeling logic was changed, only lifted into a callable.

IMPORTANT — YOU MUST EDIT THE CONFIG SECTION BELOW
----------------------------------------------------
I do not have access to your local CSV files or `configs/training.yaml`,
so I can't execute this end-to-end myself. Fill in CSV_PATH (and
BASELINE_PATHS) below with the same paths your individual scripts use,
then run:

    python allOptimizedThreshold.py

Output: `auprc_comparison_all_models.png` (combined PR curves) and a
printed/saved AUPRC ranking table.
"""

import importlib.util
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
)

import mlflow

# --------------------------------------------------------------------- #
# CONFIG — EDIT THESE PATHS FOR YOUR ENVIRONMENT
# --------------------------------------------------------------------- #
UPLOADS_DIR = ""  # folder holding the 7 *.py files

# Same csv_path your individual EvasionModel scripts load via
# model_runner.load_config()["TRAINING_DATASET"], or the hardcoded path
# used in svm/voting/stacking ("training_ciencia_da_computacao_ativos_2017_2025_1.csv").
CSV_PATH = "training_ciencia_da_computacao_ativos_2017_2025_1.csv"

# Baseline uses a different pair of raw files (see baselineOptimizedThreshold.py __main__)
BASELINE_INATIVOS_PATH = "discentes_inativos_anonimizados.csv"
BASELINE_HISTORY_PATH = "raw_ciencia_da_computacao_historico_escolar_2017_2025_1.csv"

OUTPUT_DIR = "."  # where to save the combined plot / CSV ranking


# --------------------------------------------------------------------- #
# Helper: import a sibling script as a module without running its
# `if __name__ == "__main__"` block (each script guards training behind
# that block already, so a plain import is safe and side-effect free).
# --------------------------------------------------------------------- #
def _load_module(filename, module_name):
    path = os.path.join(UPLOADS_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_one_model(module_name, filename, run_method_name, label, csv_path=CSV_PATH):
    """
    Imports one of the EvasionModel-based scripts, trains its model via
    its own run_<model>() method, then re-derives X_test/y_test from that
    same module's prepare_data() (identical split, since random_state=42
    is shared) so we can score the *calibrated* model and get y_proba for
    the combined PR plot.
    """
    print(f"\n{'=' * 70}\nTraining: {label}\n{'=' * 70}")
    t0 = time.time()

    module = _load_module(filename, module_name)
    model_runner = module.EvasionModel()

    run_method = getattr(model_runner, run_method_name)
    clf, calibrated_clf, metrics = run_method(csv_path)

    # Re-derive the same held-out test split this module's run_* method
    # already trained/evaluated against (calib_size=0.2 to match the
    # default each run_* method uses internally).
    (
        X_train,
        X_calib,
        X_test,
        y_train,
        y_calib,
        y_test,
        groups_train,
        groups_test,
    ) = model_runner.prepare_data(csv_path, calib_size=0.2)

    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_proba)

    print(f"{label} AUPRC: {auprc:.4f}  (took {time.time() - t0:.1f}s)")

    return {
        "label": label,
        "y_test": y_test,
        "y_proba": y_proba,
        "auprc": auprc,
        "metrics": metrics,
    }


def run_baseline():
    """
    Mirrors baselineOptimizedThreshold.py's __main__ logic exactly
    (same cleaning/merge/split/calibration/threshold steps), just lifted
    into a function so it slots into the same results-collection loop as
    the other six models.
    """
    print(f"\n{'=' * 70}\nTraining: Baseline (Logistic Regression)\n{'=' * 70}")
    t0 = time.time()

    baseline = _load_module("baselineOptimizedThreshold.py", "baseline_mod")

    df_inativos = pd.read_csv(BASELINE_INATIVOS_PATH)
    df_history = pd.read_csv(BASELINE_HISTORY_PATH)

    columns_to_drop = [
        "Data ocorrência",
        "Tempo_Permanencia_Em_Semestres",
        "Total_creditos_estrutura",
        "Reprovação_Media_Semestral",
        "Total_Creditos_Acumulados",
        "Lag_Academico_Em_Semestres",
        "Coeficiente_Rendimento",
        "Modalidade_Ensino",
        "Eficiencia_Academica",
        "Idade_Academica",
        "Ano_Ingresso",
        "Idade_Ingresso",
        "Idade_No_Semestre",
        "Total_Falhas_Gatekeeper_Acumulado",
        "Frequencia_Trend",
        "Frequencia_Rolling_3S",
        "Lag_Academico_Delta",
        "Coeficiente_Rendimento_Delta",
        "Eficiencia_Academica_Lag_01",
        "Eficiencia_Academica_Lag_02",
        "Eficiencia_Academica_Lag_03",
        "IMI",
        "Estado Civil",
        "Coeficiente",
        "Rolling_Reprovacao_Media_3_Semestres",
        "Data nascimento",
    ]

    df_merged = df_inativos.merge(df_history, on="RGA_Anon")
    df_cleaned = df_merged.copy()
    existing_cols = [c for c in columns_to_drop if c in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=existing_cols)

    df_model = df_cleaned.copy()
    df_model["Target_Evaded"] = np.where(
        df_model["Situação atual"] == "EXCLUSAO POR CONCLUSAO (FORMADO)", 0, 1
    )

    groups = df_model["RGA_Anon"]
    y = df_model["Target_Evaded"]

    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(splitter.split(df_model, y, groups=groups))

    train_df = df_model.iloc[train_idx]
    test_df = df_model.iloc[test_idx]

    y_train_full = train_df["Target_Evaded"]
    y_test = test_df["Target_Evaded"]
    groups_train_full = train_df["RGA_Anon"]

    X_train_full = train_df.drop(
        columns=["Target_Evaded", "Situação atual", "RGA_Anon"]
    )
    X_test = test_df.drop(columns=["Target_Evaded", "Situação atual", "RGA_Anon"])

    X_fit, X_calib, y_fit, y_calib, groups_fit, groups_calib = (
        baseline.split_calibration_set(
            X_train_full,
            y_train_full,
            groups_train_full,
            calib_size=0.2,
            random_state=42,
        )
    )

    X_fit = pd.get_dummies(X_fit)
    X_calib = pd.get_dummies(X_calib)
    X_test = pd.get_dummies(X_test)

    X_fit, X_calib = X_fit.align(X_calib, join="left", axis=1, fill_value=0)
    X_fit, X_test = X_fit.align(X_test, join="left", axis=1, fill_value=0)
    X_calib = X_calib[X_fit.columns]

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=42, max_iter=5000)
    clf.fit(X_fit, y_fit)

    calibrated_clf = baseline.calibrate_model(clf, X_calib, y_calib, method="isotonic")

    best_threshold, best_calib_f2, _ = baseline.find_best_threshold(
        calibrated_clf, X_calib, y_calib, beta=2
    )

    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_proba)

    print(f"Baseline AUPRC: {auprc:.4f}  (took {time.time() - t0:.1f}s)")

    return {
        "label": "Baseline",
        "y_test": y_test,
        "y_proba": y_proba,
        "auprc": auprc,
        "metrics": {"auprc": auprc, "best_threshold": best_threshold},
    }


def plot_combined_pr_curves(results, save_path="auprc_comparison_all_models.png"):
    """
    Overlays every model's Precision-Recall curve on one figure, sorted
    by AUPRC (best first) — same publication style as
    save_roc_pr_curves() in randomforestOptimizedThreshhold.py.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
    ]

    results_sorted = sorted(results, key=lambda r: r["auprc"], reverse=True)

    for color, r in zip(colors, results_sorted):
        PrecisionRecallDisplay.from_predictions(
            r["y_test"],
            r["y_proba"],
            ax=ax,
            color=color,
            linewidth=2,
            name=f"{r['label']} (AUPRC = {r['auprc']:.2f})",
        )

    # No-skill baseline (proportion of positives)
    pos_rate = np.mean(results_sorted[0]["y_test"])
    ax.hlines(
        pos_rate,
        0,
        1,
        linestyles="--",
        colors="gray",
        linewidth=1,
        label=f"No-skill ({pos_rate:.2f})",
    )

    ax.set_title(
        "Precision–Recall Curve Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nCombined PR/AUPRC plot saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("evasion_risk_scoring_v1")

    all_results = []

    with mlflow.start_run(run_name="All_Models_AUPRC_Comparison"):
        all_results.append(run_baseline())

        all_results.append(
            run_one_model(
                "svm_mod",
                "svmOptimizedThreshold.py",
                "run_svm",
                "SVM",
            )
        )
        all_results.append(
            run_one_model(
                "rf_mod",
                "randomforestOptimizedThreshhold.py",
                "run_random_forest",
                "Random Forest",
            )
        )
        all_results.append(
            run_one_model(
                "lgbm_mod",
                "lgbmOptimizedThreshhold.py",
                "run_lightgbm",
                "LightGBM",
            )
        )
        all_results.append(
            run_one_model(
                "xgb_mod",
                "xgbOptimizedThreshold.py",
                "run_xgboost",
                "XGBoost",
            )
        )
        all_results.append(
            run_one_model(
                "voting_mod",
                "ensemble_votingOptimizedThreshold.py",
                "run_ensemble",
                "Ensemble Voting",
            )
        )
        all_results.append(
            run_one_model(
                "stacking_mod",
                "ensemble_stackingOptimizedThreshold.py",
                "run_ensemble",
                "Ensemble Stacking",
            )
        )

        plot_path = plot_combined_pr_curves(
            all_results,
            save_path=os.path.join(OUTPUT_DIR, "auprc_comparison_all_models.png"),
        )
        mlflow.log_artifact(plot_path)

        # Ranking table
        ranking_df = (
            pd.DataFrame(
                [{"model": r["label"], "auprc": r["auprc"]} for r in all_results]
            )
            .sort_values("auprc", ascending=False)
            .reset_index(drop=True)
        )

        ranking_path = os.path.join(OUTPUT_DIR, "auprc_ranking.csv")
        ranking_df.to_csv(ranking_path, index=False)
        mlflow.log_artifact(ranking_path)

        print("\n" + "=" * 70)
        print("AUPRC RANKING (best to worst)")
        print("=" * 70)
        print(ranking_df.to_string(index=False))

        for _, row in ranking_df.iterrows():
            mlflow.log_metric(f"auprc_{row['model'].replace(' ', '_')}", row["auprc"])
