import logging
import os
import re
import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
)

import mlflow


class EvasionModel:
    """
    Encapsulates the full student-evasion modeling pipeline: config loading,
    cleaning, splitting, encoding, single-model (LightGBM) fitting, an
    ensemble (RF + LightGBM + SVM stacked with Logistic Regression) fitting,
    probability calibration, and inference/risk-scoring for currently
    active students.

    Leakage safeguards:
      - Train/test split is grouped by RGA_Anon (student id) via
        GroupShuffleSplit, so no student's records appear in both train
        and test.
      - Cross-validation is grouped by RGA_Anon throughout. The outer CV
        score in run_ensemble is computed with a manual GroupKFold loop
        (not cross_val_score — see run_ensemble's docstring for why
        StackingClassifier can't be safely wrapped in cross_val_score with
        grouped folds), and each outer fold's StackingClassifier also gets
        its own internal stacking-CV grouped against that fold's training
        subset only.
      - A further "calibration" split is grouped the same way and carved
        out of the training data *before* model fitting, so the model
        never sees the rows used to calibrate or evaluate its
        probabilities.
      - Categorical cleaning/encoding fits dummy columns on train only,
        then aligns test/calibration/inference columns to it (missing
        dummy columns filled with 0) instead of fitting encoders on data
        outside the training split.
      - Inference rows (active students) are passed through the exact same
        clean_feature_values -> get_dummies -> align-to-X_train.columns
        pipeline as train/test, so there is no schema drift between
        training-time and inference-time features.
    """

    # Statuses that define a "currently active" student (i.e. enrolled,
    # outcome not yet known). Used both to EXCLUDE these rows from
    # train/test (selecting_active_students) and to SELECT them for
    # inference/risk-scoring (load_active_students). Defined once here so
    # both paths can never drift apart.
    ACTIVE_STATUSES = [
        "MATRICULADO NO PERÍODO",
        "AFASTAMENTO POR BLOQUEIO DE MATRICULA",
        "AFASTAMENTO POR TRANCAMENTO DE MATRICULA",
    ]

    # Columns dropped before splitting (mirrors prepare_data's
    # columns_to_drop) — also referenced by inference loading.
    EXTRA_COLUMNS_TO_DROP = [
        "Sexo",
        "Raça",
        "Estrutura",
        "Período ingresso",
        "Tipo ingresso",
        "AnoSem",
    ]

    def __init__(self, dfs=None, config_path=None):
        self.config_path = config_path or self.get_config_file()
        self.dfs = dfs

    # ------------------------------------------------------------------ #
    # Config / IO helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_config_file():
        try:
            base_dir = Path(__file__).resolve().parent.parent
            path = base_dir / "configs" / "training.yaml"
            return path
        except NameError:
            return Path("/training-app/configs/training.yaml")

    def load_config(self, config_path=None):
        config_path = config_path or self.config_path
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        try:
            current_dataset = full_config["CURRENT_DATASET"]
            logging.info(f"\nloading current dataset: {current_dataset}")
            if current_dataset not in full_config["DATASETS"]:
                raise ValueError(f"\nDataset {current_dataset} not found!")

            self.dfs = full_config["DATASETS"][current_dataset]
            return self.dfs

        except Exception as e:
            logging.exception(
                f"There was an error handling the config cleaning.yaml file {e}"
            )
            raise

    # ------------------------------------------------------------------ #
    # Cleaning / feature helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean_feature_values(col):
        def normalize(x):
            if pd.isna(x):
                return "unknown"
            x = (
                unicodedata.normalize("NFKD", x)
                .encode("ascii", "ignore")
                .decode("utf-8")
            )
            x = x.lower()
            x = re.sub(r"\s+", "_", x)
            x = re.sub(r"[^a-z0-9_]", "", x)
            return x

        return col.apply(normalize)

    @staticmethod
    def normalize_text(col):
        return (
            col.str.lower()
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
            .str.replace(r"[^a-z0-9]", "", regex=True)
        )

    @classmethod
    def selecting_active_students(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the NON-active rows (i.e. students with a known final
        outcome) with Target_Evaded computed, for training/testing.
        Active students (no outcome yet) are dropped here — see
        filter_active_students() to retrieve exactly that complementary
        set for inference.
        """
        df_ativos = df[df["Situação atual"].isin(cls.ACTIVE_STATUSES)].copy()

        df = df.drop(df_ativos.index)

        df["Target_Evaded"] = np.where(
            df["Situação atual"] == "EXCLUSAO POR CONCLUSAO (FORMADO)",
            0,
            1,
        )
        return df

    @classmethod
    def filter_active_students(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns ONLY the active rows (the complement of
        selecting_active_students) — students currently enrolled, with no
        known final outcome yet. This is the population to score for
        evasion risk.
        """
        return df[df["Situação atual"].isin(cls.ACTIVE_STATUSES)].copy()

    @staticmethod
    def splitting(
        df_base: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:

        y = df_base["Target_Evaded"]
        groups = df_base["RGA_Anon"]

        cols_to_drop = [
            "RGA_Anon",
            "Situação atual",
            "Target_Evaded",
            "Idade_Ingresso",
            "IMI",
        ]
        X = df_base.drop(columns=cols_to_drop)

        cat_features = X.select_dtypes(
            include=["object", "bool", "category"]
        ).columns.tolist()

        for col in cat_features:
            X[col] = X[col].astype(str)

        gss = GroupShuffleSplit(n_splits=2, train_size=0.7, random_state=42)
        train_idx, test_idx = next(gss.split(df_base, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

        return X_train, X_test, y_train, y_test, groups_train, groups_test

    @staticmethod
    def split_calibration_set(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
        calib_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Carves a held-out calibration set out of X_train, grouped by
        student id so no student appears in both the fit set and the
        calibration set. Must be called BEFORE encoding so the encoder
        never sees calibration rows when deciding which dummy columns to
        create.

        Returns: X_fit, X_calib, y_fit, y_calib, groups_fit, groups_calib
        """
        gss = GroupShuffleSplit(
            n_splits=2, train_size=1 - calib_size, random_state=random_state
        )
        fit_idx, calib_idx = next(gss.split(X_train, groups=groups_train))

        X_fit = X_train.iloc[fit_idx]
        X_calib = X_train.iloc[calib_idx]
        y_fit = y_train.iloc[fit_idx]
        y_calib = y_train.iloc[calib_idx]
        groups_fit = groups_train.iloc[fit_idx]
        groups_calib = groups_train.iloc[calib_idx]

        return X_fit, X_calib, y_fit, y_calib, groups_fit, groups_calib

    @staticmethod
    def encoding(X_train, X_test, cat_features):
        """
        Fits dummy columns on X_train only, then aligns X_test (or any
        other frame — a calibration split, or inference rows) to those
        columns, filling any column X_train didn't see with 0.
        """
        X_train_encoded = pd.get_dummies(X_train, columns=cat_features)
        X_test_encoded = pd.get_dummies(X_test, columns=cat_features)
        X_train_encoded, X_test_encoded = X_train_encoded.align(
            X_test_encoded, join="left", axis=1, fill_value=0
        )
        return X_train_encoded, X_test_encoded

    def save_roc_pr_curves(self, y_test, y_proba, prefix=""):
        """
        Saves ROC and Precision-Recall curves in a clean publication-style format.
        """

        # -------------------------
        # ROC CURVE
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))

        RocCurveDisplay.from_predictions(
            y_test, y_proba, ax=ax, color="orange", linewidth=2
        )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        ax.grid(True, alpha=0.3)

        roc_path = f"{prefix}roc_curve.png"
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

        # -------------------------
        # PR CURVE
        # -------------------------
        fig, ax = plt.subplots(figsize=(7, 5))

        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba, ax=ax, color="orange", linewidth=2
        )

        ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        baseline = sum(y_test) / len(y_test)
        ax.hlines(baseline, 0, 1, linestyles="--", colors="gray", linewidth=1)

        ax.grid(True, alpha=0.3)

        pr_path = f"{prefix}pr_curve.png"
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()
        return roc_path, pr_path

    @staticmethod
    def find_best_threshold(
        model,
        X,
        y,
        beta=2,
        min_threshold=0.05,
        max_threshold=0.95,
        step=0.01,
    ):
        """
        Finds the probability threshold that maximizes the F-beta score.

        Parameters
        ----------
        model : fitted classifier
            Any classifier implementing predict_proba().
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Ground-truth labels.
        beta : float, default=2
            Beta parameter of the F-beta score.
        min_threshold : float, default=0.05
            Lowest threshold to evaluate.
        max_threshold : float, default=0.95
            Highest threshold to evaluate.
        step : float, default=0.01
            Threshold increment.

        Returns
        -------
        best_threshold : float
            Threshold producing the highest F-beta score.

        best_score : float
            Best F-beta score.

        results_df : pd.DataFrame
            F-beta score for every threshold tested.
        """

        if not hasattr(model, "predict_proba"):
            raise ValueError("The supplied model does not implement predict_proba().")

        probabilities = model.predict_proba(X)[:, 1]

        thresholds = np.arange(
            min_threshold,
            max_threshold + step,
            step,
        )

        scores = []

        best_threshold = 0.5
        best_score = -1

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)

            score = fbeta_score(
                y,
                predictions,
                beta=beta,
            )

            scores.append(
                {
                    "threshold": threshold,
                    "fbeta": score,
                }
            )

            if score > best_score:
                best_score = score
                best_threshold = threshold

        results_df = pd.DataFrame(scores)

        print(f"Best threshold: {best_threshold:.2f}")
        print(f"Best F{beta}: {best_score:.4f}")

        return (
            best_threshold,
            best_score,
            results_df,
        )

    def results(
        self,
        model,
        y_test,
        X_test,
        threshold,
        save_path=None,
        prefix="",
        beta=2,
    ):
        """
        Prints classification metrics (including ROC-AUC) and shows/saves
        the confusion matrix.
        """
        y_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)

        print("\n--- Métricas Detalhadas ---")
        print(
            classification_report(
                y_test, y_pred, target_names=["Formado (0)", "Evadido (1)"]
            )
        )
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fbeta = fbeta_score(y_test, y_pred, beta=beta)

        print(f"F_{beta}: {fbeta}")

        auc = None
        auprc = None
        roc_auc = None
        if y_proba is not None:
            auprc = average_precision_score(y_test, y_proba)  # Implementation of AUPRC
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"AUPRC: {auprc:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            roc_path, pr_path = self.save_roc_pr_curves(y_test, y_proba, prefix)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)

            # Log fundamental metrics to MLflow
        mlflow.log_metric(f"{prefix}accuracy", acc)
        mlflow.log_metric(f"{prefix}precision", prec)
        mlflow.log_metric(f"{prefix}recall", rec)
        mlflow.log_metric(f"{prefix}f1_score", f1)
        mlflow.log_metric(f"{prefix}fbeta_{beta}", fbeta)
        if auprc is not None:
            mlflow.log_metric(f"{prefix}auprc", auprc)
        if roc_auc is not None:
            mlflow.log_metric(f"{prefix}roc_auc", roc_auc)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Formado", "Evadido"]
        )
        disp.plot(cmap="Blues")
        plt.title("Matriz de Confusão: Evasão Estudantil")

        if save_path:
            plt.savefig(save_path)
            plt.close()
            mlflow.log_artifact(save_path)
        else:
            plt.show()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "auprc": auprc,
        }

    @staticmethod
    def plot_roc_curve(model, y_test, X_test, save_path=None):
        if not hasattr(model, "predict_proba"):
            print("Model has no predict_proba; skipping ROC curve.")
            return None

        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("Curva ROC: Evasão Estudantil")

        if save_path:
            plt.savefig(save_path)
            plt.close()
            mlflow.log_artifact(save_path)
        else:
            plt.show()

    @staticmethod
    def calibrate_model(base_model, X_calib, y_calib, method="isotonic"):
        """
        Wraps an already-fitted model so its probabilities are recalibrated
        on a held-out calibration set (X_calib/y_calib) that the base model
        has NEVER seen during training. This avoids the common leakage
        mistake of calibrating on the training set itself, which makes
        calibration curves look artificially (and misleadingly) good.

        method: "isotonic" (flexible, needs more data) or "sigmoid"
                (Platt scaling, more stable on small calibration sets).

        Uses sklearn.frozen.FrozenEstimator (scikit-learn >= 1.6) to mark
        base_model as already-fit, so CalibratedClassifierCV only fits the
        calibration map on X_calib/y_calib and never refits/retrains the
        base model itself. Falls back to the older cv="prefit" string API
        on older scikit-learn versions where FrozenEstimator doesn't exist.
        """
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(
                FrozenEstimator(base_model), method=method
            )
        except ImportError:
            calibrated = CalibratedClassifierCV(base_model, method=method, cv="prefit")

        mlflow.log_param("calibration_method", method)
        calibrated.fit(X_calib, y_calib)
        return calibrated

    @staticmethod
    def plot_calibration_curve(
        model, y_test, X_test, n_bins=10, save_path="calibration_curve.png"
    ):
        if not hasattr(model, "predict_proba"):
            print("Model has no predict_proba; skipping calibration curve.")
            return None

        CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=n_bins)
        plt.title("Curva de Calibração: Evasão Estudantil")

        if save_path:
            plt.savefig(save_path)
            plt.close()
            mlflow.log_artifact(save_path)
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    # Shared data preparation (train/test/calibration)
    # ------------------------------------------------------------------ #
    def prepare_data(self, csv_path, calib_size=0.0, log_params=True):
        """
        Loads the CSV, filters to non-active students, drops unused
        columns, splits (grouped by student id), optionally carves a
        grouped calibration set out of the training portion, cleans
        categorical values, and one-hot encodes — fitting dummy columns on
        the training split only and aligning test/calibration to it.

        Always returns groups_train/groups_test as well, so callers can
        run grouped CV downstream.

        If calib_size > 0, returns:
            X_train, X_calib, X_test, y_train, y_calib, y_test,
            groups_train, groups_test
        Otherwise, returns:
            X_train, X_test, y_train, y_test, groups_train, groups_test
        """
        df_base = pd.read_csv(csv_path)

        df_base = self.selecting_active_students(df_base)

        df_base.drop(columns=self.EXTRA_COLUMNS_TO_DROP, inplace=True)

        X_train, X_test, y_train, y_test, groups_train, groups_test = self.splitting(
            df_base
        )
        if log_params:
            mlflow.log_param("total_dataset_rows", len(df_base))
            mlflow.log_param("test_size", len(X_test))

        X_calib = None
        y_calib = None

        if calib_size and calib_size > 0:
            (
                X_train,
                X_calib,
                y_train,
                y_calib,
                groups_train,
                _groups_calib,
            ) = self.split_calibration_set(
                X_train, y_train, groups_train, calib_size=calib_size
            )
            if log_params:
                mlflow.log_param("calibration_split_size", len(X_calib))
        else:
            if log_params:
                mlflow.log_param("calibration_split_size", 0)
        if log_params:
            mlflow.log_param("final_train_size", len(X_train))

        cat_features = X_train.select_dtypes(
            include=["object", "bool", "category"]
        ).columns.tolist()

        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cat_features] = X_train[cat_features].apply(self.clean_feature_values)
        X_test[cat_features] = X_test[cat_features].apply(self.clean_feature_values)

        if X_calib is not None:
            X_calib = X_calib.copy()
            X_calib[cat_features] = X_calib[cat_features].apply(
                self.clean_feature_values
            )

        # Fit dummy columns on TRAIN only; align test (and calibration) to
        # those columns.
        X_train_enc, X_test_enc = self.encoding(X_train, X_test, cat_features)

        if log_params:
            mlflow.log_param("feature_count_after_encoding", X_train_enc.shape[1])

        if X_calib is not None:
            _, X_calib_enc = self.encoding(X_train, X_calib, cat_features)
            return (
                X_train_enc,
                X_calib_enc,
                X_test_enc,
                y_train,
                y_calib,
                y_test,
                groups_train,
                groups_test,
            )

        return X_train_enc, X_test_enc, y_train, y_test, groups_train, groups_test

    def run_random_forest(
        self,
        csv_path,
        save_path="rf_confusion_matrix.png",
        calib_size=0.2,
    ):
        (
            X_train,
            X_calib,
            X_test,
            y_train,
            y_calib,
            y_test,
            groups_train,
            groups_test,
        ) = self.prepare_data(csv_path, calib_size=calib_size)

        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_calib = X_calib.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        params = {
            "n_estimators": 500,  # Increase for more robust bagging
            "criterion": "gini",  # "gini" or "entropy" - both work, stick to gini
            "max_depth": 10,  # Constraint to prevent overfitting the majority class
            "min_samples_split": 5,  # Increase slightly to improve generalization
            "class_weight": "balanced_subsample",  # Crucial: Adjusts weights in each bootstrap sample
            "random_state": 42,
        }

        rf = RandomForestClassifier(**params)

        gkf = GroupKFold(n_splits=10)
        auc_scores = []

        for train_idx, val_idx in gkf.split(
            X_train,
            y_train,
            groups=groups_train,
        ):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]

            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            rf_fold = RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                random_state=42,
            )

            rf_fold.fit(X_fold_train, y_fold_train)

            y_proba = rf_fold.predict_proba(X_fold_val)[:, 1]

            auc_scores.append(roc_auc_score(y_fold_val, y_proba))

        print(f"CV ROC-AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

        rf.fit(X_train, y_train)

        calibrated_rf = self.calibrate_model(
            rf,
            X_calib,
            y_calib,
        )

        best_threshold, best_f2, threshold_results = self.find_best_threshold(
            calibrated_rf,
            X_calib,
            y_calib,
            beta=2,
        )

        print(f"best threshhold: {best_threshold}")

        metrics = self.results(
            calibrated_rf,
            y_test,
            X_test,
            threshold=best_threshold,
            save_path=save_path,
        )

        self.plot_roc_curve(
            rf,
            y_test,
            X_test,
        )

        calibrated_rf = self.calibrate_model(
            rf,
            X_calib,
            y_calib,
        )

        print("\n--- Calibrated RF results ---")

        self.results(
            calibrated_rf,
            y_test,
            X_test,
            threshold=0.23,
        )

        self.plot_calibration_curve(
            calibrated_rf,
            y_test,
            X_test,
        )

        return rf, calibrated_rf, metrics

    # ------------------------------------------------------------------ #
    # Inference / risk-scoring for currently active students
    # ------------------------------------------------------------------ #
    def load_active_students(self, csv_path: str) -> pd.DataFrame:
        """
        Loads the raw CSV and returns only the rows for students who are
        currently active (no known final outcome yet) — the complement of
        selecting_active_students(). Keeps the raw columns (including
        RGA_Anon, Situação atual, Tempo_Permanencia_Em_Semestres) since
        downstream steps need them before final feature alignment.
        """
        df_raw = pd.read_csv(csv_path)
        df_active = self.filter_active_students(df_raw)
        return df_active

    @staticmethod
    def latest_record_per_student(
        df_active: pd.DataFrame,
        time_col: str = "Tempo_Permanencia_Em_Semestres",
        group_col: str = "RGA_Anon",
    ) -> pd.DataFrame:
        """
        For each active student, keeps only their most recent record
        (highest value of time_col), mirroring:

            df_inference = df.sort_values([group_col, time_col])
            df_latest = df_inference.groupby(group_col).tail(1)
        """
        df_sorted = df_active.sort_values([group_col, time_col])
        df_latest = df_sorted.groupby(group_col, as_index=False).tail(1).copy()
        return df_latest

    def build_inference_features(
        self, df_latest_active: pd.DataFrame, X_train: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Builds the feature matrix for active-student inference rows using
        the EXACT same pipeline as training: drop the same extra columns,
        clean categorical text the same way, one-hot encode, then align to
        X_train's final (already-encoded) columns.

        This avoids the schema-drift bug in the original snippet, where
        cat_features was re-derived from the already-encoded X_train
        (which has no object-dtype columns left, since get_dummies already
        ran) instead of from the raw inference frame.
        """
        df = df_latest_active.copy()

        # Drop the same non-feature columns used at training time. We keep
        # RGA_Anon out of the modeling frame (it's an id, not a feature)
        # but the caller is expected to hang on to df_latest_active's
        # RGA_Anon column separately for the ranking output.
        cols_to_drop = [
            "RGA_Anon",
            "Situação atual",
            "Target_Evaded",  # not present for active students, but drop if it exists
            "Idade_Ingresso",
            "IMI",
        ] + self.EXTRA_COLUMNS_TO_DROP

        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

        # Identify categorical columns from the RAW inference frame (not
        # from the already-encoded X_train) — this is the fix for the
        # schema-drift bug in the original snippet.
        cat_features = df.select_dtypes(
            include=["object", "bool", "category"]
        ).columns.tolist()

        for col in cat_features:
            df[col] = df[col].astype(str)

        df[cat_features] = df[cat_features].apply(self.clean_feature_values)

        # One-hot encode the inference frame, then align it to X_train's
        # already-encoded columns (same logic as encoding(), but X_train
        # is already encoded here so we align directly instead of calling
        # pd.get_dummies on X_train again).
        df_encoded = pd.get_dummies(df, columns=cat_features)
        _, X_inference = X_train.align(df_encoded, join="left", axis=1, fill_value=0)
        X_inference = X_inference[X_train.columns]

        # Mirror the SAME numeric cleanup applied to X_train/X_test in
        # run_ensemble. Without this, any inf or NaN in an active
        # student's numeric features (e.g. a ratio feature with a
        # zero denominator) reaches predict_proba untouched and sklearn
        # raises "Input X contains infinity or a value too large for
        # dtype('float32')" deep inside the stacking ensemble's base
        # learners.
        X_inference = X_inference.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X_inference

    @staticmethod
    def score_active_students(
        model, df_latest_active: pd.DataFrame, X_inference: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Scores active students with the fitted model's predict_proba and
        builds the ranking + alert-level table.
        """
        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "score_active_students requires a model with predict_proba "
                "(e.g. the stacking ensemble or its calibrated version)."
            )

        probabilities = model.predict_proba(X_inference)[:, 1]

        df_ranking = pd.DataFrame(
            {
                "RGA_Anon": df_latest_active["RGA_Anon"].values,
                "Probabilidade_Evasao": probabilities,
            }
        ).sort_values(by="Probabilidade_Evasao", ascending=False)

        df_ranking["Nivel_Alerta"] = pd.cut(
            df_ranking["Probabilidade_Evasao"],
            bins=[0, 0.4, 0.7, 0.85, 1.0],
            labels=["Baixo", "Moderado", "Grave", "Critico"],
            include_lowest=True,
        )

        return df_ranking

    def run_risk_scoring(
        self,
        csv_path: str,
        model,
        training_hash: str,
        X_train: pd.DataFrame,
        output_dir: str,
    ) -> pd.DataFrame:
        """
        Full inference pipeline for currently active students:
          1. load active (currently enrolled, outcome unknown) students
          2. keep only each student's latest record
          3. build features through the SAME pipeline as training
             (clean -> encode -> align to X_train.columns)
          4. score with model.predict_proba
          5. rank by risk and assign an alert level
          6. write the result to results/

        model: a fitted model exposing predict_proba (here, the stacking
               ensemble — or its calibrated wrapper — as decided for this
               pipeline).
        X_train: the already-encoded training features (same object
                  returned by prepare_data / run_ensemble), used purely as
                  the column-alignment reference — no training happens
                  here.
        """
        df_active = self.load_active_students(csv_path)

        if df_active.empty:
            logging.warning("No active students found; nothing to score.")
            df_ranking = pd.DataFrame(
                columns=["RGA_Anon", "Probabilidade_Evasao", "Nivel_Alerta"]
            )
        else:
            df_latest_active = self.latest_record_per_student(df_active)
            X_inference = self.build_inference_features(df_latest_active, X_train)
            df_ranking = self.score_active_students(
                model, df_latest_active, X_inference
            )

        print("\n--- Top 10 students by evasion risk ---")
        print(df_ranking.head(10))

        output_filename = f"{training_hash}_risco_evasao.csv"

        out_path = f"{output_dir}/{output_filename}"
        df_ranking.to_csv(out_path, index=False)
        print(f"\nRisk ranking written to: {out_path}")

        mlflow.log_param("risk scoring filepath", out_path)
        return


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("evasion_risk_scoring_v1")
    with mlflow.start_run(run_name="Random Forest Optimized Threshold") as run:
        start_time = time.time()

        model_runner = EvasionModel()
        dfs = model_runner.load_config()
        csv_path = dfs["TRAINING_DATASET"]
        results_path = dfs["RESULTS_PATH"]

        training_hash = run.info.run_id
        print(f"\n[MLflow] Started Run. Training Hash: {training_hash}")

        # 3. If you want to tag it for easy searching in the UI:
        mlflow.set_tag("version", "v1.0")
        mlflow.set_tag("dataset", "ciencia_da_computacao")

        # Train the stacking ensemble (with calibration + ROC-AUC reporting).
        clf, calibrated_clf, metrics = model_runner.run_random_forest(csv_path)

        # Recompute X_train (without the calibration carve-out) purely as the
        # column-alignment reference for inference, matching what `clf` was
        # actually fit on. (run_ensemble already fit `clf` on this same
        # X_train internally; we just need its columns here.)
        X_train_for_alignment, _X_test, _y_train, _y_test, _gtr, _gte = (
            model_runner.prepare_data(csv_path, calib_size=0.0, log_params=False)
        )

        # Score currently active students and write results/ranking CSV.
        model_runner.run_risk_scoring(
            csv_path,
            model=clf,
            training_hash=training_hash,
            X_train=X_train_for_alignment,
            output_dir=results_path,
        )

        mlflow.log_param("training_hash", training_hash)

        total_time = time.time() - start_time
        mlflow.log_metric("total_execution_time_seconds", total_time)
        print(f"Total training time: {total_time:.2f} seconds")
