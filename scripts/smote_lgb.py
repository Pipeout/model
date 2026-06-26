import logging
import os
import re
import time
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE

# Adições vitais para lidar com SMOTE sem vazamento de dados
from imblearn.pipeline import Pipeline as ImbPipeline
from lgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
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


class EvasionModel:
    ACTIVE_STATUSES = [
        "MATRICULADO NO PERÍODO",
        "AFASTAMENTO POR BLOQUEIO DE MATRICULA",
        "AFASTAMENTO POR TRANCAMENTO DE MATRICULA",
    ]

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
        X_train_encoded = pd.get_dummies(X_train, columns=cat_features)
        X_test_encoded = pd.get_dummies(X_test, columns=cat_features)
        X_train_encoded, X_test_encoded = X_train_encoded.align(
            X_test_encoded, join="left", axis=1, fill_value=0
        )
        return X_train_encoded, X_test_encoded

    @staticmethod
    def optimize_threshold(model, X_val, y_val):
        y_proba = model.predict_proba(X_val)[:, 1]
        best_thresh = 0.5
        best_score = 0.0

        thresholds = np.linspace(0.1, 0.9, 81)
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = f1_score(y_val, y_pred, average="macro")
            if score > best_score:
                best_score = score
                best_thresh = t

        print(f"\n--- Limiar de Decisão Otimizado via F1 Macro: {best_thresh:.4f} ---")
        mlflow.log_param("optimal_threshold", best_thresh)
        return best_thresh

    @staticmethod
    def results(
        model, y_test, X_test, save_path=None, prefix="", beta=2, threshold=None
    ):
        """
        Prints classification metrics (including ROC-AUC) and shows/saves
        the confusion matrix.

        threshold: if provided (and the model exposes predict_proba),
        predictions are derived from y_proba >= threshold instead of the
        model's internal default 0.5 cutoff. Without this, passing a
        tuned threshold from optimize_threshold() has zero effect on the
        reported metrics, since model.predict() always uses 0.5 internally.
        """
        has_proba = hasattr(model, "predict_proba")
        y_proba = model.predict_proba(X_test)[:, 1] if has_proba else None

        if threshold is not None:
            if not has_proba:
                raise ValueError(
                    "threshold foi informado, mas o modelo não expõe predict_proba."
                )
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        used_thresh = threshold if threshold is not None else 0.5
        print(f"\n--- Métricas Detalhadas (Limiar: {used_thresh:.4f}) ---")
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

        auprc = None
        roc_auc = None
        if y_proba is not None:
            auprc = average_precision_score(y_test, y_proba)  # Implementation of AUPRC
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"AUPRC: {auprc:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")

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
            f"fbeta_{beta}": fbeta,
            "auprc": auprc,
            "roc_auc": roc_auc,
            "threshold": used_thresh,
        }

    @staticmethod
    def plot_roc_curve(model, y_test, X_test, save_path=None):
        if not hasattr(model, "predict_proba"):
            return None

        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("Curva ROC: Evasão Estudantil")

        if save_path:
            plt.savefig(save_path)
            plt.close()
            mlflow.log_artifact(save_path)
        else:
            plt.show()

    def prepare_data(self, csv_path, calib_size=0.0, log_params=True):
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

    def run_lgbm_smote(
        self, csv_path, save_path="confusion_matrix.png", calib_size=0.2
    ):
        """
        Treina um único LightGBM dentro de um Pipeline imblearn com SMOTE
        aplicado apenas nos folds/treino (sem vazamento), no lugar do
        StackingClassifier anterior.
        """
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

        # Sem class_weight="balanced": o SMOTE já corrige a proporção das classes
        # no treino. Combinar os dois normalmente sobre-corrige o desbalanceamento.
        lgb_params = {
            "n_estimators": 1000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "verbose": -1,
            "random_state": 42,
        }

        mlflow.log_param("model_type", "LGBMClassifier_SMOTE")
        mlflow.log_param(
            "base_estimators", "LightGBM (single model with SMOTE pipeline)"
        )

        def _build_pipeline():
            # SMOTE entra dentro do Pipeline para ser refeito a cada fold de CV,
            # evitando que exemplos sintéticos contaminem o conjunto de validação.
            return ImbPipeline(
                [
                    ("smote", SMOTE(random_state=42)),
                    ("lgb", LGBMClassifier(**lgb_params)),
                ]
            )

        gkf_scoring = GroupKFold(n_splits=10)
        outer_scores = []

        for fold_i, (outer_train_idx, outer_val_idx) in enumerate(
            gkf_scoring.split(X_train, y_train, groups=groups_train)
        ):
            X_outer_train = X_train.iloc[outer_train_idx]
            y_outer_train = y_train.iloc[outer_train_idx]

            X_outer_val = X_train.iloc[outer_val_idx]
            y_outer_val = y_train.iloc[outer_val_idx]

            fold_clf = _build_pipeline()
            fold_clf.fit(X_outer_train, y_outer_train)

            y_outer_val_proba = fold_clf.predict_proba(X_outer_val)[:, 1]
            fold_auc = roc_auc_score(y_outer_val, y_outer_val_proba)
            outer_scores.append(fold_auc)

        outer_scores = np.array(outer_scores)
        mean_cv_auc = outer_scores.mean()
        print(f"CV ROC-AUC: {mean_cv_auc:.4f} (+/- {outer_scores.std():.4f})")
        mlflow.log_metric("base_cv_roc_auc_mean", mean_cv_auc)
        mlflow.log_metric("base_cv_roc_auc_std", outer_scores.std())

        clf = _build_pipeline()
        clf.fit(X_train, y_train)

        print("\n--- Resultados baseados no limiar cego padrão (0.5) ---")
        _ = self.results(clf, y_test, X_test, threshold=0.5, prefix="base_")
        self.plot_roc_curve(clf, y_test, X_test)

        opt_thresh = self.optimize_threshold(clf, X_calib, y_calib)
        print("\n--- Resultados do LightGBM (SMOTE) com Limiar Otimizado ---")
        metrics = self.results(
            clf,
            y_test,
            X_test,
            save_path=save_path,
            threshold=opt_thresh,
            prefix="tuned_",
        )

        mlflow.sklearn.log_model(clf, "smote_lgbm_model")

        return clf, opt_thresh, metrics

    def load_active_students(self, csv_path: str) -> pd.DataFrame:
        df_raw = pd.read_csv(csv_path)
        df_active = self.filter_active_students(df_raw)
        return df_active

    @staticmethod
    def latest_record_per_student(
        df_active: pd.DataFrame,
        time_col: str = "Tempo_Permanencia_Em_Semestres",
        group_col: str = "RGA_Anon",
    ) -> pd.DataFrame:
        df_sorted = df_active.sort_values([group_col, time_col])
        df_latest = df_sorted.groupby(group_col, as_index=False).tail(1).copy()
        return df_latest

    def build_inference_features(
        self, df_latest_active: pd.DataFrame, X_train: pd.DataFrame
    ) -> pd.DataFrame:
        df = df_latest_active.copy()

        cols_to_drop = [
            "RGA_Anon",
            "Situação atual",
            "Target_Evaded",
            "Idade_Ingresso",
            "IMI",
        ] + self.EXTRA_COLUMNS_TO_DROP

        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

        cat_features = df.select_dtypes(
            include=["object", "bool", "category"]
        ).columns.tolist()

        for col in cat_features:
            df[col] = df[col].astype(str)

        df[cat_features] = df[cat_features].apply(self.clean_feature_values)

        df_encoded = pd.get_dummies(df, columns=cat_features)
        _, X_inference = X_train.align(df_encoded, join="left", axis=1, fill_value=0)
        X_inference = X_inference[X_train.columns]

        X_inference = X_inference.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X_inference

    @staticmethod
    def score_active_students(
        model, df_latest_active: pd.DataFrame, X_inference: pd.DataFrame
    ) -> pd.DataFrame:
        if not hasattr(model, "predict_proba"):
            raise ValueError("O modelo precisa expor predict_proba.")

        probabilities = model.predict_proba(X_inference)[:, 1]

        df_ranking = pd.DataFrame(
            {
                "RGA_Anon": df_latest_active["RGA_Anon"].values,
                "Probabilidade_Evasao": probabilities,
            }
        ).sort_values(by="Probabilidade_Evasao", ascending=False)

        df_ranking["Nivel_Alerta"] = pd.cut(
            df_ranking["Probabilidade_Evasao"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=["Baixo", "Moderado", "Critico"],
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
        df_active = self.load_active_students(csv_path)

        if df_active.empty:
            logging.warning("Nenhum aluno ativo encontrado.")
            df_ranking = pd.DataFrame(
                columns=["RGA_Anon", "Probabilidade_Evasao", "Nivel_Alerta"]
            )
        else:
            df_latest_active = self.latest_record_per_student(df_active)
            X_inference = self.build_inference_features(df_latest_active, X_train)
            df_ranking = self.score_active_students(
                model, df_latest_active, X_inference
            )

        print("\n--- Top 10 estudantes com risco de evasão ---")
        print(df_ranking.head(10))

        output_filename = f"{training_hash}_risco_evasao.csv"
        out_path = f"{output_dir}/{output_filename}"
        df_ranking.to_csv(out_path, index=False)
        print(f"\nRanking salvo em: {out_path}")

        mlflow.log_param("risk scoring filepath", out_path)
        return df_ranking


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("evasion_risk_scoring_v1")
    with mlflow.start_run(run_name="LGBM_SMOTE") as run:
        start_time = time.time()

        model_runner = EvasionModel()
        dfs = model_runner.load_config()
        csv_path = dfs["TRAINING_DATASET"]
        results_path = dfs["RESULTS_PATH"]

        training_hash = run.info.run_id
        print(f"\n[MLflow] Started Run. Training Hash: {training_hash}")

        mlflow.set_tag("version", "v4.0_lgbm_smote")
        mlflow.set_tag("dataset", "ciencia_da_computacao")

        clf, opt_thresh, metrics = model_runner.run_lgbm_smote(csv_path)

        X_train_for_alignment, _X_test, _y_train, _y_test, _gtr, _gte = (
            model_runner.prepare_data(csv_path, calib_size=0.0, log_params=False)
        )

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
        print(f"Tempo total: {total_time:.2f} segundos")
