import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import re
import unicodedata

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

import mlflow


# =========================
# MLFLOW SETUP
# =========================
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("fourthr")


# =========================
# CONFIG LOADING
# =========================
def load_config(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        full_config = yaml.safe_load(f)

    current_dataset = full_config["CURRENT_DATASET"]
    return full_config["DATASETS"][current_dataset]


def get_config_file():
    try:
        base_dir = Path(__file__).resolve().parent.parent
        return base_dir / "configs" / "training.yaml"
    except NameError:
        return Path("/training-app/configs/training.yaml")


# =========================
# CLEANING
# =========================
def clean_feature_values(col):
    def normalize(x):
        if pd.isna(x):
            return "unknown"

        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8")
        x = x.lower()
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^a-z0-9_]", "", x)

        return x

    return col.apply(normalize)


def selecting_active_students(df: pd.DataFrame) -> pd.DataFrame:
    df_ativos = df[
        df["Situação atual"].isin(
            [
                "MATRICULADO NO PERÍODO",
                "AFASTAMENTO POR BLOQUEIO DE MATRICULA",
                "AFASTAMENTO POR TRANCAMENTO DE MATRICULA",
            ]
        )
    ].copy()

    df = df.drop(df_ativos.index)

    df["Target_Evaded"] = np.where(
        df["Situação atual"] == "EXCLUSAO POR CONCLUSAO (FORMADO)", 0, 1
    )

    return df


# =========================
# SPLITTING
# =========================
def splitting(df_base: pd.DataFrame):

    y = df_base["Target_Evaded"]

    cols_to_drop = [
        "RGA_Anon",
        "Situação atual",
        "Target_Evaded",
        "Idade_Ingresso",
        "IMI",
        "Periodo_Atual",
    ]

    X = df_base.drop(columns=cols_to_drop)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, test_idx = next(gss.split(df_base, groups=df_base["RGA_Anon"]))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


# =========================
# ENCODING
# =========================
def encoding(X_train, X_test):

    cat_features = X_train.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    # clean categorical values
    for col in cat_features:
        X_train[col] = clean_feature_values(X_train[col])
        X_test[col] = clean_feature_values(X_test[col])

    # one-hot encoding
    X_train = pd.get_dummies(X_train, columns=cat_features)
    X_test = pd.get_dummies(X_test, columns=cat_features)

    # align
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_test


def sanitize(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0)


def model_fitting(X_train, y_train, params):

    model = SVC(**params)
    model.fit(X_train, y_train)

    mlflow.log_params(params)

    return model


def results(model, X_test, y_test):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Métricas ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["Formado (0)", "Evadido (1)"]
        )
    )

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Formado", "Evadido"]
    )
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.show()


if __name__ == "__main__":
    with mlflow.start_run():

        start = time.time()

        CONFIG_PATH = get_config_file()
        dfs = load_config(CONFIG_PATH)

        df_base = pd.read_csv(dfs["TRAINING_DATASET"])

        df_base = selecting_active_students(df_base)

        df_base.drop(
            columns=[
                "Sexo",
                "Raça",
                "Estrutura",
                "Período ingresso",
                "Tipo ingresso",
                "AnoSem",
            ],
            inplace=True,
        )

        X_train, X_test, y_train, y_test = splitting(df_base)

        X_train, X_test = encoding(X_train, X_test)

        # sanitize numeric issues
        X_train = sanitize(X_train)
        X_test = sanitize(X_test)

       
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(X_train)

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # SVM params
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
        }

        model = model_fitting(X_train, y_train, params)

        results(model, X_test, y_test)

        mlflow.sklearn.log_model(model, "model")

        print(f"Total time: {time.time() - start:.2f}s")