import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


def split_calibration_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    calib_size: float = 0.2,
    random_state: int = 42,
):
    """
    Carves a held-out calibration set out of X_train, grouped by student id
    so no student appears in both the fit set and the calibration set.
    Must be called BEFORE encoding so the encoder never sees calibration
    rows when deciding which dummy columns to create.

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


def calibrate_model(base_model, X_calib, y_calib, method="isotonic"):
    """
    Wraps an already-fitted model so its probabilities are recalibrated on
    a held-out calibration set (X_calib/y_calib) the base model has NEVER
    seen during training. Calibrating on the training set itself would
    make the calibration look artificially good.

    Uses sklearn.frozen.FrozenEstimator (scikit-learn >= 1.6) to mark
    base_model as already-fit, so CalibratedClassifierCV only fits the
    calibration map and never refits the base model. Falls back to the
    older cv="prefit" API on older scikit-learn versions.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(base_model, method=method, cv="prefit")

    calibrated.fit(X_calib, y_calib)
    return calibrated


def find_best_threshold(model, X, y, beta=2):
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
    thresholds = np.unique(probabilities)

    scores = []
    best_threshold = 0.5
    best_score = -1

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        f2 = fbeta_score(y, predictions, beta=beta, zero_division=0)

        scores.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
            }
        )

        if f2 > best_score:
            best_score = f2
            best_threshold = threshold

    results_df = pd.DataFrame(scores)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best F{beta}: {best_score:.4f}")

    return best_threshold, best_score, results_df


if __name__ == "__main__":
    path_of_local_training_data = "discentes_inativos_anonimizados.csv"

    path_to_history_file = "raw_ciencia_da_computacao_historico_escolar_2017_2025_1.csv"

    df_inativos = pd.read_csv(path_of_local_training_data)
    df_history = pd.read_csv(path_to_history_file)

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

    # Pegar somente os inativos
    # E mergear com o history

    df_model = df_cleaned.copy()

    df_model["Target_Evaded"] = np.where(
        df_model["Situação atual"] == "EXCLUSAO POR CONCLUSAO (FORMADO)",
        0,  # formado
        1,  # evadido
    )

    groups = df_model["RGA_Anon"]

    X = df_model.drop(
        ["Target_Evaded", "Situação atual", "RGA_Anon"],
        axis=1,
    )

    print(X.info())

    y = df_model["Target_Evaded"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

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

    # --- Carve out a grouped calibration set from the training portion ---
    # (done BEFORE encoding, so the dummy columns are fit on X_fit only)
    X_fit, X_calib, y_fit, y_calib, groups_fit, groups_calib = split_calibration_set(
        X_train_full,
        y_train_full,
        groups_train_full,
        calib_size=0.2,
        random_state=42,
    )

    X_fit = pd.get_dummies(X_fit)
    X_calib = pd.get_dummies(X_calib)
    X_test = pd.get_dummies(X_test)

    X_fit, X_calib = X_fit.align(X_calib, join="left", axis=1, fill_value=0)
    X_fit, X_test = X_fit.align(X_test, join="left", axis=1, fill_value=0)
    X_calib = X_calib[X_fit.columns]

    clf = LogisticRegression(random_state=42, max_iter=5000)

    clf.fit(X_fit, y_fit)

    # --- Calibrate the model's probabilities on the held-out calibration set ---
    calibrated_clf = calibrate_model(clf, X_calib, y_calib, method="isotonic")

    # --- Find the optimal threshold using the CALIBRATED model on the
    #     calibration set (never seen during fit, never the test set) ---
    best_threshold, best_calib_f2, threshold_results_df = find_best_threshold(
        calibrated_clf,
        X_calib,
        y_calib,
        beta=2,
    )

    # --- Apply the calibrated model + optimal threshold to the test set ---
    y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_threshold).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    acc = accuracy_score(
        y_test,
        y_pred,
    )

    fbeta = fbeta_score(
        y_test,
        y_pred,
        beta=2,
    )

    auprc = average_precision_score(
        y_test,
        y_proba,
    )

    print("\nClassification Report (optimal threshold applied)")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Formado",
                "Evadido",
            ],
        )
    )

    print("\n--- Métricas Detalhadas ---")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"F2-Score:  {fbeta:.4f}")
    print(f"AUPRC:     {auprc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # --- Feature importance (Logistic Regression coefficients) ---
    # For a linear model, the fitted coefficients are the cheapest and most
    # direct measure of importance: magnitude = how strongly a feature
    # pushes the log-odds, sign = direction (positive -> pushes toward
    # "Evadido", negative -> pushes toward "Formado"). Pulled from `clf`
    # (the base model) rather than `calibrated_clf`, since the calibration
    # wrapper (isotonic/sigmoid) doesn't expose a single coefficient
    # vector in the same way.
    # feature_importance_df = (
    #     pd.DataFrame(
    #         {
    #             "feature": X_fit.columns,
    #             "coefficient": clf.coef_[0],
    #         }
    #     )
    #     .assign(abs_coefficient=lambda d: d["coefficient"].abs())
    #     .sort_values("abs_coefficient", ascending=False)
    #     .drop(columns="abs_coefficient")
    #     .reset_index(drop=True)
    # )

    # print("\n--- Top 20 Features by |Coefficient| (Logistic Regression) ---")
    # print(feature_importance_df.head(20).to_string(index=False))

    # # --- Feature importance (permutation importance, model-agnostic) ---
    # # Slower but accounts for the calibration wrapper and any feature
    # # correlations, since it measures the actual drop in test performance
    # # when a feature's values are shuffled.
    # perm_result = permutation_importance(
    #     calibrated_clf,
    #     X_test,
    #     y_test,
    #     n_repeats=10,
    #     random_state=42,
    #     scoring="average_precision",
    # )

    # perm_importance_df = (
    #     pd.DataFrame(
    #         {
    #             "feature": X_test.columns,
    #             "importance_mean": perm_result.importances_mean,
    #             "importance_std": perm_result.importances_std,
    #         }
    #     )
    #     .sort_values("importance_mean", ascending=False)
    #     .reset_index(drop=True)
    # )

    # print("\n--- Top 20 Features by Permutation Importance (AUPRC drop) ---")
    # print(perm_importance_df.head(20).to_string(index=False))
