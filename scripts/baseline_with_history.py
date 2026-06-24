from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance

if __name__ == "__main__":

    path_of_local_training_data = (
        "discentes_inativos_anonimizados.csv"
    )


    path_to_history_file = (
        "raw_ciencia_da_computacao_historico_escolar_2017_2025_1.csv"
    )

    df_inativos  = pd.read_csv(path_of_local_training_data)
    df_history = pd.read_csv(path_to_history_file)


    columns_to_drop = [

        'Data ocorrência',
        'Tempo_Permanencia_Em_Semestres',
        'Total_creditos_estrutura',
        'Reprovação_Media_Semestral',
        'Total_Creditos_Acumulados',
        'Lag_Academico_Em_Semestres',
        'Coeficiente_Rendimento',
        'Modalidade_Ensino',
        'Eficiencia_Academica',
        'Idade_Academica',
        'Ano_Ingresso',
        'Idade_Ingresso',
        'Idade_No_Semestre',
        'Total_Falhas_Gatekeeper_Acumulado',
        'Frequencia_Trend',
        'Frequencia_Rolling_3S',
        'Lag_Academico_Delta',
        'Coeficiente_Rendimento_Delta',
        'Eficiencia_Academica_Lag_01',
        'Eficiencia_Academica_Lag_02',
        'Eficiencia_Academica_Lag_03',
        'IMI', 
        'Estado Civil',
        'Coeficiente',
        'Rolling_Reprovacao_Media_3_Semestres',
        'Data nascimento',
        
    ]



    df_merged = df_inativos.merge(df_history, on='RGA_Anon')


    df_cleaned = df_merged.copy()

    existing_cols = [
        c for c in columns_to_drop
        if c in df_cleaned.columns
    ]

    df_cleaned = df_cleaned.drop(
        columns=existing_cols
    )



    # Pegar somente os inativos 
    # E mergear com o history  

    df_model = df_cleaned.copy()

    df_model["Target_Evaded"] = np.where(
        df_model["Situação atual"]
        == "EXCLUSAO POR CONCLUSAO (FORMADO)",
        0, # formado
        1, # evadido 
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

    train_idx, test_idx = next(
        splitter.split(df_model, y, groups=groups)
    )

    train_df = df_model.iloc[train_idx]
    test_df = df_model.iloc[test_idx]

    y_train = train_df["Target_Evaded"]
    y_test = test_df["Target_Evaded"]

    X_train = train_df.drop(
        columns=["Target_Evaded", "Situação atual", "RGA_Anon"]
    )

    X_test = test_df.drop(
        columns=["Target_Evaded", "Situação atual", "RGA_Anon"]
    )

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


    clf = LogisticRegression(
        random_state=42,
        max_iter=5000,
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_proba = clf.predict_proba(
        X_test
    )[:, 1]

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

    print("\nClassification Report")
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


    print(X.info())
    # result = permutation_importance(clf, X_test, y_test, n_repeats=10)

    # importance = pd.DataFrame({
    #     "feature": X_test.columns,
    #     "importance": result.importances_mean
    # }).sort_values("importance", ascending=False)

    # print(importance)

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")