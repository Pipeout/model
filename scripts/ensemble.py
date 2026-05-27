from scripts import get_config_file, load_config

import mlflow


class Ensemble:
    def __init__(self, dfs):
        self.dfs = dfs

    def results(self, model, y_test, X_test):

        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        y_pred = model.predict(X_test)

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

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Formado", "Evadido"]
        )
        disp.plot(cmap="Blues")
        plt.title("Matriz de Confusão: Evasão Estudantil")
        plt.show()

    def run(self):
        import time

        import numpy as np
        import pandas as pd
        from lightgbm import LGBMClassifier
        from scripts import (
            clean_feature_values,
            encoding,
            model_fitting,
            results,
            selecting_active_students,
            splitting,
        )
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        start = time.time()

        df_base = pd.read_csv(self.dfs["TRAINING_DATASET"])

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

        cat_features = X_train.select_dtypes(
            include=["object", "bool", "category"]
        ).columns.tolist()

        for col in cat_features:
            X_train[col] = clean_feature_values(X_train[col])
            X_test[col] = clean_feature_values(X_test[col])

        X_train, X_test = encoding(X_train, X_test, cat_features)

        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        lgbParams = {
            "iterations": 1000,
            "learning_rate": 0.01,
            "depth": 6,
            "verbose": 0,
            "random_state": 42,
        }

        scvParams = {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        }

        rfParams = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "random_state": 42,
        }

        lrParams = {
            "random_state": 42,
        }

        estimators = [
            ("rf", RandomForestClassifier(**rfParams)),
            ("lgb", LGBMClassifier(**lgbParams)),
            ("svm", make_pipeline(StandardScaler(), SVC(**scvParams))),
        ]

        clf = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(**lrParams)
        )

        scores = cross_val_score(clf, X_train, y_train, cv=10, scoring="roc_auc")
        print(f"CV Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))

        results(clf, y_test, X_test)

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))


if __name__ == "__main__":
    with mlflow.start_run() as run:
        print(f"Experiment ID: {run.info.experiment_id}")
        CONFIG_PATH = get_config_file()
        dfs = load_config(CONFIG_PATH)
        ensemble = Ensemble(dfs)
        ensemble.run()
