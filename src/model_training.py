import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class ModelTrainer:

    def __init__(self, X, y):

        self.X = X
        self.y = y

        self.scaler = StandardScaler()

        self.best_model = None
        self.best_model_name = None

        self.results_df = None

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    def split_data(self):

        X_scaled = self.scaler.fit_transform(
            self.X
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test

        ) = train_test_split(

            X_scaled,
            self.y,

            test_size=0.20,

            random_state=42,

            stratify=self.y

        )

    # ==========================================
    # EVALUATION
    # ==========================================

    def evaluate_model(
        self,
        model,
        model_name
    ):

        pred = model.predict(
            self.X_test
        )

        prob = model.predict_proba(
            self.X_test
        )[:, 1]

        accuracy = accuracy_score(
            self.y_test,
            pred
        )

        precision = precision_score(
            self.y_test,
            pred,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            pred,
            zero_division=0
        )

        auc = roc_auc_score(
            self.y_test,
            prob
        )

        return {

            "Model": model_name,

            "Accuracy": round(
                accuracy,
                4
            ),

            "Precision": round(
                precision,
                4
            ),

            "Recall": round(
                recall,
                4
            ),

            "F1 Score": round(
                f1,
                4
            ),

            "ROC AUC": round(
                auc,
                4
            )

        }

    # ==========================================
    # LOGISTIC REGRESSION
    # ==========================================

    def train_logistic_regression(self):

        model = LogisticRegression(
            max_iter=2000
        )

        model.fit(
            self.X_train,
            self.y_train
        )

        return model

    # ==========================================
    # KNN
    # ==========================================

    def train_knn(self):

        param_grid = {

            "n_neighbors": [
                3,5,7,9,11
            ],

            "weights": [
                "uniform",
                "distance"
            ]

        }

        grid = GridSearchCV(

            KNeighborsClassifier(),

            param_grid,

            cv=5,

            scoring="roc_auc",

            n_jobs=-1

        )

        grid.fit(
            self.X_train,
            self.y_train
        )

        return grid.best_estimator_

    # ==========================================
    # RANDOM FOREST
    # ==========================================

    def train_random_forest(self):

        param_grid = {

            "n_estimators": [
                100,
                200,
                300
            ],

            "max_depth": [
                5,
                10,
                20,
                None
            ],

            "min_samples_split": [
                2,
                5,
                10
            ]

        }

        grid = GridSearchCV(

            RandomForestClassifier(
                random_state=42
            ),

            param_grid,

            cv=5,

            scoring="roc_auc",

            n_jobs=-1

        )

        grid.fit(
            self.X_train,
            self.y_train
        )

        return grid.best_estimator_

    # ==========================================
    # XGBOOST
    # ==========================================

    def train_xgboost(self):

        param_grid = {

            "n_estimators": [
                100,
                200
            ],

            "max_depth": [
                3,
                5,
                7
            ],

            "learning_rate": [
                0.01,
                0.05,
                0.1
            ]

        }

        grid = GridSearchCV(

            XGBClassifier(

                eval_metric="logloss",

                random_state=42

            ),

            param_grid,

            cv=5,

            scoring="roc_auc",

            n_jobs=-1

        )

        grid.fit(
            self.X_train,
            self.y_train
        )

        return grid.best_estimator_

    # ==========================================
    # TRAIN ALL
    # ==========================================

    def train_all_models(self):

        self.split_data()

        results = []

        models = {

            "Logistic Regression":
                self.train_logistic_regression(),

            "KNN":
                self.train_knn(),

            "Random Forest":
                self.train_random_forest(),

            "XGBoost":
                self.train_xgboost()

        }

        best_auc = 0

        for name, model in models.items():

            metrics = self.evaluate_model(
                model,
                name
            )

            results.append(
                metrics
            )

            if metrics["ROC AUC"] > best_auc:

                best_auc = metrics["ROC AUC"]

                self.best_model = model

                self.best_model_name = name

        self.results_df = pd.DataFrame(
            results
        )

        return self.results_df

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    def get_feature_importance(
        self,
        feature_names
    ):

        if hasattr(
            self.best_model,
            "feature_importances_"
        ):

            importance_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Importance":
                    self.best_model.feature_importances_

            })

            importance_df = (
                importance_df
                .sort_values(
                    by="Importance",
                    ascending=False
                )
            )

            return importance_df

        return None

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    def get_confusion_matrix(self):

        pred = self.best_model.predict(
            self.X_test
        )

        cm = confusion_matrix(

            self.y_test,

            pred

        )

        return cm

    # ==========================================
    # SAVE MODEL
    # ==========================================

    def save_model(

        self,

        feature_names,

        model_path="best_model.pkl",

        scaler_path="scaler.pkl",

        feature_path="feature_names.pkl"

    ):

        joblib.dump(
            self.best_model,
            model_path
        )

        joblib.dump(
            self.scaler,
            scaler_path
        )

        joblib.dump(
            feature_names,
            feature_path
        )

    # ==========================================
    # MODEL SUMMARY
    # ==========================================

    def get_best_model_summary(self):

        return {

            "Best Model":
                self.best_model_name,

            "ROC AUC":
                self.results_df[
                    self.results_df["Model"]
                    ==
                    self.best_model_name
                ]["ROC AUC"].values[0]

        }
