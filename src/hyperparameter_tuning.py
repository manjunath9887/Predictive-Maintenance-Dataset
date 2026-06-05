from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


class HyperparameterTuner:

    def __init__(
        self,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    ):

        self.X_train = X_train
        self.y_train = y_train

        self.cv = cv
        self.scoring = scoring

    # ==========================================
    # LOGISTIC REGRESSION
    # ==========================================

    def tune_logistic_regression(self):

        param_grid = {

            "C": [
                0.01,
                0.1,
                1,
                10,
                100
            ],

            "solver": [
                "lbfgs",
                "liblinear"
            ]
        }

        grid = GridSearchCV(

            LogisticRegression(
                max_iter=3000
            ),

            param_grid,

            cv=self.cv,

            scoring=self.scoring,

            n_jobs=-1

        )

        grid.fit(
            self.X_train,
            self.y_train
        )

        return grid.best_estimator_

    # ==========================================
    # KNN
    # ==========================================

    def tune_knn(self):

        param_grid = {

            "n_neighbors": [
                3, 5, 7, 9, 11, 13
            ],

            "weights": [
                "uniform",
                "distance"
            ],

            "metric": [
                "euclidean",
                "manhattan"
            ]
        }

        grid = GridSearchCV(

            KNeighborsClassifier(),

            param_grid,

            cv=self.cv,

            scoring=self.scoring,

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

    def tune_random_forest(self):

        param_grid = {

            "n_estimators": [
                100,
                200,
                300,
                500
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
            ],

            "min_samples_leaf": [
                1,
                2,
                4
            ]
        }

        grid = RandomizedSearchCV(

            RandomForestClassifier(
                random_state=42
            ),

            param_distributions=param_grid,

            n_iter=20,

            cv=self.cv,

            scoring=self.scoring,

            random_state=42,

            n_jobs=-1

        )

        grid.fit(
            self.X_train,
            self.y_train
        )

        return grid.best_estimator_

    # ==========================================
    # GRADIENT BOOSTING
    # ==========================================

    def tune_gradient_boosting(self):

        param_grid = {

            "n_estimators": [
                100,
                200,
                300
            ],

            "learning_rate": [
                0.01,
                0.05,
                0.1
            ],

            "max_depth": [
                3,
                5,
                7
            ]
        }

        grid = RandomizedSearchCV(

            GradientBoostingClassifier(),

            param_distributions=param_grid,

            n_iter=15,

            cv=self.cv,

            scoring=self.scoring,

            random_state=42,

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

    def tune_xgboost(self):

        param_grid = {

            "n_estimators": [
                100,
                200,
                300
            ],

            "max_depth": [
                3,
                5,
                7,
                9
            ],

            "learning_rate": [
                0.01,
                0.05,
                0.1
            ],

            "subsample": [
                0.8,
                1.0
            ],

            "colsample_bytree": [
                0.8,
                1.0
            ]
        }

        grid = RandomizedSearchCV(

            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            ),

            param_distributions=param_grid,

            n_iter=25,

            cv=self.cv,

            scoring=self.scoring,

            random_state=42,

            n_jobs=-1

        )

        grid.fit(
            self.X_train,
            self.y_train
        )

        return grid.best_estimator_

    # ==========================================
    # TUNE ALL MODELS
    # ==========================================

    def tune_all_models(self):

        models = {

            "Logistic Regression":
                self.tune_logistic_regression(),

            "KNN":
                self.tune_knn(),

            "Random Forest":
                self.tune_random_forest(),

            "Gradient Boosting":
                self.tune_gradient_boosting(),

            "XGBoost":
                self.tune_xgboost()
        }

        return models
