import pandas as pd
import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report,

    roc_curve

)


class ModelEvaluator:

    def __init__(self):

        pass

    # ==========================================
    # CLASSIFICATION METRICS
    # ==========================================

    def evaluate_classification_model(

        self,

        model,

        X_test,

        y_test,

        model_name="Model"

    ):

        y_pred = model.predict(
            X_test
        )

        if hasattr(
            model,
            "predict_proba"
        ):

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_prob = y_pred

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_prob
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
                roc_auc,
                4
            )

        }

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    def get_confusion_matrix(

        self,

        model,

        X_test,

        y_test

    ):

        y_pred = model.predict(
            X_test
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        cm_df = pd.DataFrame(

            cm,

            columns=[
                "Predicted 0",
                "Predicted 1"
            ],

            index=[
                "Actual 0",
                "Actual 1"
            ]

        )

        return cm_df

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

    def get_classification_report(

        self,

        model,

        X_test,

        y_test

    ):

        y_pred = model.predict(
            X_test
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        return report_df

    # ==========================================
    # ROC CURVE DATA
    # ==========================================

    def get_roc_curve(

        self,

        model,

        X_test,

        y_test

    ):

        if hasattr(
            model,
            "predict_proba"
        ):

            y_prob = model.predict_proba(
                X_test
            )[:, 1]

        else:

            return None

        fpr, tpr, thresholds = roc_curve(

            y_test,

            y_prob

        )

        roc_df = pd.DataFrame({

            "False Positive Rate": fpr,

            "True Positive Rate": tpr,

            "Threshold": thresholds

        })

        return roc_df

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    def get_feature_importance(

        self,

        model,

        feature_names

    ):

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Importance":
                    model.feature_importances_

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
    # MULTI MODEL COMPARISON
    # ==========================================

    def compare_models(

        self,

        models,

        X_test,

        y_test

    ):

        results = []

        for name, model in models.items():

            metrics = (

                self.evaluate_classification_model(

                    model,

                    X_test,

                    y_test,

                    name

                )

            )

            results.append(
                metrics
            )

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(

            by="ROC AUC",

            ascending=False

        )

        return results_df

    # ==========================================
    # BEST MODEL
    # ==========================================

    def get_best_model(

        self,

        comparison_df

    ):

        best = comparison_df.iloc[0]

        return {

            "Best Model":
                best["Model"],

            "Accuracy":
                best["Accuracy"],

            "Precision":
                best["Precision"],

            "Recall":
                best["Recall"],

            "F1 Score":
                best["F1 Score"],

            "ROC AUC":
                best["ROC AUC"]

        }

    # ==========================================
    # FAILURE DETECTION SUMMARY
    # ==========================================

    def failure_detection_summary(

        self,

        model,

        X_test,

        y_test

    ):

        y_pred = model.predict(
            X_test
        )

        total_records = len(
            y_test
        )

        failures_detected = np.sum(
            y_pred == 1
        )

        actual_failures = np.sum(
            y_test == 1
        )

        summary = {

            "Total Records":
                total_records,

            "Actual Failures":
                int(actual_failures),

            "Predicted Failures":
                int(failures_detected)

        }

        return summary
