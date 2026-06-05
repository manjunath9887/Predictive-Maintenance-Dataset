import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go


class Visualizer:

    def __init__(self):
        pass

    # =====================================================
    # FAILURE DISTRIBUTION
    # =====================================================

    def plot_failure_distribution(
        self,
        df,
        target_col="failure"
    ):

        fig = px.pie(
            df,
            names=target_col,
            title="Failure Distribution"
        )

        return fig

    # =====================================================
    # SENSOR TREND
    # =====================================================

    def plot_sensor_trend(
        self,
        df,
        date_col,
        sensor_col
    ):

        fig = px.line(
            df,
            x=date_col,
            y=sensor_col,
            title=f"{sensor_col} Trend Analysis"
        )

        return fig

    # =====================================================
    # SENSOR DRIFT
    # =====================================================

    def plot_sensor_drift(
        self,
        df,
        date_col,
        sensor_col
    ):

        drift_df = (
            df.groupby(date_col)[sensor_col]
            .mean()
            .reset_index()
        )

        fig = px.line(
            drift_df,
            x=date_col,
            y=sensor_col,
            title=f"{sensor_col} Drift Analysis"
        )

        return fig

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    def plot_correlation_heatmap(
        self,
        df
    ):

        corr = df.corr(
            numeric_only=True
        )

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Feature Correlation Heatmap"
        )

        return fig

    # =====================================================
    # FAILURE CORRELATION
    # =====================================================

    def plot_failure_correlation(
        self,
        df,
        target_col="failure"
    ):

        corr = (
            df.corr(numeric_only=True)
            [target_col]
            .drop(target_col)
            .sort_values(
                ascending=False
            )
        )

        corr_df = pd.DataFrame({

            "Feature":
                corr.index,

            "Correlation":
                corr.values

        })

        fig = px.bar(
            corr_df,
            x="Feature",
            y="Correlation",
            title="Feature Correlation with Failure"
        )

        return fig

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    def plot_feature_importance(
        self,
        importance_df,
        top_n=15
    ):

        fig = px.bar(

            importance_df.head(top_n),

            x="Feature",

            y="Importance",

            title="Feature Importance"

        )

        return fig

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    def plot_model_comparison(
        self,
        results_df
    ):

        fig = px.bar(

            results_df,

            x="Model",

            y="ROC AUC",

            title="Model Performance Comparison"

        )

        return fig

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    def plot_confusion_matrix(
        self,
        cm
    ):

        fig = px.imshow(

            cm,

            text_auto=True,

            labels=dict(

                x="Predicted",

                y="Actual"

            ),

            title="Confusion Matrix"

        )

        return fig

    # =====================================================
    # ROC CURVE
    # =====================================================

    def plot_roc_curve(
        self,
        roc_df
    ):

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=roc_df[
                    "False Positive Rate"
                ],

                y=roc_df[
                    "True Positive Rate"
                ],

                mode="lines",

                name="ROC Curve"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=[0, 1],

                y=[0, 1],

                mode="lines",

                name="Random Guess"

            )

        )

        fig.update_layout(

            title="ROC Curve",

            xaxis_title="False Positive Rate",

            yaxis_title="True Positive Rate"

        )

        return fig

    # =====================================================
    # ANOMALY DETECTION
    # =====================================================

    def plot_anomalies(
        self,
        df,
        feature_col,
        anomaly_col
    ):

        fig = px.scatter(

            df,

            x=df.index,

            y=feature_col,

            color=anomaly_col,

            title=f"Anomaly Detection - {feature_col}"

        )

        return fig

    # =====================================================
    # SENSOR DISTRIBUTION
    # =====================================================

    def plot_sensor_distribution(
        self,
        df,
        sensor_col
    ):

        fig = px.histogram(

            df,

            x=sensor_col,

            nbins=40,

            title=f"{sensor_col} Distribution"

        )

        return fig

    # =====================================================
    # FAILED VS HEALTHY COMPARISON
    # =====================================================

    def plot_failed_vs_healthy(
        self,
        df,
        sensor_col,
        target_col="failure"
    ):

        fig = px.box(

            df,

            x=target_col,

            y=sensor_col,

            color=target_col,

            title=f"{sensor_col}: Failed vs Healthy"

        )

        return fig

    # =====================================================
    # FAILURE TIMELINE
    # =====================================================

    def plot_failure_timeline(
        self,
        df,
        date_col,
        target_col="failure"
    ):

        timeline = (

            df.groupby(date_col)
            [target_col]
            .sum()
            .reset_index()

        )

        fig = px.line(

            timeline,

            x=date_col,

            y=target_col,

            title="Failure Timeline"

        )

        return fig

    # =====================================================
    # RISK GAUGE
    # =====================================================

    def plot_risk_gauge(
        self,
        probability
    ):

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability * 100,

                title={
                    "text":
                    "Failure Probability (%)"
                },

                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }

            )

        )

        return fig

    # =====================================================
    # KPI METRICS
    # =====================================================

    def calculate_kpis(
        self,
        df,
        target_col="failure"
    ):

        total_records = len(df)

        total_failures = int(
            df[target_col].sum()
        )

        failure_rate = round(

            (
                total_failures
                /
                total_records
            ) * 100,

            2

        )

        return {

            "Total Records":
                total_records,

            "Total Failures":
                total_failures,

            "Failure Rate (%)":
                failure_rate

        }
