import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Root Cause Analysis",
    layout="wide"
)

st.title("🔍 Root Cause Analysis Dashboard")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    st.success("Dataset Loaded Successfully")

    # ==================================================
    # BASIC INFO
    # ==================================================

    st.header("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(df)
    )

    col2.metric(
        "Total Failures",
        int(df["failure"].sum())
    )

    col3.metric(
        "Failure Rate %",
        round(
            (df["failure"].sum()/len(df))*100,
            2
        )
    )

    # ==================================================
    # NUMERIC FEATURES
    # ==================================================

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if "failure" in numeric_cols:
        numeric_cols.remove("failure")

    # ==================================================
    # FAILED VS HEALTHY COMPARISON
    # ==================================================

    st.header("⚠ Failed vs Healthy Equipment")

    failed_df = df[df["failure"] == 1]
    healthy_df = df[df["failure"] == 0]

    comparison = pd.DataFrame({

        "Failed Mean":
            failed_df[numeric_cols].mean(),

        "Healthy Mean":
            healthy_df[numeric_cols].mean(),

        "Difference":
            (
                failed_df[numeric_cols].mean()
                -
                healthy_df[numeric_cols].mean()
            )

    })

    comparison = comparison.sort_values(
        by="Difference",
        ascending=False
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    # ==================================================
    # ROOT CAUSE CONTRIBUTORS
    # ==================================================

    st.header("🎯 Top Root Cause Contributors")

    fig = px.bar(
        comparison.reset_index(),
        x="index",
        y="Difference",
        title="Sensor Impact Difference"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # SENSOR DISTRIBUTION ANALYSIS
    # ==================================================

    st.header("📈 Sensor Distribution Analysis")

    selected_sensor = st.selectbox(
        "Select Sensor",
        numeric_cols
    )

    fig = px.box(
        df,
        x="failure",
        y=selected_sensor,
        color="failure",
        title=f"{selected_sensor} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # CORRELATION TO FAILURE
    # ==================================================

    st.header("🔥 Correlation With Failure")

    corr = df.corr(
        numeric_only=True
    )

    failure_corr = (
        corr["failure"]
        .drop("failure")
        .sort_values(
            ascending=False
        )
    )

    corr_df = pd.DataFrame({

        "Feature":
            failure_corr.index,

        "Correlation":
            failure_corr.values

    })

    fig = px.bar(
        corr_df,
        x="Feature",
        y="Correlation",
        title="Feature Correlation With Failure"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ==================================================

    st.header("🌲 Feature Importance Analysis")

    X = df[numeric_cols]
    y = df["failure"]

    X = X.fillna(X.mean())

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X, y)

    importance_df = pd.DataFrame({

        "Feature":
            numeric_cols,

        "Importance":
            rf.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # Z-SCORE ANOMALY ANALYSIS
    # ==================================================

    st.header("🚨 Anomaly Detection")

    selected_feature = st.selectbox(
        "Select Feature for Anomaly Detection",
        numeric_cols,
        key="anomaly"
    )

    threshold = st.slider(
        "Z-Score Threshold",
        2.0,
        5.0,
        3.0,
        0.1
    )

    df["zscore"] = np.abs(
        zscore(
            df[selected_feature]
        )
    )

    anomalies = df[
        df["zscore"] > threshold
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Anomalies",
        len(anomalies)
    )

    col2.metric(
        "Threshold",
        threshold
    )

    fig = px.scatter(
        df,
        y=selected_feature,
        color=df["zscore"] > threshold,
        title="Anomaly Detection"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==================================================
    # SENSOR DRIFT ANALYSIS
    # ==================================================

    if "date" in df.columns:

        st.header("📉 Sensor Drift Analysis")

        sensor = st.selectbox(
            "Select Sensor for Drift",
            numeric_cols,
            key="drift"
        )

        drift_df = (
            df.groupby("date")[sensor]
            .mean()
            .reset_index()
        )

        fig = px.line(
            drift_df,
            x="date",
            y=sensor,
            title=f"{sensor} Drift Over Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==================================================
    # FAILURE PRECURSOR ANALYSIS
    # ==================================================

    if (
        "date" in df.columns and
        "device" in df.columns
    ):

        st.header("🔮 Failure Precursor Analysis")

        failure_records = df[
            df["failure"] == 1
        ]

        st.write(
            "Records immediately before failures help identify early warning signs."
        )

        st.dataframe(
            failure_records.head(20),
            use_container_width=True
        )

    # ==================================================
    # MAINTENANCE RECOMMENDATION
    # ==================================================

    st.header("🛠 Maintenance Recommendation Engine")

    top_feature = (
        importance_df
        .iloc[0]["Feature"]
    )

    st.success(
        f"""
        Primary Failure Driver: {top_feature}

        Recommended Actions:
        • Increase monitoring frequency.
        • Inspect associated subsystem.
        • Schedule preventive maintenance.
        • Configure alerts for abnormal sensor values.
        • Prioritize equipment with repeated anomalies.
        """
    )

    # ==================================================
    # ROOT CAUSE SUMMARY
    # ==================================================

    st.header("📋 Root Cause Summary")

    top5 = importance_df.head(5)

    st.dataframe(
        top5,
        use_container_width=True
    )

    st.info(
        """
        The features listed above contribute most strongly
        to equipment failures and should be monitored as
        Key Performance Indicators (KPIs).
        """
    )

else:

    st.info(
        "Please upload the predictive maintenance dataset."
    )
