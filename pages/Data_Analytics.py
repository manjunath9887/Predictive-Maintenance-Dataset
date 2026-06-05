import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Data Analytics",
    layout="wide"
)

st.title("📊 Predictive Maintenance - Data Analytics")

# =====================================================
# DATA LOADING
# =====================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Predictive Maintenance Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==========================================
    # PREPROCESSING
    # ==========================================

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    st.success("Dataset Loaded Successfully")

    # ==========================================
    # KPI SECTION
    # ==========================================

    st.header("📌 Key Performance Indicators")

    total_records = len(df)

    total_devices = (
        df["device"].nunique()
        if "device" in df.columns
        else 0
    )

    total_failures = (
        int(df["failure"].sum())
        if "failure" in df.columns
        else 0
    )

    failure_rate = round(
        (total_failures / total_records) * 100,
        2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Total Devices",
        total_devices
    )

    col3.metric(
        "Failure Events",
        total_failures
    )

    col4.metric(
        "Failure Rate (%)",
        failure_rate
    )

    # ==========================================
    # DATASET OVERVIEW
    # ==========================================

    st.header("📋 Dataset Overview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # ==========================================
    # DATA QUALITY
    # ==========================================

    st.header("🔎 Data Quality Assessment")

    col1, col2 = st.columns(2)

    with col1:

        missing_values = (
            df.isnull()
            .sum()
            .reset_index()
        )

        missing_values.columns = [
            "Column",
            "Missing Values"
        ]

        st.subheader("Missing Values")

        st.dataframe(
            missing_values,
            use_container_width=True
        )

    with col2:

        st.subheader("Dataset Shape")

        st.write(
            f"Rows : {df.shape[0]}"
        )

        st.write(
            f"Columns : {df.shape[1]}"
        )

        st.write(
            f"Duplicate Rows : {df.duplicated().sum()}"
        )

    # ==========================================
    # FAILURE DISTRIBUTION
    # ==========================================

    if "failure" in df.columns:

        st.header("⚠ Failure Distribution")

        failure_counts = (
            df["failure"]
            .value_counts()
            .reset_index()
        )

        failure_counts.columns = [
            "Failure Status",
            "Count"
        ]

        fig = px.pie(
            failure_counts,
            names="Failure Status",
            values="Count",
            title="Failure vs Healthy Machines"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # DEVICE FAILURE ANALYSIS
    # ==========================================

    if (
        "device" in df.columns
        and "failure" in df.columns
    ):

        st.header("🏭 Device-wise Failure Analysis")

        device_failures = (
            df.groupby("device")
            ["failure"]
            .sum()
            .reset_index()
        )

        device_failures = (
            device_failures
            .sort_values(
                by="failure",
                ascending=False
            )
        )

        fig = px.bar(
            device_failures.head(20),
            x="device",
            y="failure",
            title="Top Devices with Failures"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # TIME SERIES ANALYSIS
    # ==========================================

    if (
        "date" in df.columns
        and "failure" in df.columns
    ):

        st.header("📅 Failure Trend Analysis")

        monthly_failures = (
            df.groupby(
                df["date"]
                .dt.to_period("M")
            )["failure"]
            .sum()
            .reset_index()
        )

        monthly_failures["date"] = (
            monthly_failures["date"]
            .astype(str)
        )

        fig = px.line(
            monthly_failures,
            x="date",
            y="failure",
            markers=True,
            title="Monthly Failure Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # SENSOR ANALYSIS
    # ==========================================

    st.header("📈 Sensor Analytics")

    numeric_cols = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    sensor_cols = [
        col for col in numeric_cols
        if col != "failure"
    ]

    selected_sensor = st.selectbox(
        "Select Sensor",
        sensor_cols
    )

    fig = px.histogram(
        df,
        x=selected_sensor,
        nbins=50,
        title=f"{selected_sensor} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================
    # SENSOR TREND
    # ==========================================

    if "date" in df.columns:

        sensor_trend = st.selectbox(
            "Select Sensor Trend",
            sensor_cols,
            key="trend_sensor"
        )

        trend_df = (
            df.groupby("date")
            [sensor_trend]
            .mean()
            .reset_index()
        )

        fig = px.line(
            trend_df,
            x="date",
            y=sensor_trend,
            title=f"{sensor_trend} Trend Over Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # DESCRIPTIVE STATISTICS
    # ==========================================

    st.header("📑 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # ==========================================
    # CORRELATION MATRIX
    # ==========================================

    st.header("🔥 Correlation Analysis")

    corr = df.corr(
        numeric_only=True
    )

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================
    # TOP RISK DEVICES
    # ==========================================

    if (
        "device" in df.columns
        and "failure" in df.columns
    ):

        st.header("🚨 Top Risk Devices")

        risk_devices = (
            df.groupby("device")
            ["failure"]
            .sum()
            .reset_index()
            .sort_values(
                by="failure",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            risk_devices,
            use_container_width=True
        )

else:

    st.info(
        "Please upload the Predictive Maintenance Dataset."
    )
