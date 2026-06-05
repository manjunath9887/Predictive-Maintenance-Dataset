import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Live Prediction",
    layout="wide"
)

st.title("🔮 Real-Time Equipment Failure Prediction")

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("best_model.pkl not found. Please train the model first.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("scaler.pkl not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =====================================================
# FEATURE INPUT SECTION
# =====================================================

st.header("⚙️ Sensor Input Parameters")

st.info(
    "Enter the current sensor readings of the equipment."
)

col1, col2, col3 = st.columns(3)

with col1:
    metric1 = st.number_input(
        "Metric 1",
        value=50.0
    )

    metric2 = st.number_input(
        "Metric 2",
        value=50.0
    )

    metric3 = st.number_input(
        "Metric 3",
        value=50.0
    )

with col2:
    metric4 = st.number_input(
        "Metric 4",
        value=50.0
    )

    metric5 = st.number_input(
        "Metric 5",
        value=50.0
    )

    metric6 = st.number_input(
        "Metric 6",
        value=50.0
    )

with col3:
    metric7 = st.number_input(
        "Metric 7",
        value=50.0
    )

    metric8 = st.number_input(
        "Metric 8",
        value=50.0
    )

    metric9 = st.number_input(
        "Metric 9",
        value=50.0
    )

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Failure Risk"):

    input_data = pd.DataFrame(
        [[
            metric1,
            metric2,
            metric3,
            metric4,
            metric5,
            metric6,
            metric7,
            metric8,
            metric9
        ]],
        columns=[
            "metric1",
            "metric2",
            "metric3",
            "metric4",
            "metric5",
            "metric6",
            "metric7",
            "metric8",
            "metric9"
        ]
    )

    scaled_input = scaler.transform(
        input_data
    )

    prediction = model.predict(
        scaled_input
    )[0]

    # ==========================================
    # PROBABILITY
    # ==========================================

    if hasattr(model, "predict_proba"):

        probability = (
            model.predict_proba(
                scaled_input
            )[0][1]
        )

    else:
        probability = float(prediction)

    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

    with col2:

        if prediction == 1:

            st.error(
                "⚠️ FAILURE LIKELY"
            )

        else:

            st.success(
                "✅ HEALTHY EQUIPMENT"
            )

    # ==========================================
    # RISK LEVEL
    # ==========================================

    st.subheader("Risk Assessment")

    if probability < 0.30:

        st.success(
            "🟢 LOW RISK"
        )

        recommendation = """
        • Continue normal operations.
        • Follow standard maintenance schedule.
        • Monitor equipment periodically.
        """

    elif probability < 0.70:

        st.warning(
            "🟡 MEDIUM RISK"
        )

        recommendation = """
        • Increase inspection frequency.
        • Check critical components.
        • Schedule preventive maintenance.
        """

    else:

        st.error(
            "🔴 HIGH RISK"
        )

        recommendation = """
        • Immediate inspection recommended.
        • Schedule maintenance urgently.
        • Consider equipment shutdown if critical.
        • Replace faulty components.
        """

    # ==========================================
    # GAUGE CHART
    # ==========================================

    st.subheader(
        "Failure Risk Gauge"
    )

    import plotly.graph_objects as go

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

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================================
    # MAINTENANCE RECOMMENDATION
    # ==========================================

    st.subheader(
        "🛠 Maintenance Recommendation"
    )

    st.info(recommendation)

    # ==========================================
    # SENSOR SUMMARY
    # ==========================================

    st.subheader(
        "📋 Input Sensor Summary"
    )

    st.dataframe(
        input_data,
        use_container_width=True
    )

# =====================================================
# BATCH PREDICTION
# =====================================================

st.divider()

st.header(
    "📂 Batch Prediction"
)

uploaded_batch = st.file_uploader(
    "Upload CSV for Batch Prediction",
    type=["csv"]
)

if uploaded_batch is not None:

    batch_df = pd.read_csv(
        uploaded_batch
    )

    feature_columns = [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9"
    ]

    batch_scaled = scaler.transform(
        batch_df[feature_columns]
    )

    batch_df["Prediction"] = model.predict(
        batch_scaled
    )

    if hasattr(model, "predict_proba"):

        batch_df["Failure_Probability"] = (
            model.predict_proba(
                batch_scaled
            )[:,1]
        )

    st.subheader(
        "Prediction Results"
    )

    st.dataframe(
        batch_df,
        use_container_width=True
    )

    csv = batch_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
