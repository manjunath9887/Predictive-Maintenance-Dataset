import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Live Prediction",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predictive Maintenance - Live Prediction")

# =====================================================
# MODEL CHECK
# =====================================================

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_PATH = "models/feature_names.pkl"

if not os.path.exists(MODEL_PATH):

    st.warning(
        "No trained model found.\n\n"
        "Please go to Model Training page and train the model first."
    )

    st.stop()

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURE_PATH)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Model Information")

st.sidebar.success(
    f"Features Loaded: {len(feature_names)}"
)

# =====================================================
# MANUAL PREDICTION
# =====================================================

st.header("⚙ Manual Sensor Input")

user_inputs = {}

col1, col2 = st.columns(2)

for idx, feature in enumerate(feature_names):

    if idx % 2 == 0:

        with col1:

            user_inputs[feature] = st.number_input(
                feature,
                value=0.0
            )

    else:

        with col2:

            user_inputs[feature] = st.number_input(
                feature,
                value=0.0
            )

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Predict Failure"):

    input_df = pd.DataFrame(
        [user_inputs]
    )

    input_scaled = scaler.transform(
        input_df
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

    with col2:

        st.metric(
            "Prediction",
            "Failure" if prediction == 1 else "Healthy"
        )

    with col3:

        risk = "HIGH" if probability > 0.7 else \
               "MEDIUM" if probability > 0.4 else \
               "LOW"

        st.metric(
            "Risk Level",
            risk
        )

    # =========================================
    # GAUGE CHART
    # =========================================

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=probability * 100,

            title={

                "text":
                "Failure Risk (%)"

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

    # =========================================
    # MAINTENANCE ADVICE
    # =========================================

    st.subheader(
        "Maintenance Recommendation"
    )

    if probability < 0.40:

        st.success(
            "Equipment operating normally. Continue routine maintenance."
        )

    elif probability < 0.70:

        st.warning(
            "Preventive inspection recommended within the next maintenance cycle."
        )

    else:

        st.error(
            "Immediate maintenance recommended. High failure probability detected."
        )

# =====================================================
# BATCH PREDICTION
# =====================================================

st.divider()

st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(
        uploaded_file
    )

    missing_cols = [

        col

        for col in feature_names

        if col not in batch_df.columns

    ]

    if missing_cols:

        st.error(
            f"Missing Columns: {missing_cols}"
        )

    else:

        X = batch_df[
            feature_names
        ]

        X_scaled = scaler.transform(X)

        batch_df["Prediction"] = (

            model.predict(
                X_scaled
            )

        )

        batch_df["Failure_Probability"] = (

            model.predict_proba(
                X_scaled
            )[:, 1]

        )

        st.dataframe(
            batch_df.head()
        )

        csv = batch_df.to_csv(
            index=False
        )

        st.download_button(

            label="Download Predictions",

            data=csv,

            file_name="prediction_results.csv",

            mime="text/csv"

        )
