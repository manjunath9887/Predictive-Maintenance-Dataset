import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    layout="wide"
)

st.title("🔧 Predictive Maintenance Analytics Dashboard")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    # ---------------------------------------------------
    # DATA PREPROCESSING
    # ---------------------------------------------------

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------------------------
    # ANALYTICS
    # ---------------------------------------------------

    st.header("📊 Data Analytics")

    col1, col2, col3 = st.columns(3)

    total_records = len(df)

    total_failures = df["failure"].sum()

    failure_rate = (
        total_failures / total_records
    ) * 100

    col1.metric(
        "Total Records",
        total_records
    )

    col2.metric(
        "Total Failures",
        int(total_failures)
    )

    col3.metric(
        "Failure Rate %",
        round(failure_rate, 2)
    )

    # ---------------------------------------------------
    # FAILURE TREND
    # ---------------------------------------------------

    if "date" in df.columns:

        monthly = df.groupby(
            df["date"].dt.to_period("M")
        )["failure"].sum()

        monthly = monthly.reset_index()

        monthly["date"] = monthly["date"].astype(str)

        fig = px.line(
            monthly,
            x="date",
            y="failure",
            title="Monthly Failure Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------------------
    # ROOT CAUSE ANALYSIS
    # ---------------------------------------------------

    st.header("🔍 Root Cause Analysis")

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if "failure" in numeric_cols:
        numeric_cols.remove("failure")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    failed = df[df["failure"] == 1]

    healthy = df[df["failure"] == 0]

    comparison = pd.DataFrame({
        "Failed Mean":
            failed[numeric_cols].mean(),
        "Healthy Mean":
            healthy[numeric_cols].mean()
    })

    st.subheader(
        "Sensor Comparison"
    )

    st.dataframe(comparison)

    # ---------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------

    st.header("🤖 Machine Learning")

    feature_cols = [
        c for c in df.columns
        if c not in [
            "failure",
            "date",
            "device"
        ]
    ]

    X = df[feature_cols]

    y = df["failure"]

    X = X.fillna(X.mean())

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------------------------------------------------
    # MODEL TRAIN BUTTON
    # ---------------------------------------------------

    if st.button("Train Models"):

        results = []

        # ====================================
        # Linear Regression
        # ====================================

        lr = LinearRegression()

        lr.fit(
            X_train,
            y_train
        )

        pred_lr = lr.predict(X_test)

        results.append([
            "Linear Regression",
            mean_squared_error(
                y_test,
                pred_lr
            ),
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_lr
                )
            ),
            mean_absolute_error(
                y_test,
                pred_lr
            ),
            r2_score(
                y_test,
                pred_lr
            )
        ])

        # ====================================
        # KNN
        # ====================================

        knn_grid = {

            "n_neighbors":
                [3,5,7,9],

            "weights":
                ["uniform",
                 "distance"]
        }

        knn_search = GridSearchCV(
            KNeighborsRegressor(),
            knn_grid,
            cv=3,
            scoring="neg_mean_squared_error"
        )

        knn_search.fit(
            X_train,
            y_train
        )

        knn = knn_search.best_estimator_

        pred_knn = knn.predict(
            X_test
        )

        results.append([
            "KNN",
            mean_squared_error(
                y_test,
                pred_knn
            ),
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_knn
                )
            ),
            mean_absolute_error(
                y_test,
                pred_knn
            ),
            r2_score(
                y_test,
                pred_knn
            )
        ])

        # ====================================
        # Random Forest
        # ====================================

        rf_grid = {

            "n_estimators":
                [100,200],

            "max_depth":
                [5,10,None]
        }

        rf_search = GridSearchCV(
            RandomForestRegressor(
                random_state=42
            ),
            rf_grid,
            cv=3,
            scoring="neg_mean_squared_error"
        )

        rf_search.fit(
            X_train,
            y_train
        )

        rf = rf_search.best_estimator_

        pred_rf = rf.predict(
            X_test
        )

        results.append([
            "Random Forest",
            mean_squared_error(
                y_test,
                pred_rf
            ),
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_rf
                )
            ),
            mean_absolute_error(
                y_test,
                pred_rf
            ),
            r2_score(
                y_test,
                pred_rf
            )
        ])

        # ====================================
        # XGBOOST
        # ====================================

        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )

        xgb.fit(
            X_train,
            y_train
        )

        pred_xgb = xgb.predict(
            X_test
        )

        results.append([
            "XGBoost",
            mean_squared_error(
                y_test,
                pred_xgb
            ),
            np.sqrt(
                mean_squared_error(
                    y_test,
                    pred_xgb
                )
            ),
            mean_absolute_error(
                y_test,
                pred_xgb
            ),
            r2_score(
                y_test,
                pred_xgb
            )
        ])

        # ====================================
        # RESULT TABLE
        # ====================================

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "MSE",
                "RMSE",
                "MAE",
                "R2"
            ]
        )

        st.subheader(
            "Model Performance"
        )

        st.dataframe(results_df)

        best_model = results_df.sort_values(
            by="RMSE"
        ).iloc[0]["Model"]

        st.success(
            f"Best Model: {best_model}"
        )

        # ====================================
        # PERFORMANCE CHART
        # ====================================

        fig = px.bar(
            results_df,
            x="Model",
            y="RMSE",
            title="RMSE Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================
        # SAVE BEST MODEL
        # ====================================

        if best_model == "Random Forest":

            joblib.dump(
                rf,
                "best_model.pkl"
            )

            best_estimator = rf

        elif best_model == "XGBoost":

            joblib.dump(
                xgb,
                "best_model.pkl"
            )

            best_estimator = xgb

        elif best_model == "KNN":

            joblib.dump(
                knn,
                "best_model.pkl"
            )

            best_estimator = knn

        else:

            joblib.dump(
                lr,
                "best_model.pkl"
            )

            best_estimator = lr

        # ====================================
        # FEATURE IMPORTANCE
        # ====================================

        if hasattr(
            best_estimator,
            "feature_importances_"
        ):

            importance = pd.DataFrame({

                "Feature":
                    feature_cols,

                "Importance":
                    best_estimator.feature_importances_

            })

            importance = importance.sort_values(
                by="Importance",
                ascending=False
            )

            st.subheader(
                "Feature Importance"
            )

            fig = px.bar(
                importance.head(10),
                x="Feature",
                y="Importance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ====================================
        # LIVE PREDICTION
        # ====================================

        st.header(
            "🔮 Live Prediction"
        )

        sample = X.iloc[[0]]

        prediction = best_estimator.predict(
            scaler.transform(sample)
        )

        st.metric(
            "Predicted Failure Risk",
            round(
                float(prediction[0]),
                4
            )
        )

else:

    st.info(
        "Upload Predictive Maintenance Dataset"
    )
