import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

st.set_page_config(
    page_title="Model Training",
    layout="wide"
)

st.title("🤖 Predictive Maintenance Model Training")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")

    # ============================================
    # Feature Preparation
    # ============================================

    features = [
        c for c in df.columns
        if c not in [
            "failure",
            "device",
            "date"
        ]
    ]

    X = df[features]

    y = df["failure"]

    X = X.fillna(X.mean())

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.subheader("Dataset Split")

    col1, col2 = st.columns(2)

    col1.metric("Training Records", len(X_train))
    col2.metric("Testing Records", len(X_test))

    # ============================================
    # Train Models
    # ============================================

    if st.button("Train & Compare Models"):

        results = []

        best_model = None
        best_auc = 0

        # ========================================
        # Logistic Regression
        # ========================================

        log_model = LogisticRegression(
            max_iter=1000
        )

        log_model.fit(
            X_train,
            y_train
        )

        pred = log_model.predict(X_test)
        prob = log_model.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, prob)

        results.append([
            "Logistic Regression",
            accuracy_score(y_test,pred),
            precision_score(y_test,pred),
            recall_score(y_test,pred),
            f1_score(y_test,pred),
            auc
        ])

        if auc > best_auc:
            best_auc = auc
            best_model = log_model

        # ========================================
        # KNN Hyperparameter Tuning
        # ========================================

        knn_grid = {
            "n_neighbors":[3,5,7,9],
            "weights":["uniform","distance"]
        }

        knn_search = GridSearchCV(
            KNeighborsClassifier(),
            knn_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1
        )

        knn_search.fit(
            X_train,
            y_train
        )

        knn = knn_search.best_estimator_

        pred = knn.predict(X_test)
        prob = knn.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, prob)

        results.append([
            "KNN",
            accuracy_score(y_test,pred),
            precision_score(y_test,pred),
            recall_score(y_test,pred),
            f1_score(y_test,pred),
            auc
        ])

        if auc > best_auc:
            best_auc = auc
            best_model = knn

        # ========================================
        # Random Forest Hyperparameter Tuning
        # ========================================

        rf_grid = {

            "n_estimators":[100,200],

            "max_depth":[5,10,None],

            "min_samples_split":[2,5]
        }

        rf_search = GridSearchCV(
            RandomForestClassifier(
                random_state=42
            ),
            rf_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1
        )

        rf_search.fit(
            X_train,
            y_train
        )

        rf = rf_search.best_estimator_

        pred = rf.predict(X_test)
        prob = rf.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, prob)

        results.append([
            "Random Forest",
            accuracy_score(y_test,pred),
            precision_score(y_test,pred),
            recall_score(y_test,pred),
            f1_score(y_test,pred),
            auc
        ])

        if auc > best_auc:
            best_auc = auc
            best_model = rf

        # ========================================
        # XGBoost Hyperparameter Tuning
        # ========================================

        xgb_grid = {

            "n_estimators":[100,200],

            "max_depth":[3,5,7],

            "learning_rate":[0.01,0.05,0.1]
        }

        xgb_search = GridSearchCV(
            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            ),
            xgb_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1
        )

        xgb_search.fit(
            X_train,
            y_train
        )

        xgb = xgb_search.best_estimator_

        pred = xgb.predict(X_test)
        prob = xgb.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, prob)

        results.append([
            "XGBoost",
            accuracy_score(y_test,pred),
            precision_score(y_test,pred),
            recall_score(y_test,pred),
            f1_score(y_test,pred),
            auc
        ])

        if auc > best_auc:
            best_auc = auc
            best_model = xgb

        # ========================================
        # Results
        # ========================================

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC AUC"
            ]
        )

        st.header("📊 Model Comparison")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # ========================================
        # ROC AUC Comparison
        # ========================================

        fig = px.bar(
            results_df,
            x="Model",
            y="ROC AUC",
            title="ROC-AUC Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ========================================
        # Best Model
        # ========================================

        best_model_name = (
            results_df
            .sort_values(
                by="ROC AUC",
                ascending=False
            )
            .iloc[0]["Model"]
        )

        st.success(
            f"Best Model Selected: {best_model_name}"
        )

        # ========================================
        # Save Model
        # ========================================

        joblib.dump(
            best_model,
            "best_model.pkl"
        )

        joblib.dump(
            scaler,
            "scaler.pkl"
        )

        st.success(
            "Model Saved Successfully"
        )

        # ========================================
        # Confusion Matrix
        # ========================================

        final_pred = best_model.predict(
            X_test
        )

        cm = confusion_matrix(
            y_test,
            final_pred
        )

        cm_df = pd.DataFrame(
            cm,
            columns=["Pred 0","Pred 1"],
            index=["Actual 0","Actual 1"]
        )

        st.subheader("Confusion Matrix")

        st.dataframe(cm_df)

        # ========================================
        # Feature Importance
        # ========================================

        if hasattr(
            best_model,
            "feature_importances_"
        ):

            importance = pd.DataFrame({

                "Feature":features,

                "Importance":
                    best_model.feature_importances_

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

else:

    st.info(
        "Upload the predictive maintenance dataset to start model training."
    )
