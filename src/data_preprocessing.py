import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()

    # ==================================================
    # LOAD DATA
    # ==================================================

    def load_data(self):

        self.df = pd.read_csv(self.filepath)

        return self.df

    # ==================================================
    # BASIC CLEANING
    # ==================================================

    def clean_data(self):

        # Remove duplicate records
        self.df = self.df.drop_duplicates()

        # Convert date column
        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(
                self.df["date"],
                errors="coerce"
            )

        return self.df

    # ==================================================
    # HANDLE MISSING VALUES
    # ==================================================

    def handle_missing_values(self):

        numeric_cols = self.df.select_dtypes(
            include=np.number
        ).columns

        self.df[numeric_cols] = (
            self.df[numeric_cols]
            .fillna(
                self.df[numeric_cols].mean()
            )
        )

        categorical_cols = self.df.select_dtypes(
            exclude=np.number
        ).columns

        for col in categorical_cols:

            self.df[col] = (
                self.df[col]
                .fillna(
                    self.df[col].mode()[0]
                )
            )

        return self.df

    # ==================================================
    # FEATURE ENGINEERING
    # ==================================================

    def create_time_features(self):

        if "date" in self.df.columns:

            self.df["year"] = (
                self.df["date"].dt.year
            )

            self.df["month"] = (
                self.df["date"].dt.month
            )

            self.df["day"] = (
                self.df["date"].dt.day
            )

            self.df["weekday"] = (
                self.df["date"].dt.weekday
            )

            self.df["week"] = (
                self.df["date"]
                .dt.isocalendar()
                .week
            )

        return self.df

    # ==================================================
    # ROLLING STATISTICS
    # ==================================================

    def create_rolling_features(
        self,
        window=10
    ):

        if "device" not in self.df.columns:
            return self.df

        sensor_cols = [

            col for col in self.df.columns

            if col.startswith("metric")

        ]

        for sensor in sensor_cols:

            self.df[
                f"{sensor}_mean"
            ] = (

                self.df.groupby("device")[sensor]

                .transform(
                    lambda x:
                    x.rolling(
                        window,
                        min_periods=1
                    ).mean()
                )

            )

            self.df[
                f"{sensor}_std"
            ] = (

                self.df.groupby("device")[sensor]

                .transform(
                    lambda x:
                    x.rolling(
                        window,
                        min_periods=1
                    ).std()
                )

            )

        return self.df

    # ==================================================
    # Z SCORE ANOMALY FEATURES
    # ==================================================

    def create_zscore_features(self):

        sensor_cols = [

            col for col in self.df.columns

            if col.startswith("metric")

        ]

        for sensor in sensor_cols:

            mean = self.df[sensor].mean()

            std = self.df[sensor].std()

            self.df[
                f"{sensor}_zscore"
            ] = (

                self.df[sensor] - mean

            ) / std

        return self.df

    # ==================================================
    # FAILURE RATE FEATURE
    # ==================================================

    def create_device_failure_rate(self):

        if (
            "device" in self.df.columns
            and "failure" in self.df.columns
        ):

            failure_rate = (

                self.df.groupby("device")
                ["failure"]

                .mean()

                .reset_index()

            )

            failure_rate.columns = [

                "device",

                "device_failure_rate"

            ]

            self.df = self.df.merge(
                failure_rate,
                on="device",
                how="left"
            )

        return self.df

    # ==================================================
    # PREPARE FEATURES
    # ==================================================

    def prepare_features(self):

        exclude_cols = [

            "failure",
            "date",
            "device"

        ]

        feature_cols = [

            col

            for col in self.df.columns

            if col not in exclude_cols

        ]

        X = self.df[feature_cols]

        y = self.df["failure"]

        return X, y, feature_cols

    # ==================================================
    # SCALE FEATURES
    # ==================================================

    def scale_features(
        self,
        X
    ):

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled

    # ==================================================
    # TRAIN TEST SPLIT
    # ==================================================

    def split_data(
        self,
        X,
        y,
        test_size=0.20
    ):

        return train_test_split(

            X,
            y,

            test_size=test_size,

            random_state=42,

            stratify=y

        )

    # ==================================================
    # COMPLETE PIPELINE
    # ==================================================

    def preprocess_pipeline(self):

        self.load_data()

        self.clean_data()

        self.handle_missing_values()

        self.create_time_features()

        self.create_rolling_features()

        self.create_zscore_features()

        self.create_device_failure_rate()

        X, y, feature_cols = self.prepare_features()

        X_scaled = self.scale_features(X)

        return (
            self.df,
            X_scaled,
            y,
            feature_cols,
            self.scaler
        )
