import pandas as pd
import numpy as np


class FeatureEngineering:

    def __init__(self, df):
        self.df = df.copy()

    # ==================================================
    # TIME FEATURES
    # ==================================================

    def create_time_features(self):

        if "date" in self.df.columns:

            self.df["year"] = self.df["date"].dt.year

            self.df["month"] = self.df["date"].dt.month

            self.df["day"] = self.df["date"].dt.day

            self.df["weekday"] = self.df["date"].dt.weekday

            self.df["quarter"] = self.df["date"].dt.quarter

            self.df["week"] = (
                self.df["date"]
                .dt.isocalendar()
                .week
                .astype(int)
            )

        return self.df

    # ==================================================
    # SENSOR FEATURE LIST
    # ==================================================

    def get_sensor_columns(self):

        sensors = [

            col

            for col in self.df.columns

            if col.startswith("metric")

        ]

        return sensors

    # ==================================================
    # ROLLING MEAN
    # ==================================================

    def rolling_mean(self, window=10):

        sensors = self.get_sensor_columns()

        if "device" not in self.df.columns:
            return self.df

        for sensor in sensors:

            self.df[f"{sensor}_roll_mean"] = (

                self.df.groupby("device")[sensor]

                .transform(
                    lambda x:
                    x.rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                )

            )

        return self.df

    # ==================================================
    # ROLLING STD
    # ==================================================

    def rolling_std(self, window=10):

        sensors = self.get_sensor_columns()

        if "device" not in self.df.columns:
            return self.df

        for sensor in sensors:

            self.df[f"{sensor}_roll_std"] = (

                self.df.groupby("device")[sensor]

                .transform(
                    lambda x:
                    x.rolling(
                        window=window,
                        min_periods=1
                    ).std()
                )

            )

        return self.df

    # ==================================================
    # ROLLING MAX
    # ==================================================

    def rolling_max(self, window=10):

        sensors = self.get_sensor_columns()

        if "device" not in self.df.columns:
            return self.df

        for sensor in sensors:

            self.df[f"{sensor}_roll_max"] = (

                self.df.groupby("device")[sensor]

                .transform(
                    lambda x:
                    x.rolling(
                        window=window,
                        min_periods=1
                    ).max()
                )

            )

        return self.df

    # ==================================================
    # ROLLING MIN
    # ==================================================

    def rolling_min(self, window=10):

        sensors = self.get_sensor_columns()

        if "device" not in self.df.columns:
            return self.df

        for sensor in sensors:

            self.df[f"{sensor}_roll_min"] = (

                self.df.groupby("device")[sensor]

                .transform(
                    lambda x:
                    x.rolling(
                        window=window,
                        min_periods=1
                    ).min()
                )

            )

        return self.df

    # ==================================================
    # LAG FEATURES
    # ==================================================

    def create_lag_features(self):

        sensors = self.get_sensor_columns()

        if "device" not in self.df.columns:
            return self.df

        for sensor in sensors:

            self.df[f"{sensor}_lag1"] = (

                self.df.groupby("device")[sensor]
                .shift(1)

            )

            self.df[f"{sensor}_lag3"] = (

                self.df.groupby("device")[sensor]
                .shift(3)

            )

        return self.df

    # ==================================================
    # SENSOR CHANGE RATE
    # ==================================================

    def create_change_rate_features(self):

        sensors = self.get_sensor_columns()

        for sensor in sensors:

            self.df[f"{sensor}_change"] = (

                self.df[sensor].diff()

            )

        return self.df

    # ==================================================
    # Z SCORE FEATURES
    # ==================================================

    def create_zscore_features(self):

        sensors = self.get_sensor_columns()

        for sensor in sensors:

            mean = self.df[sensor].mean()

            std = self.df[sensor].std()

            if std != 0:

                self.df[f"{sensor}_zscore"] = (

                    self.df[sensor] - mean

                ) / std

        return self.df

    # ==================================================
    # SENSOR INTERACTION FEATURES
    # ==================================================

    def create_interaction_features(self):

        sensors = self.get_sensor_columns()

        if len(sensors) >= 2:

            self.df["metric1_metric2_ratio"] = (

                self.df[sensors[0]]

                /

                (self.df[sensors[1]] + 1e-6)

            )

        if len(sensors) >= 3:

            self.df["metric2_metric3_product"] = (

                self.df[sensors[1]]

                *

                self.df[sensors[2]]

            )

        return self.df

    # ==================================================
    # DEVICE FAILURE RATE
    # ==================================================

    def create_failure_rate_feature(self):

        if (

            "device" in self.df.columns

            and

            "failure" in self.df.columns

        ):

            rate = (

                self.df.groupby("device")
                ["failure"]

                .mean()

                .reset_index()

            )

            rate.columns = [

                "device",

                "device_failure_rate"

            ]

            self.df = self.df.merge(
                rate,
                on="device",
                how="left"
            )

        return self.df

    # ==================================================
    # ANOMALY COUNT FEATURE
    # ==================================================

    def create_anomaly_count(self):

        sensors = self.get_sensor_columns()

        anomaly_count = np.zeros(
            len(self.df)
        )

        for sensor in sensors:

            z = (

                self.df[sensor]

                -

                self.df[sensor].mean()

            ) / self.df[sensor].std()

            anomaly_count += (
                np.abs(z) > 3
            ).astype(int)

        self.df[
            "anomaly_count"
        ] = anomaly_count

        return self.df

    # ==================================================
    # RUN ALL FEATURES
    # ==================================================

    def build_features(self):

        self.create_time_features()

        self.rolling_mean()

        self.rolling_std()

        self.rolling_max()

        self.rolling_min()

        self.create_lag_features()

        self.create_change_rate_features()

        self.create_zscore_features()

        self.create_interaction_features()

        self.create_failure_rate_feature()

        self.create_anomaly_count()

        self.df = self.df.fillna(0)

        return self.df
