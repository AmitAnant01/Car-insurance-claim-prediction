import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import get_logger, save_object

logger = get_logger(__name__)

YES_NO_COLUMNS = [
    "is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
    "is_power_door_locks", "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert",
]

CATEGORICAL_COLUMNS = [
    "area_cluster", "segment", "model", "fuel_type", "engine_type",
    "rear_brakes_type", "transmission_type", "steering_type",
]

DROP_COLUMNS = ["policy_id", "max_torque", "max_power"]


def _extract_torque_power(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    torque_split = df["max_torque"].str.extract(r"([\d.]+)Nm@([\d.]+)rpm")
    df["torque_nm"] = torque_split[0].astype(float)
    df["torque_rpm"] = torque_split[1].astype(float)

    power_split = df["max_power"].str.extract(r"([\d.]+)bhp@([\d.]+)rpm")
    df["power_bhp"] = power_split[0].astype(float)
    df["power_rpm"] = power_split[1].astype(float)

    return df


def _encode_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in YES_NO_COLUMNS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


class DataTransformation:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.numeric_columns = None
        self.fitted = False

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _extract_torque_power(df)
        df = _encode_yes_no(df)
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = "is_claim"):
        logger.info("Fitting data transformation pipeline")
        df = self._base_transform(df)

        y = df[target_col]
        X = df.drop(columns=[target_col])

        cat_encoded = self.encoder.fit_transform(X[CATEGORICAL_COLUMNS])
        cat_feature_names = self.encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X.index)

        self.numeric_columns = [
            c for c in X.columns if c not in CATEGORICAL_COLUMNS
        ]
        num_scaled = self.scaler.fit_transform(X[self.numeric_columns])
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_columns, index=X.index)

        X_final = pd.concat([num_df, cat_df], axis=1)
        self.fitted = True

        os.makedirs(self.artifacts_dir, exist_ok=True)
        save_object(os.path.join(self.artifacts_dir, "encoder.pkl"), self.encoder)
        save_object(os.path.join(self.artifacts_dir, "scaler.pkl"), self.scaler)
        save_object(
            os.path.join(self.artifacts_dir, "numeric_columns.pkl"),
            self.numeric_columns,
        )
        save_object(
            os.path.join(self.artifacts_dir, "feature_columns.pkl"),
            list(X_final.columns),
        )
        logger.info(f"Final feature matrix shape: {X_final.shape}")
        return X_final, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit_transform before transform, or load artifacts")

        df = self._base_transform(df)
        cat_encoded = self.encoder.transform(df[CATEGORICAL_COLUMNS])
        cat_feature_names = self.encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)

        num_scaled = self.scaler.transform(df[self.numeric_columns])
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_columns, index=df.index)

        return pd.concat([num_df, cat_df], axis=1)
