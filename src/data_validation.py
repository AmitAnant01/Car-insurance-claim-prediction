"""
Data validation module.
Checks that an incoming dataframe matches what the model expects
before it goes anywhere near feature engineering or prediction.
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "policy_id", "policy_tenure", "age_of_car", "age_of_policyholder",
    "area_cluster", "population_density", "make", "segment", "model",
    "fuel_type", "max_torque", "max_power", "engine_type", "airbags",
    "is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "rear_brakes_type", "displacement", "cylinder",
    "transmission_type", "gear_box", "steering_type", "turning_radius",
    "length", "width", "height", "gross_weight", "is_front_fog_lights",
    "is_rear_window_wiper", "is_rear_window_washer",
    "is_rear_window_defogger", "is_brake_assist", "is_power_door_locks",
    "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert", "ncap_rating",
]

YES_NO_COLUMNS = [
    "is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors",
    "is_parking_camera", "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
    "is_power_door_locks", "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert",
]


class DataValidation:
    """Small rule based validator, no external schema library needed."""

    def __init__(self, expected_columns=None):
        self.expected_columns = expected_columns or EXPECTED_COLUMNS

    def validate_columns(self, df: pd.DataFrame) -> bool:
        missing = set(self.expected_columns) - set(df.columns)
        if missing:
            logger.error(f"Missing columns in input data: {missing}")
            return False
        return True

    def validate_no_nulls(self, df: pd.DataFrame) -> bool:
        null_counts = df[self.expected_columns].isnull().sum()
        bad_cols = null_counts[null_counts > 0]
        if len(bad_cols) > 0:
            logger.warning(f"Null values found in columns: {bad_cols.to_dict()}")
            return False
        return True

    def validate_yes_no_columns(self, df: pd.DataFrame) -> bool:
        for col in YES_NO_COLUMNS:
            if col not in df.columns:
                continue
            bad_values = set(df[col].dropna().unique()) - {"Yes", "No"}
            if bad_values:
                logger.error(f"Unexpected values in {col}: {bad_values}")
                return False
        return True

    def run_all_checks(self, df: pd.DataFrame) -> bool:
        checks = [
            self.validate_columns(df),
            self.validate_no_nulls(df),
            self.validate_yes_no_columns(df),
        ]
        passed = all(checks)
        logger.info(f"Validation passed: {passed}")
        return passed


if __name__ == "__main__":
    from src.utils import read_csv_safely

    sample = read_csv_safely("data/train.csv")
    validator = DataValidation()
    validator.run_all_checks(sample)
