import os

import pandas as pd

from src.data_transformation import (
    _extract_torque_power,
    _encode_yes_no,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
)
from src.utils import get_logger, load_object

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model = load_object(os.path.join(artifacts_dir, "model.pkl"))
        self.encoder = load_object(os.path.join(artifacts_dir, "encoder.pkl"))
        self.scaler = load_object(os.path.join(artifacts_dir, "scaler.pkl"))
        self.numeric_columns = load_object(
            os.path.join(artifacts_dir, "numeric_columns.pkl")
        )
        self.feature_columns = load_object(
            os.path.join(artifacts_dir, "feature_columns.pkl")
        )
        logger.info("Prediction pipeline loaded and ready")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _extract_torque_power(df)
        df = _encode_yes_no(df)
        df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

        cat_encoded = self.encoder.transform(df[CATEGORICAL_COLUMNS])
        cat_names = self.encoder.get_feature_names_out(CATEGORICAL_COLUMNS)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_names, index=df.index)

        num_scaled = self.scaler.transform(df[self.numeric_columns])
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_columns, index=df.index)

        features = pd.concat([num_df, cat_df], axis=1)
        return features[self.feature_columns]

    def predict(self, raw_df: pd.DataFrame) -> dict:
        features = self._prepare_features(raw_df)
        proba = self.model.predict_proba(features)[:, 1]
        prediction = (proba >= 0.5).astype(int)

        return {
            "prediction": prediction.tolist(),
            "claim_probability": [round(float(p), 4) for p in proba],
        }
