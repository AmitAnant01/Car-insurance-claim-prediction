import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

from src.utils import get_logger, save_object, save_json

logger = get_logger(__name__)


def get_candidate_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
        ),
    }


class ModelTrainer:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        candidates = get_candidate_models(scale_pos_weight)

        results = {}
        fitted_models = {}

        for name, model in candidates.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)[:, 1]
            val_pred = model.predict(X_val)

            auc = roc_auc_score(y_val, val_proba)
            f1 = f1_score(y_val, val_pred)

            results[name] = {"roc_auc": round(auc, 4), "f1_score": round(f1, 4)}
            fitted_models[name] = model
            logger.info(f"{name} -> ROC-AUC: {auc:.4f}, F1: {f1:.4f}")

        best_name = max(results, key=lambda k: results[k]["roc_auc"])
        best_model = fitted_models[best_name]
        logger.info(f"Best model: {best_name}")

        os.makedirs(self.artifacts_dir, exist_ok=True)
        save_object(os.path.join(self.artifacts_dir, "model.pkl"), best_model)
        save_json(os.path.join(self.artifacts_dir, "metrics.json"), results)

        return best_name, best_model, results
