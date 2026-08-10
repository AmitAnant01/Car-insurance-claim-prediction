import os
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.utils import get_logger, read_csv_safely

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "train.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion")
        df = read_csv_safely(self.config.raw_data_path)
        logger.info(f"Raw dataset shape: {df.shape}")

        os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df["is_claim"],
        )

        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)

        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return self.config.train_data_path, self.config.test_data_path


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
