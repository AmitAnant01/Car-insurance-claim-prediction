"""
Common helper functions used across the project.
Keeping these in one place so notebooks and src modules do not repeat code.
"""

import os
import json
import pickle
import logging

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Return a basic console logger. Used instead of print() in src modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_object(file_path: str, obj) -> None:
    """Pickle any python object (model, encoder, scaler) to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(file_path: str):
    """Load a pickled object from disk."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_json(file_path: str, data: dict) -> None:
    """Save a dict as a formatted json file, mainly used for metrics."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def read_csv_safely(file_path: str) -> pd.DataFrame:
    """Wrapper around pd.read_csv with a clearer error if file is missing."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find data file at: {file_path}")
    return pd.read_csv(file_path)
