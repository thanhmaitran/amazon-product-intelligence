"""
feature_engineering.py
-----------------------
Phase 1 (continued). Takes the cleaned DataFrame from preprocessing.py and
derives modeling/EDA-ready features:

  - discount_amount      = actual_price - discounted_price
  - review_length         = word count of review_content
  - title_length           = word count of review_title
  - description_length     = word count of about_product
  - popularity_score       = rating * log(1 + rating_count)
  - price_bucket           = Budget / Mid / Premium (tertiles of discounted_price)
  - discount_bucket        = Low / Medium / High (tertiles of discount_percentage)

Run directly:
    python src/feature_engineering.py
reads data/processed/cleaned_products.csv, writes data/processed/products_features.csv

Import and reuse:
    from src.feature_engineering import run_feature_engineering
    df = run_feature_engineering()
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.preprocessing import run_cleaning_pipeline
from src.utils import processed_data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def add_discount_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["discount_amount"] = df["actual_price"] - df["discounted_price"]
    return df


def _word_count(text) -> int:
    if pd.isna(text):
        return 0
    return len(str(text).split())


def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_length"] = df["review_content"].apply(_word_count)
    df["title_length"] = df["review_title"].apply(_word_count)
    df["description_length"] = df["about_product"].apply(_word_count)
    return df


def add_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["popularity_score"] = df["rating"].fillna(0) * np.log1p(df["rating_count"].fillna(0))
    return df


def _tertile_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    """Bucket a numeric series into 3 equal-frequency bins with given labels."""
    try:
        return pd.qcut(series, q=3, labels=labels, duplicates="drop")
    except ValueError:
        # Fall back to equal-width bins if there isn't enough distinct data for tertiles
        return pd.cut(series, bins=3, labels=labels)


def add_price_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_bucket"] = _tertile_bucket(df["discounted_price"], ["Budget", "Mid", "Premium"])
    return df


def add_discount_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["discount_bucket"] = _tertile_bucket(df["discount_percentage"], ["Low", "Medium", "High"])
    return df


def run_feature_engineering(cleaned_df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    if cleaned_df is None:
        cleaned_path = processed_data_path("cleaned_products.csv")
        if cleaned_path.exists():
            logger.info("Loading cleaned data from %s", cleaned_path)
            cleaned_df = pd.read_csv(cleaned_path)
        else:
            logger.info("No cleaned data found on disk — running preprocessing pipeline first")
            cleaned_df = run_cleaning_pipeline(save=True)

    df = cleaned_df.copy()
    df = add_discount_amount(df)
    df = add_text_length_features(df)
    df = add_popularity_score(df)
    df = add_price_bucket(df)
    df = add_discount_bucket(df)

    if save:
        out_path = processed_data_path("products_features.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved feature-engineered data to %s (%d rows, %d columns)", out_path, *df.shape)

    return df


if __name__ == "__main__":
    run_feature_engineering()
