"""
preprocessing.py
-----------------
Phase 1 of the pipeline. Loads the raw Amazon dataset and produces a cleaned,
analysis-ready DataFrame:

  1. Load raw CSV
  2. Convert price columns (₹1,099 -> 1099.0)
  3. Convert rating_count (24,269 -> 24269)
  4. Convert discount_percentage (64% -> 0.64)
  5. Split the pipe-delimited `category` column into main_category /
     sub_category / product_type
  6. Clean free-text columns (about_product, review_title, review_content):
     lowercase, strip HTML/punctuation/emoji, remove stopwords, lemmatize

Run directly:
    python src/preprocessing.py
writes data/processed/cleaned_products.csv

Import and reuse:
    from src.preprocessing import run_cleaning_pipeline
    df = run_cleaning_pipeline()
"""

from __future__ import annotations

import logging

import pandas as pd

from src.utils import (
    parse_currency,
    parse_count,
    parse_percentage,
    basic_clean,
    remove_stopwords,
    raw_data_path,
    processed_data_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COLUMNS = ["about_product", "review_title", "review_content"]
CATEGORY_LEVELS = ["main_category", "sub_category", "product_type"]

# --------------------------------------------------------------------------- #
# NLTK setup (optional — falls back gracefully if unavailable/offline)
# --------------------------------------------------------------------------- #

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        _STOPWORDS = set(nltk_stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS = set(nltk_stopwords.words("english"))

    try:
        _LEMMATIZER = WordNetLemmatizer()
        _LEMMATIZER.lemmatize("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        _LEMMATIZER = WordNetLemmatizer()

    _NLTK_READY = True
except Exception as exc:  # pragma: no cover - environment dependent
    logger.warning(
        "NLTK stopwords/lemmatizer unavailable (%s). "
        "Falling back to a built-in stopword list and skipping lemmatization. "
        "Run `pip install nltk` and download 'stopwords'/'wordnet' for full text cleaning.",
        exc,
    )
    _STOPWORDS = None  # utils.remove_stopwords will use its fallback list
    _LEMMATIZER = None
    _NLTK_READY = False


def lemmatize(text: str) -> str:
    if _LEMMATIZER is None:
        return text
    return " ".join(_LEMMATIZER.lemmatize(w) for w in text.split())


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_raw_data(path=None) -> pd.DataFrame:
    path = path or raw_data_path("amazon.csv")
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# Cleaning steps
# --------------------------------------------------------------------------- #

def clean_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["discounted_price"] = df["discounted_price"].apply(parse_currency)
    df["actual_price"] = df["actual_price"].apply(parse_currency)
    return df


def clean_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = df["rating_count"].apply(parse_count)
    return df


def clean_discount_percentage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["discount_percentage"] = df["discount_percentage"].apply(parse_percentage)
    return df


def split_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables'
    -> main_category='Computers & Accessories', sub_category='Accessories & Peripherals',
       product_type='USB Cables' (last segment of the hierarchy)
    """
    df = df.copy()

    def _pretty(segment: str) -> str:
        import re
        segment = segment.replace("&", " & ")
        # Insert a space at lower/digit -> upper transitions (fastCharging -> fast Charging)
        segment = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", segment)
        # Insert a space inside acronym runs followed by a new word (USBCables -> USB Cables)
        segment = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", segment)
        return re.sub(r"\s+", " ", segment).strip()

    levels = df["category"].fillna("").str.split("|")
    df["main_category"] = levels.apply(lambda x: _pretty(x[0]) if len(x) > 0 and x[0] else pd.NA)
    df["sub_category"] = levels.apply(lambda x: _pretty(x[1]) if len(x) > 1 and x[1] else pd.NA)
    df["product_type"] = levels.apply(lambda x: _pretty(x[-1]) if len(x) > 0 and x[-1] else pd.NA)
    return df


def clean_text_column(series: pd.Series) -> pd.Series:
    cleaned = series.apply(basic_clean)
    cleaned = cleaned.apply(lambda t: remove_stopwords(t, _STOPWORDS))
    if _NLTK_READY:
        cleaned = cleaned.apply(lemmatize)
    return cleaned


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in TEXT_COLUMNS:
        df[f"{col}_clean"] = clean_text_column(df[col])
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate product_id rows", dropped)
    return df


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_cleaning_pipeline(raw_path=None, save: bool = True) -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = drop_exact_duplicates(df)
    df = clean_price_columns(df)
    df = clean_rating_columns(df)
    df = clean_discount_percentage(df)
    df = split_category(df)
    df = clean_text_columns(df)

    n_missing_rating_count = df["rating_count"].isna().sum()
    if n_missing_rating_count:
        logger.info("%d rows have missing rating_count after parsing", n_missing_rating_count)

    if save:
        out_path = processed_data_path("cleaned_products.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved cleaned data to %s (%d rows, %d columns)", out_path, *df.shape)

    return df


if __name__ == "__main__":
    run_cleaning_pipeline()
