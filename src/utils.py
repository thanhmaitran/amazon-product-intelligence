"""
utils.py
--------
Small, dependency-light helper functions shared across the src/ modules
(price/count parsing, text cleaning, path helpers). Kept separate from
preprocessing.py so they can be unit-tested and reused by feature_engineering.py,
eda.py, nlp.py, etc. without circular imports.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def project_root() -> Path:
    """Return the repository root (parent of src/), regardless of cwd."""
    return Path(__file__).resolve().parent.parent


def raw_data_path(filename: str = "amazon.csv") -> Path:
    return project_root() / "data" / "raw" / filename


def processed_data_path(filename: str) -> Path:
    return project_root() / "data" / "processed" / filename


# --------------------------------------------------------------------------- #
# Numeric parsing
# --------------------------------------------------------------------------- #

_CURRENCY_RE = re.compile(r"[^\d.]")


def parse_currency(value) -> float:
    """
    Convert an Indian-Rupee-formatted price string to a float.
    '₹1,099' -> 1099.0, '₹399' -> 399.0, NaN/empty -> NaN.
    """
    if pd.isna(value):
        return float("nan")
    cleaned = _CURRENCY_RE.sub("", str(value))
    return float(cleaned) if cleaned else float("nan")


def parse_count(value) -> float:
    """
    Convert a comma-formatted integer string to a float (float so NaN is
    representable; caller can cast to Int64 after dropping/filling NaNs).
    '24,269' -> 24269.0
    """
    if pd.isna(value):
        return float("nan")
    cleaned = re.sub(r"[^\d.]", "", str(value))
    return float(cleaned) if cleaned else float("nan")


def parse_percentage(value) -> float:
    """'64%' -> 0.64"""
    if pd.isna(value):
        return float("nan")
    cleaned = str(value).replace("%", "").strip()
    return float(cleaned) / 100.0 if cleaned else float("nan")


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
# Map punctuation to a SPACE (not removal) so that things like "X/XsMax/Xr" or
# "iPad Air 1/2,iPad mini" don't get glued into one token ("xxsmaxxr", "12ipad").
_PUNCT_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

# Minimal built-in stopword list used only as a fallback when NLTK's corpus
# is not available in the environment (e.g. no network to download it).
# preprocessing.py prefers NLTK's stopwords whenever present.
_FALLBACK_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "at", "by",
    "this", "that", "these", "those", "it", "its", "as", "from", "so", "than",
    "too", "very", "can", "will", "just", "not", "no", "do", "does", "did",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "my", "your", "our", "have", "has", "had", "there", "here", "which",
    "what", "who", "whom", "into", "about", "up", "down", "out", "over",
    "again", "then", "once", "s", "t",
}


def strip_html(text: str) -> str:
    return _HTML_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub(" ", text)


def strip_punctuation(text: str) -> str:
    return text.translate(_PUNCT_TABLE)


def basic_clean(text) -> str:
    """
    Lowercase + strip HTML + strip emoji + strip punctuation + collapse
    whitespace. Does NOT remove stopwords or lemmatize (see preprocessing.py's
    `clean_text_column`, which layers those on top using NLTK when available).
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = strip_html(text)
    text = strip_urls(text)
    text = strip_emoji(text)
    text = strip_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: set[str] | None = None) -> str:
    sw = stopwords if stopwords is not None else _FALLBACK_STOPWORDS
    return " ".join(w for w in text.split() if w not in sw)
