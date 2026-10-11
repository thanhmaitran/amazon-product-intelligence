"""
sentiment.py
------------
Phase 5 — Sentiment Analysis.

  1. Rule-based baseline: uses `vaderSentiment` if installed; otherwise falls
     back to a small built-in positive/negative lexicon scorer so the pipeline
     still produces positive/neutral/negative labels without the dependency.
     Run `pip install vaderSentiment` for the real VADER scores.
  2. Transformer sentiment: uses a HuggingFace `transformers` sentiment
     pipeline (e.g. DistilBERT) if `transformers`+`torch` are installed and
     model weights can be downloaded; otherwise this step is skipped with a
     clear log message rather than failing the whole run.
  3. Sentiment-by-category summary + chart.

Run directly:
    python src/sentiment.py
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.eda import load_features
from src.visualization import plot_bar, set_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = "review_content_clean"

# --------------------------------------------------------------------------- #
# Rule-based baseline: VADER if available, else a tiny built-in lexicon
# --------------------------------------------------------------------------- #

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
    logger.warning(
        "`vaderSentiment` not installed — using a small built-in lexicon fallback "
        "for the rule-based baseline. Run `pip install vaderSentiment` for real VADER scores."
    )

# Minimal fallback lexicon (used only if vaderSentiment is unavailable). Real
# VADER covers thousands of terms with intensity weights; this fallback is
# deliberately small and exists only so the pipeline still runs end-to-end.
_POSITIVE_WORDS = {
    "good", "great", "excellent", "awesome", "nice", "love", "best", "perfect",
    "amazing", "fast", "durable", "easy", "satisfied", "worth", "recommend",
    "quality", "value", "happy", "smooth", "comfortable", "sturdy", "reliable",
}
_NEGATIVE_WORDS = {
    "bad", "poor", "worst", "waste", "disappointed", "broken", "slow",
    "cheap", "defective", "issue", "problem", "return", "damaged", "fake",
    "stopped", "faulty", "useless", "horrible", "terrible", "not working",
}


def _fallback_score(text: str) -> float:
    """Return a compound-like score in [-1, 1] using simple word matching."""
    if not text:
        return 0.0
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def vader_score(text: str) -> float:
    """Compound sentiment score in [-1, 1]."""
    if pd.isna(text) or not str(text).strip():
        return 0.0
    if _VADER_AVAILABLE:
        return _analyzer.polarity_scores(str(text))["compound"]
    return _fallback_score(str(text))


def label_from_score(score: float, threshold: float = 0.05) -> str:
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"


def run_vader_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_score"] = df[TEXT_COL].apply(vader_score)
    df["sentiment_label"] = df["sentiment_score"].apply(label_from_score)
    return df


# --------------------------------------------------------------------------- #
# Transformer sentiment (optional — requires transformers + torch + network
# access to download model weights on first use)
# --------------------------------------------------------------------------- #

def run_transformer_sentiment(df: pd.DataFrame, text_col: str = TEXT_COL,
                               model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                               max_length: int = 256) -> pd.DataFrame | None:
    try:
        from transformers import pipeline
    except ImportError:
        logger.warning(
            "`transformers`/`torch` not installed — skipping transformer sentiment. "
            "Run `pip install transformers torch` to enable this comparison."
        )
        return None

    try:
        clf = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=max_length)
    except Exception as exc:  # e.g. no network to fetch weights
        logger.warning("Could not load transformer model '%s' (%s) — skipping.", model_name, exc)
        return None

    df = df.copy()
    texts = df[text_col].fillna("").tolist()
    predictions = clf(texts, batch_size=16)
    df["transformer_label"] = [p["label"].lower() for p in predictions]
    df["transformer_score"] = [p["score"] for p in predictions]
    return df


def compare_vader_vs_transformer(df: pd.DataFrame) -> dict | None:
    """Agreement rate between the rule-based baseline and the transformer model."""
    if "transformer_label" not in df.columns:
        return None
    # transformer labels are 'positive'/'negative' (binary); map VADER's 3-way
    # label down to binary for a fair comparison, dropping neutral cases.
    binary = df[df["sentiment_label"] != "neutral"].copy()
    binary["vader_binary"] = binary["sentiment_label"]
    agreement = (binary["vader_binary"] == binary["transformer_label"]).mean()
    return {"n_compared": len(binary), "agreement_rate": agreement}


# --------------------------------------------------------------------------- #
# Sentiment by category
# --------------------------------------------------------------------------- #

def sentiment_by_category(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    summary = (
        df.groupby("main_category")["sentiment_label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0)
        .sort_values("positive", ascending=False)
    )

    set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot(kind="bar", stacked=True, ax=ax,
                 color={"positive": "#2ca02c", "neutral": "#bbbbbb", "negative": "#d62728"})
    ax.set_title("Sentiment Mix by Category")
    ax.set_ylabel("Share of reviews")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    from src.visualization import save_fig
    save_fig(fig, "sentiment_by_category", save)

    return summary


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_sentiment(df: pd.DataFrame | None = None, save: bool = True,
                   include_transformer: bool = True) -> dict:
    if df is None:
        df = load_features()

    df = run_vader_baseline(df)

    if include_transformer:
        transformer_df = run_transformer_sentiment(df)
        if transformer_df is not None:
            df = transformer_df

    category_summary = sentiment_by_category(df, save=save)
    comparison = compare_vader_vs_transformer(df)

    overall = df["sentiment_label"].value_counts(normalize=True).round(3).to_dict()
    logger.info("Overall sentiment mix: %s", overall)

    return {
        "df": df,
        "overall": overall,
        "by_category": category_summary,
        "vader_vs_transformer": comparison,
    }


if __name__ == "__main__":
    results = run_sentiment()
    print("Overall sentiment mix:", results["overall"])
    print("\nBy category:\n", results["by_category"])
    if results["vader_vs_transformer"]:
        print("\nVADER vs Transformer agreement:", results["vader_vs_transformer"])
