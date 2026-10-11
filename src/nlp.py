"""
nlp.py
------
Phase 4 — NLP Review Analysis. Works on the `*_clean` text columns produced by
preprocessing.py (about_product_clean, review_title_clean, review_content_clean).

  - Text length analysis (already computed as *_length in feature_engineering.py;
    this module re-summarizes and visualizes it)
  - Word frequency: unigrams, bigrams, trigrams (sklearn CountVectorizer)
  - WordCloud (uses the `wordcloud` package if installed; otherwise falls back
    to a horizontal bar chart of top terms so the pipeline still produces a
    usable visual without the optional dependency)
  - TF-IDF top keywords per category (sklearn TfidfVectorizer)
  - Topic modelling: 5-8 topics via NMF on TF-IDF (no gensim dependency needed)

Run directly:
    python src/nlp.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF

from src.eda import load_features
from src.visualization import plot_bar, plot_hist, save_fig, set_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = "review_content_clean"

try:
    from wordcloud import WordCloud
    _WORDCLOUD_AVAILABLE = True
except ImportError:
    _WORDCLOUD_AVAILABLE = False
    logger.warning(
        "`wordcloud` package not installed — plot_wordcloud() will fall back to a "
        "top-terms bar chart. Run `pip install wordcloud` for the real word cloud image."
    )


# --------------------------------------------------------------------------- #
# Text length analysis
# --------------------------------------------------------------------------- #

def text_length_summary(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    cols = ["review_length", "title_length", "description_length"]
    summary = df[cols].describe().T
    for col in cols:
        plot_hist(df[col], f"{col.replace('_', ' ').title()} Distribution",
                  xlabel="Word count", name=f"{col}_distribution", save=save)
    return summary


# --------------------------------------------------------------------------- #
# N-gram frequency
# --------------------------------------------------------------------------- #

def top_ngrams(texts: pd.Series, ngram_range=(1, 1), top_n: int = 20) -> pd.DataFrame:
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(texts.fillna(""))
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    freq = pd.DataFrame({"term": vectorizer.get_feature_names_out(), "count": counts})
    return freq.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)


def plot_ngrams(freq: pd.DataFrame, title: str, name: str, save: bool = True):
    return plot_bar(freq["term"], freq["count"], title, xlabel="Frequency", name=name, save=save)


def ngram_analysis(df: pd.DataFrame, save: bool = True) -> dict:
    unigrams = top_ngrams(df[TEXT_COL], (1, 1))
    bigrams = top_ngrams(df[TEXT_COL], (2, 2))
    trigrams = top_ngrams(df[TEXT_COL], (3, 3))

    plot_ngrams(unigrams, "Top 20 Unigrams in Reviews", "top_unigrams", save)
    plot_ngrams(bigrams, "Top 20 Bigrams in Reviews", "top_bigrams", save)
    plot_ngrams(trigrams, "Top 20 Trigrams in Reviews", "top_trigrams", save)

    return {"unigrams": unigrams, "bigrams": bigrams, "trigrams": trigrams}


# --------------------------------------------------------------------------- #
# WordCloud (with graceful fallback)
# --------------------------------------------------------------------------- #

def plot_wordcloud(texts: pd.Series, title="Review WordCloud", name="review_wordcloud",
                    save: bool = True, max_words: int = 100):
    import matplotlib.pyplot as plt

    set_style()
    joined = " ".join(texts.fillna(""))

    if _WORDCLOUD_AVAILABLE:
        wc = WordCloud(width=1000, height=500, background_color="white",
                        max_words=max_words, colormap="viridis").generate(joined)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        save_fig(fig, name, save)
        return fig

    # Fallback: top-terms bar chart stands in for the word cloud
    logger.info("Using top-terms bar chart fallback for '%s' (wordcloud not installed)", title)
    freq = top_ngrams(texts, (1, 1), top_n=max_words // 4)
    return plot_ngrams(freq, f"{title} (fallback: top terms bar chart)", name, save)


# --------------------------------------------------------------------------- #
# TF-IDF keywords per category
# --------------------------------------------------------------------------- #

def tfidf_keywords_by_category(df: pd.DataFrame, top_n: int = 10) -> dict[str, list[str]]:
    results = {}
    for category, group in df.groupby("main_category"):
        docs = group[TEXT_COL].fillna("")
        if docs.str.strip().eq("").all() or len(docs) < 2:
            continue
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=1)
        try:
            matrix = vectorizer.fit_transform(docs)
        except ValueError:
            continue
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:top_n]
        terms = vectorizer.get_feature_names_out()
        results[category] = [terms[i] for i in top_idx]
    return results


# --------------------------------------------------------------------------- #
# Topic modelling (NMF on TF-IDF — avoids a gensim dependency)
# --------------------------------------------------------------------------- #

def topic_model(df: pd.DataFrame, n_topics: int = 6, n_top_words: int = 8, random_state: int = 42) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=3)
    matrix = vectorizer.fit_transform(df[TEXT_COL].fillna(""))
    terms = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=400)
    nmf.fit(matrix)

    topics = {}
    for idx, component in enumerate(nmf.components_):
        top_idx = component.argsort()[::-1][:n_top_words]
        topics[f"Topic {idx + 1}"] = [terms[i] for i in top_idx]

    logger.info("Fitted NMF topic model with %d topics", n_topics)
    return topics


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_nlp(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        df = load_features()

    results = {}
    results["length_summary"] = text_length_summary(df, save=save)
    results["ngrams"] = ngram_analysis(df, save=save)
    plot_wordcloud(df[TEXT_COL], save=save)
    results["tfidf_keywords"] = tfidf_keywords_by_category(df)
    results["topics"] = topic_model(df)

    logger.info("NLP analysis complete")
    return results


if __name__ == "__main__":
    results = run_nlp()
    print("\nTop unigrams:\n", results["ngrams"]["unigrams"].head(10))
    print("\nTF-IDF keywords by category:")
    for cat, kw in results["tfidf_keywords"].items():
        print(f"  {cat}: {', '.join(kw)}")
    print("\nTopics:")
    for topic, words in results["topics"].items():
        print(f"  {topic}: {', '.join(words)}")
