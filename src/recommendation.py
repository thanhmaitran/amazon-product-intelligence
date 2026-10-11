"""
recommendation.py
------------------
Phase 7 — Recommendation System.

  1. Content-based recommender using TF-IDF over product text
     (product_name + about_product_clean) + cosine similarity. Fully
     functional with only scikit-learn (no extra downloads required).
  2. Embedding-based recommender using `sentence-transformers`, used if the
     package (and model weights) are available; otherwise this path is
     skipped with a clear message — the TF-IDF recommender remains the
     working default.
  3. A simple offline evaluation: category-precision@k, i.e. what fraction of
     a product's top-k recommendations share its main_category (a reasonable
     proxy for topical relevance without needing labeled relevance judgments).

Run directly:
    python src/recommendation.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.eda import load_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _combined_text(df: pd.DataFrame) -> pd.Series:
    name = df["product_name"].fillna("")
    about = df.get("about_product_clean", pd.Series("", index=df.index)).fillna("")
    return (name.str.lower() + " " + about).str.strip()


# --------------------------------------------------------------------------- #
# TF-IDF content-based recommender
# --------------------------------------------------------------------------- #

class TfidfRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=1)
        self._matrix = self._vectorizer.fit_transform(_combined_text(self.df))
        self._sim = cosine_similarity(self._matrix)
        self._id_to_idx = {pid: i for i, pid in enumerate(self.df["product_id"])}

    def _resolve_index(self, product_id: str | None = None, product_name: str | None = None) -> int:
        if product_id is not None:
            if product_id not in self._id_to_idx:
                raise KeyError(f"Unknown product_id: {product_id}")
            return self._id_to_idx[product_id]
        if product_name is not None:
            matches = self.df.index[
                self.df["product_name"].str.contains(product_name, case=False, na=False, regex=False)
            ]
            if len(matches) == 0:
                raise KeyError(f"No product found matching name: {product_name!r}")
            return matches[0]
        raise ValueError("Provide either product_id or product_name")

    def recommend(self, product_id: str | None = None, product_name: str | None = None,
                   top_k: int = 5) -> pd.DataFrame:
        idx = self._resolve_index(product_id, product_name)
        scores = self._sim[idx]
        order = np.argsort(scores)[::-1]
        order = [i for i in order if i != idx][:top_k]

        result = self.df.loc[order, ["product_id", "product_name", "main_category", "rating", "discounted_price"]].copy()
        result["similarity_score"] = scores[order]
        return result.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Sentence-Transformer embedding recommender (optional)
# --------------------------------------------------------------------------- #

class EmbeddingRecommender:
    """Same interface as TfidfRecommender, backed by sentence-transformers
    embeddings instead of TF-IDF. Returns None from `build()` if the package
    or model weights aren't available (e.g. no network in this environment)."""

    def __init__(self, df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # raises ImportError if missing
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        self._embeddings = self.model.encode(_combined_text(self.df).tolist(), show_progress_bar=False)
        self._sim = cosine_similarity(self._embeddings)
        self._id_to_idx = {pid: i for i, pid in enumerate(self.df["product_id"])}

    def recommend(self, product_id: str, top_k: int = 5) -> pd.DataFrame:
        idx = self._id_to_idx[product_id]
        scores = self._sim[idx]
        order = np.argsort(scores)[::-1]
        order = [i for i in order if i != idx][:top_k]
        result = self.df.loc[order, ["product_id", "product_name", "main_category", "rating", "discounted_price"]].copy()
        result["similarity_score"] = scores[order]
        return result.reset_index(drop=True)


def build_embedding_recommender(df: pd.DataFrame):
    try:
        return EmbeddingRecommender(df)
    except ImportError:
        logger.warning(
            "`sentence-transformers` not installed — skipping embedding-based recommender. "
            "Run `pip install sentence-transformers` (requires network for model weights) to enable it. "
            "The TF-IDF recommender above remains fully functional."
        )
        return None
    except Exception as exc:  # e.g. no network to download model weights
        logger.warning("Could not load sentence-transformers model (%s) — skipping.", exc)
        return None


# --------------------------------------------------------------------------- #
# Offline evaluation: category-precision@k
# --------------------------------------------------------------------------- #

def category_precision_at_k(recommender: TfidfRecommender, k: int = 5, n_samples: int = 200,
                             random_state: int = 42) -> float:
    df = recommender.df
    sample_ids = df["product_id"].sample(min(n_samples, len(df)), random_state=random_state)

    hits, total = 0, 0
    for pid in sample_ids:
        true_category = df.loc[df["product_id"] == pid, "main_category"].iloc[0]
        recs = recommender.recommend(product_id=pid, top_k=k)
        hits += (recs["main_category"] == true_category).sum()
        total += len(recs)

    precision = hits / total if total else 0.0
    logger.info("Category-precision@%d over %d sampled products: %.3f", k, len(sample_ids), precision)
    return precision


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_recommendation_demo(df: pd.DataFrame | None = None, example_query: str | None = None) -> dict:
    if df is None:
        df = load_features()

    tfidf_rec = TfidfRecommender(df)
    query = example_query or " ".join(df["product_name"].iloc[0].split()[:4])
    recs = tfidf_rec.recommend(product_name=query, top_k=5)

    precision = category_precision_at_k(tfidf_rec, k=5)

    embed_rec = build_embedding_recommender(df)

    return {
        "recommender": tfidf_rec,
        "embedding_recommender": embed_rec,
        "example_query": query,
        "example_recommendations": recs,
        "category_precision_at_5": precision,
    }


if __name__ == "__main__":
    results = run_recommendation_demo()
    print(f"Query product: {results['example_query']}\n")
    print(results["example_recommendations"].to_string(index=False))
    print(f"\nCategory-precision@5 (TF-IDF recommender): {results['category_precision_at_5']:.3f}")
