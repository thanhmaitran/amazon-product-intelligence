"""
streamlit_app.py
-----------------
Phase 8 — Streamlit Dashboard. Three pages:

  1. Business Dashboard — KPIs, category/pricing/rating/discount charts, correlation heatmap.
  2. Review Explorer — filter products by category/price/rating, search text, inspect
     reviews + sentiment for a selected product.
  3. Recommendation — pick a product, get TF-IDF content-based recommendations.

Run:
    streamlit run dashboard/streamlit_app.py

Design note: page-layout code (the `st.*` calls) is kept separate from the
underlying data logic (`filter_products`, `compute_kpis`, etc.), which are
plain functions taking/returning DataFrames. This keeps the logic unit-testable
without a running Streamlit server — see the `if __name__ == "__main__":` block
at the bottom, which exercises these functions with `python dashboard/streamlit_app.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.eda import load_features
from src.sentiment import run_vader_baseline
from src.recommendation import TfidfRecommender

# --------------------------------------------------------------------------- #
# Data logic (plain functions — no Streamlit dependency, unit-testable)
# --------------------------------------------------------------------------- #

def load_data() -> pd.DataFrame:
    df = load_features()
    df = run_vader_baseline(df)
    return df


def filter_products(
    df: pd.DataFrame,
    categories: list[str] | None = None,
    price_range: tuple[float, float] | None = None,
    rating_range: tuple[float, float] | None = None,
    search_term: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if categories:
        out = out[out["main_category"].isin(categories)]
    if price_range:
        lo, hi = price_range
        out = out[(out["discounted_price"] >= lo) & (out["discounted_price"] <= hi)]
    if rating_range:
        lo, hi = rating_range
        out = out[(out["rating"] >= lo) & (out["rating"] <= hi)]
    if search_term:
        term = search_term.lower()
        mask = (
            out["product_name"].str.lower().str.contains(term, na=False)
            | out["review_content"].str.lower().str.contains(term, na=False)
        )
        out = out[mask]
    return out


def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"Products": 0, "Categories": 0, "Avg Rating": "-", "Avg Discount": "-",
                "Avg Price (₹)": "-", "Total Reviews": 0}
    return {
        "Products": f"{df['product_id'].nunique():,}",
        "Categories": f"{df['main_category'].nunique()}",
        "Avg Rating": f"{df['rating'].mean():.2f}",
        "Avg Discount": f"{df['discount_percentage'].mean() * 100:.0f}%",
        "Avg Price (₹)": f"{df['discounted_price'].mean():,.0f}",
        "Total Reviews": f"{df['rating_count'].sum():,.0f}",
    }


def sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    if df.empty or "sentiment_label" not in df.columns:
        return pd.Series(dtype=float)
    return df["sentiment_label"].value_counts(normalize=True)


def get_category_options(df: pd.DataFrame) -> list[str]:
    return sorted(df["main_category"].dropna().unique().tolist())


def get_product_options(df: pd.DataFrame) -> list[str]:
    return df["product_name"].tolist()


# --------------------------------------------------------------------------- #
# Streamlit UI (only imported/executed when actually run via `streamlit run`)
# --------------------------------------------------------------------------- #

def _run_app():
    import matplotlib.pyplot as plt
    import streamlit as st

    from src.visualization import (
        plot_bar, plot_hist, plot_box, plot_scatter, plot_heatmap, plot_treemap,
    )

    st.set_page_config(page_title="Amazon Product Intelligence", layout="wide")

    @st.cache_data
    def _cached_load():
        return load_data()

    @st.cache_resource
    def _cached_recommender(_df):
        return TfidfRecommender(_df)

    df = _cached_load()

    st.sidebar.title("Amazon Product Intelligence")
    page = st.sidebar.radio("Page", ["📊 Business Dashboard", "🔍 Review Explorer", "🎯 Recommendation"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    categories = st.sidebar.multiselect("Category", get_category_options(df))
    price_min, price_max = float(df["discounted_price"].min()), float(df["discounted_price"].max())
    price_range = st.sidebar.slider("Price range (₹)", price_min, price_max, (price_min, price_max))
    rating_range = st.sidebar.slider("Rating range", 1.0, 5.0, (1.0, 5.0))

    filtered = filter_products(df, categories=categories or None, price_range=price_range,
                                rating_range=rating_range)

    # ----------------------------------------------------------------- #
    # Page 1 — Business Dashboard
    # ----------------------------------------------------------------- #
    if page == "📊 Business Dashboard":
        st.title("📊 Business Dashboard")
        st.caption(f"{len(filtered):,} products match the current filters")

        kpis = compute_kpis(filtered)
        cols = st.columns(len(kpis))
        for col, (label, value) in zip(cols, kpis.items()):
            col.metric(label, value)

        if filtered.empty:
            st.warning("No products match the current filters.")
            return

        c1, c2 = st.columns(2)
        with c1:
            counts = filtered["main_category"].value_counts()
            st.pyplot(plot_bar(counts.index, counts.values, "Products by Category", save=False))
        with c2:
            st.pyplot(plot_treemap(counts.index.tolist(), counts.values.tolist(),
                                    "Category Share", save=False))

        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(plot_hist(filtered["discounted_price"], "Price Distribution",
                                 xlabel="Price (₹)", save=False))
        with c4:
            st.pyplot(plot_hist(filtered["rating"], "Rating Distribution",
                                 xlabel="Rating", bins=20, save=False))

        c5, c6 = st.columns(2)
        with c5:
            st.pyplot(plot_scatter(filtered, x="discount_percentage", y="rating",
                                    title="Discount % vs. Rating", save=False))
        with c6:
            corr_cols = ["rating", "rating_count", "discounted_price", "discount_percentage",
                         "review_length", "popularity_score"]
            st.pyplot(plot_heatmap(filtered[corr_cols].corr(), "Feature Correlation", save=False))

    # ----------------------------------------------------------------- #
    # Page 2 — Review Explorer
    # ----------------------------------------------------------------- #
    elif page == "🔍 Review Explorer":
        st.title("🔍 Review Explorer")
        st.caption(
            "Each row aggregates all reviews Amazon scraped for that product "
            "(this dataset is one row per product, not per individual review)."
        )

        search = st.text_input("Search product name or review text")
        results = filter_products(filtered, search_term=search or None)
        st.caption(f"{len(results):,} products found")

        if results.empty:
            st.warning("No products match your search/filters.")
            return

        dist = sentiment_distribution(results)
        if not dist.empty:
            st.bar_chart(dist)

        selected_name = st.selectbox("Select a product to inspect", get_product_options(results))
        row = results[results["product_name"] == selected_name].iloc[0]

        st.subheader(row["product_name"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rating", f"{row['rating']:.1f}")
        m2.metric("Rating count", f"{int(row['rating_count']):,}")
        m3.metric("Price", f"₹{row['discounted_price']:,.0f}")
        m4.metric("Sentiment", row.get("sentiment_label", "n/a"))

        st.markdown("**About this product:**")
        st.write(row["about_product"])
        st.markdown("**Review title(s):**")
        st.write(row["review_title"])
        st.markdown("**Review content:**")
        st.write(row["review_content"])

    # ----------------------------------------------------------------- #
    # Page 3 — Recommendation
    # ----------------------------------------------------------------- #
    else:
        st.title("🎯 Product Recommendation")
        st.caption("Content-based recommendation using TF-IDF over product name + description.")

        recommender = _cached_recommender(df)
        product_name = st.selectbox("Pick a product", get_product_options(df))
        top_k = st.slider("Number of recommendations", 3, 10, 5)

        recs = recommender.recommend(product_name=product_name, top_k=top_k)
        st.dataframe(recs, use_container_width=True)


if __name__ == "__main__":
    _run_app()
