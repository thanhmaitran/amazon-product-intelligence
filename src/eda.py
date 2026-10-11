"""
eda.py
------
Phase 2 — Exploratory Data Analysis. Reads data/processed/products_features.csv
and produces the full set of summary tables + charts (saved to
outputs/figures/) described in the project plan:

  A. Dataset overview (KPI cards)
  B. Category analysis (counts, treemap, avg rating, avg discount)
  C. Pricing analysis (histograms, actual vs discounted, price buckets, log-price)
  D. Rating analysis (histograms, rating by category, rating vs rating_count)
  E. Discount analysis (discount vs rating, discount by category)
  F. Popular products (top 10 by rating / reviews / popularity)
  G. Correlation analysis (numeric feature correlation heatmap)

Run directly:
    python src/eda.py
generates all charts into outputs/figures/ and prints summary tables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.feature_engineering import run_feature_engineering
from src.utils import processed_data_path
from src.visualization import (
    plot_bar, plot_box, plot_hist, plot_scatter, plot_heatmap,
    plot_treemap, plot_kpi_cards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_CORR_COLS = [
    "rating", "rating_count", "discounted_price", "discount_percentage",
    "review_length", "popularity_score",
]


def load_features() -> pd.DataFrame:
    path = processed_data_path("products_features.csv")
    if path.exists():
        return pd.read_csv(path)
    logger.info("products_features.csv not found — building it now")
    return run_feature_engineering()


# --------------------------------------------------------------------------- #
# Section A — Dataset Overview
# --------------------------------------------------------------------------- #

def section_overview(df: pd.DataFrame, save: bool = True) -> dict:
    kpis = {
        "Products": f"{df['product_id'].nunique():,}",
        "Categories": f"{df['main_category'].nunique()}",
        "Avg Rating": f"{df['rating'].mean():.2f}",
        "Avg Discount": f"{df['discount_percentage'].mean() * 100:.0f}%",
        "Avg Price (₹)": f"{df['discounted_price'].mean():,.0f}",
        "Total Reviews": f"{df['rating_count'].sum():,.0f}",
    }
    plot_kpi_cards(kpis, title="Amazon Product Intelligence — Dataset Overview", save=save)
    return kpis


# --------------------------------------------------------------------------- #
# Section B — Category Analysis
# --------------------------------------------------------------------------- #

def section_category(df: pd.DataFrame, save: bool = True) -> dict:
    counts = df["main_category"].value_counts()
    plot_bar(counts.index, counts.values, "Product Count by Category",
             xlabel="Number of products", name="category_counts", save=save)

    plot_treemap(counts.index.tolist(), counts.values.tolist(),
                 "Category Share (Treemap)", name="category_treemap", save=save)

    avg_rating = df.groupby("main_category")["rating"].mean().sort_values(ascending=False)
    plot_bar(avg_rating.index, avg_rating.values, "Average Rating by Category",
              xlabel="Average rating", name="category_avg_rating", save=save)

    plot_box(df, x="main_category", y="discount_percentage",
              title="Discount % Distribution by Category", name="category_discount_box", save=save)

    return {"counts": counts, "avg_rating": avg_rating}


# --------------------------------------------------------------------------- #
# Section C — Pricing Analysis
# --------------------------------------------------------------------------- #

def section_pricing(df: pd.DataFrame, save: bool = True) -> dict:
    plot_hist(df["discounted_price"], "Discounted Price Distribution",
              xlabel="Price (₹)", name="price_histogram", save=save)

    plot_hist(df["discount_percentage"] * 100, "Discount Percentage Distribution",
              xlabel="Discount (%)", name="discount_histogram", save=save)

    plot_scatter(df, x="actual_price", y="discounted_price",
                 title="Actual vs. Discounted Price", hue="price_bucket",
                 name="actual_vs_discounted_price", save=save)

    bucket_counts = df["price_bucket"].value_counts()
    plot_bar(bucket_counts.index, bucket_counts.values, "Products by Price Bucket",
              xlabel="Number of products", name="price_bucket_counts", save=save)

    plot_hist(df["discounted_price"], "Log-Price Distribution",
              xlabel="Price (₹, log scale)", log_scale=True, name="log_price_histogram", save=save)

    skew = df["discounted_price"].skew()
    under_5000_pct = (df["discounted_price"] < 5000).mean() * 100
    insights = {
        "price_skew": skew,
        "pct_under_5000": under_5000_pct,
    }
    logger.info(
        "Pricing insight: skew=%.2f, %.1f%% of products priced under ₹5,000",
        skew, under_5000_pct,
    )
    return insights


# --------------------------------------------------------------------------- #
# Section D — Rating Analysis
# --------------------------------------------------------------------------- #

def section_rating(df: pd.DataFrame, save: bool = True) -> dict:
    plot_hist(df["rating"], "Rating Distribution", xlabel="Rating",
              bins=20, name="rating_histogram", save=save)

    plot_box(df, x="main_category", y="rating", title="Rating by Category",
              name="rating_by_category", save=save)

    plot_hist(df["rating_count"], "Rating Count Distribution", xlabel="Number of ratings",
              log_scale=True, name="rating_count_histogram", save=save)

    plot_scatter(df, x="rating_count", y="rating", title="Rating vs. Rating Count",
                 size="popularity_score", name="rating_vs_rating_count_bubble", save=save)

    return {"mean_rating": df["rating"].mean(), "median_rating": df["rating"].median()}


# --------------------------------------------------------------------------- #
# Section E — Discount Analysis
# --------------------------------------------------------------------------- #

def section_discount(df: pd.DataFrame, save: bool = True) -> dict:
    corr = df[["discount_percentage", "rating"]].corr().iloc[0, 1]

    plot_scatter(df, x="discount_percentage", y="rating",
                 title=f"Discount % vs. Rating (corr={corr:.2f})",
                 name="discount_vs_rating", save=save)

    biggest_discounts = df.groupby("main_category")["discount_percentage"].mean().sort_values(ascending=False)
    plot_bar(biggest_discounts.index, biggest_discounts.values * 100,
              "Average Discount % by Category", xlabel="Avg discount (%)",
              name="discount_by_category", save=save)

    logger.info("Discount~Rating correlation: %.3f", corr)
    return {"discount_rating_corr": corr, "biggest_discount_categories": biggest_discounts}


# --------------------------------------------------------------------------- #
# Section F — Popular Products
# --------------------------------------------------------------------------- #

def section_popular_products(df: pd.DataFrame, save: bool = True) -> dict:
    top_rated = df.nlargest(10, "rating")[["product_name", "rating", "rating_count"]]
    most_reviewed = df.nlargest(10, "rating_count")[["product_name", "rating", "rating_count"]]
    most_popular = df.nlargest(10, "popularity_score")[["product_name", "popularity_score", "rating"]]

    def _short(name: str, n: int = 40) -> str:
        return name if len(name) <= n else name[: n - 1] + "…"

    plot_bar(
        most_popular["product_name"].apply(_short), most_popular["popularity_score"],
        "Top 10 Products by Popularity Score", xlabel="Popularity score",
        name="top10_popularity", save=save, figsize=(9, 6),
    )
    plot_bar(
        most_reviewed["product_name"].apply(_short), most_reviewed["rating_count"],
        "Top 10 Most-Reviewed Products", xlabel="Number of ratings",
        name="top10_most_reviewed", save=save, figsize=(9, 6),
    )

    return {"top_rated": top_rated, "most_reviewed": most_reviewed, "most_popular": most_popular}


# --------------------------------------------------------------------------- #
# Section G — Correlation Analysis
# --------------------------------------------------------------------------- #

def section_correlation(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    corr = df[NUMERIC_CORR_COLS].corr()
    plot_heatmap(corr, "Feature Correlation Matrix", name="correlation_heatmap", save=save)
    return corr


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_eda(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        df = load_features()

    results = {}
    results["overview"] = section_overview(df, save=save)
    results["category"] = section_category(df, save=save)
    results["pricing"] = section_pricing(df, save=save)
    results["rating"] = section_rating(df, save=save)
    results["discount"] = section_discount(df, save=save)
    results["popular_products"] = section_popular_products(df, save=save)
    results["correlation"] = section_correlation(df, save=save)
    logger.info("EDA complete — charts saved to outputs/figures/")
    return results


if __name__ == "__main__":
    run_eda()
