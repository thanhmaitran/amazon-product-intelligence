"""
statistics.py
-------------
Phase 3 — Statistical Analysis. Runs 5 hypothesis tests against
data/processed/products_features.csv, each returned as a structured dict with
H0, H1, test statistic, p-value, and a plain-language interpretation.

  1. Does rating differ across categories?          -> Kruskal-Wallis
  2. Does discount % differ across categories?       -> One-way ANOVA
  3. Is price associated with rating?                -> Spearman correlation
  4. Is discount associated with popularity?         -> Pearson correlation
  5. Does review length differ by rating group?       -> Mann-Whitney U

Uses scipy.stats only (no statsmodels dependency needed for these five tests).

Run directly:
    python src/statistics.py
prints a formatted report and returns a list of result dicts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.eda import load_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ALPHA = 0.05


def _verdict(p_value: float, alpha: float = ALPHA) -> str:
    return "Reject H0" if p_value < alpha else "Fail to reject H0"


def _result(question, h0, h1, test_name, statistic, p_value, interpretation) -> dict:
    return {
        "question": question,
        "H0": h0,
        "H1": h1,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": ALPHA,
        "verdict": _verdict(p_value),
        "interpretation": interpretation,
    }


# --------------------------------------------------------------------------- #
# 1. Rating across categories — Kruskal-Wallis
# --------------------------------------------------------------------------- #

def test_rating_across_categories(df: pd.DataFrame) -> dict:
    groups = [g["rating"].dropna().values for _, g in df.groupby("main_category") if len(g) >= 5]
    stat, p = stats.kruskal(*groups)
    interp = (
        "Median rating differs across at least one category pair."
        if p < ALPHA else
        "No statistically significant difference in rating medians across categories."
    )
    return _result(
        question="Does rating differ between product categories?",
        h0="The median rating is the same across all categories.",
        h1="At least one category has a different median rating.",
        test_name="Kruskal-Wallis H-test",
        statistic=stat, p_value=p, interpretation=interp,
    )


# --------------------------------------------------------------------------- #
# 2. Discount % across categories — One-way ANOVA
# --------------------------------------------------------------------------- #

def test_discount_across_categories(df: pd.DataFrame) -> dict:
    groups = [g["discount_percentage"].dropna().values for _, g in df.groupby("main_category") if len(g) >= 5]
    stat, p = stats.f_oneway(*groups)
    interp = (
        "Mean discount percentage differs across at least one category pair."
        if p < ALPHA else
        "No statistically significant difference in mean discount across categories."
    )
    return _result(
        question="Does discount percentage differ between product categories?",
        h0="The mean discount percentage is the same across all categories.",
        h1="At least one category has a different mean discount percentage.",
        test_name="One-way ANOVA",
        statistic=stat, p_value=p, interpretation=interp,
    )


# --------------------------------------------------------------------------- #
# 3. Price vs rating — Spearman correlation
# --------------------------------------------------------------------------- #

def test_price_vs_rating(df: pd.DataFrame) -> dict:
    sub = df[["discounted_price", "rating"]].dropna()
    rho, p = stats.spearmanr(sub["discounted_price"], sub["rating"])
    direction = "positive" if rho > 0 else "negative"
    interp = (
        f"A statistically significant {direction} monotonic relationship (rho={rho:.3f}) "
        "exists between price and rating."
        if p < ALPHA else
        "No statistically significant monotonic relationship between price and rating."
    )
    return _result(
        question="Is product price associated with rating?",
        h0="There is no monotonic association between price and rating (rho = 0).",
        h1="There is a monotonic association between price and rating (rho != 0).",
        test_name="Spearman rank correlation",
        statistic=rho, p_value=p, interpretation=interp,
    )


# --------------------------------------------------------------------------- #
# 4. Discount vs popularity — Pearson correlation
# --------------------------------------------------------------------------- #

def test_discount_vs_popularity(df: pd.DataFrame) -> dict:
    sub = df[["discount_percentage", "popularity_score"]].dropna()
    r, p = stats.pearsonr(sub["discount_percentage"], sub["popularity_score"])
    direction = "positive" if r > 0 else "negative"
    interp = (
        f"A statistically significant {direction} linear relationship (r={r:.3f}) "
        "exists between discount percentage and popularity score."
        if p < ALPHA else
        "No statistically significant linear relationship between discount and popularity."
    )
    return _result(
        question="Is discount percentage associated with product popularity?",
        h0="There is no linear association between discount percentage and popularity score (r = 0).",
        h1="There is a linear association between discount percentage and popularity score (r != 0).",
        test_name="Pearson correlation",
        statistic=r, p_value=p, interpretation=interp,
    )


# --------------------------------------------------------------------------- #
# 5. Review length by rating group — Mann-Whitney U
# --------------------------------------------------------------------------- #

def test_review_length_by_rating_group(df: pd.DataFrame, threshold: float = 4.2) -> dict:
    high = df.loc[df["rating"] >= threshold, "review_length"].dropna()
    low = df.loc[df["rating"] < threshold, "review_length"].dropna()
    stat, p = stats.mannwhitneyu(high, low, alternative="two-sided")
    interp = (
        f"Review length distributions differ between high-rated (>= {threshold}) and "
        "lower-rated products."
        if p < ALPHA else
        f"No statistically significant difference in review length between high-rated "
        f"(>= {threshold}) and lower-rated products."
    )
    return _result(
        question=f"Does review length differ between high-rated (>= {threshold}) and lower-rated products?",
        h0="Review length distributions are the same for high-rated and lower-rated products.",
        h1="Review length distributions differ for high-rated and lower-rated products.",
        test_name="Mann-Whitney U test",
        statistic=stat, p_value=p, interpretation=interp,
    )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

TESTS = [
    test_rating_across_categories,
    test_discount_across_categories,
    test_price_vs_rating,
    test_discount_vs_popularity,
    test_review_length_by_rating_group,
]


def run_all_tests(df: pd.DataFrame | None = None) -> list[dict]:
    if df is None:
        df = load_features()

    results = [test(df) for test in TESTS]
    for r in results:
        logger.info(
            "[%s] stat=%.4f p=%.4g -> %s", r["test"], r["statistic"], r["p_value"], r["verdict"]
        )
    return results


def report(results: list[dict] | None = None) -> str:
    results = results or run_all_tests()
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"Q{i}. {r['question']}")
        lines.append(f"    H0: {r['H0']}")
        lines.append(f"    H1: {r['H1']}")
        lines.append(f"    Test: {r['test']} | statistic={r['statistic']:.4f} | p-value={r['p_value']:.4g}")
        lines.append(f"    Verdict (alpha={r['alpha']}): {r['verdict']}")
        lines.append(f"    Interpretation: {r['interpretation']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
