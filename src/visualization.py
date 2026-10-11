"""
visualization.py
-----------------
Shared plotting helpers used across eda.py, nlp.py, sentiment.py, and the
notebooks. Centralizing style + save logic here means every chart in the
project looks consistent and lands in outputs/figures/ the same way.

All `plot_*` functions return the matplotlib Figure (so callers/notebooks can
display it inline) AND save a PNG to outputs/figures/ when `save=True`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = project_root() / "outputs" / "figures"

PALETTE = "viridis"


def set_style() -> None:
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"


def save_fig(fig: plt.Figure, name: str, save: bool = True) -> Path | None:
    if not save:
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    logger.info("Saved figure: %s", path)
    return path


# --------------------------------------------------------------------------- #
# Generic chart primitives
# --------------------------------------------------------------------------- #

def plot_bar(
    labels, values, title, xlabel="", ylabel="", horizontal=True,
    name: str | None = None, save: bool = True, figsize=(8, 5),
):
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        sns.barplot(x=list(values), y=list(labels), ax=ax, hue=list(labels), legend=False)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
    else:
        sns.barplot(x=list(labels), y=list(values), ax=ax, hue=list(labels), legend=False)
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "")
        plt.xticks(rotation=45, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, name or title.lower().replace(" ", "_"), save)
    return fig


def plot_hist(series, title, xlabel="", bins=30, log_scale=False,
              name: str | None = None, save: bool = True, figsize=(8, 5)):
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    data = series.dropna()
    sns.histplot(data, bins=bins, kde=True, ax=ax, log_scale=log_scale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    save_fig(fig, name or title.lower().replace(" ", "_"), save)
    return fig


def plot_box(df: pd.DataFrame, x: str, y: str, title, name: str | None = None,
             save: bool = True, figsize=(9, 5), rotate_x=True):
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    order = df.groupby(x)[y].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=x, y=y, order=order, ax=ax, hue=x, legend=False)
    ax.set_title(title)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, name or title.lower().replace(" ", "_"), save)
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, title, hue: str | None = None,
                  size: str | None = None, name: str | None = None, save: bool = True,
                  figsize=(8, 6), alpha=0.6):
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, alpha=alpha, ax=ax, sizes=(20, 300))
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, name or title.lower().replace(" ", "_"), save)
    return fig


def plot_heatmap(corr: pd.DataFrame, title, name: str | None = None, save: bool = True, figsize=(8, 7)):
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, name or title.lower().replace(" ", "_"), save)
    return fig


# --------------------------------------------------------------------------- #
# Treemap — implemented from scratch (squarified algorithm) since the
# `squarify` package isn't part of the project's core dependency footprint.
# --------------------------------------------------------------------------- #

def _squarify(values, x, y, w, h):
    """Classic squarified treemap layout. Returns list of (x, y, w, h) rects
    in the same order as `values` (values must be pre-sorted descending)."""
    if not values:
        return []
    if len(values) == 1:
        return [(x, y, w, h)]

    total = sum(values)
    # split so the first row's total is close to half, minimizing aspect ratio
    best_split, best_ratio = 1, float("inf")
    for i in range(1, len(values)):
        row = values[:i]
        row_area = sum(row) / total * (w * h)
        side = w if w < h else h
        long_side = row_area / side if side else 0
        ratios = [max(long_side / (v / total * (w * h) / side) if v else 1,
                      (v / total * (w * h) / side) / long_side if long_side else 1)
                  for v in row]
        worst = max(ratios) if ratios else float("inf")
        if worst < best_ratio:
            best_ratio, best_split = worst, i

    row = values[:best_split]
    rest = values[best_split:]
    row_area = sum(row) / total * (w * h)

    rects = []
    if w >= h:
        row_w = row_area / h if h else 0
        cy = y
        for v in row:
            rh = (v / sum(row)) * h if sum(row) else 0
            rects.append((x, cy, row_w, rh))
            cy += rh
        rects += _squarify(rest, x + row_w, y, w - row_w, h)
    else:
        row_h = row_area / w if w else 0
        cx = x
        for v in row:
            rw = (v / sum(row)) * w if sum(row) else 0
            rects.append((cx, y, rw, row_h))
            cx += rw
        rects += _squarify(rest, x, y + row_h, w, h - row_h)
    return rects


def plot_treemap(labels, values, title, name: str | None = None, save: bool = True, figsize=(10, 6)):
    set_style()
    pairs = sorted(zip(values, labels), reverse=True)
    sorted_values = [v for v, _ in pairs]
    sorted_labels = [l for _, l in pairs]
    rects = _squarify(sorted_values, 0, 0, 100, 100)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(PALETTE)
    colors = cmap(np.linspace(0.15, 0.9, len(sorted_labels)))
    for (rx, ry, rw, rh), label, val, color in zip(rects, sorted_labels, sorted_values, colors):
        ax.add_patch(plt.Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor="white", linewidth=2))
        if rw > 6 and rh > 4:
            ax.text(rx + rw / 2, ry + rh / 2, f"{label}\n({val:g})", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, name or title.lower().replace(" ", "_"), save)
    return fig


def plot_kpi_cards(kpis: dict, title="Dataset Overview", name: str | None = None,
                    save: bool = True, figsize=(12, 3)):
    """Render a row of simple KPI 'cards' (big number + label)."""
    set_style()
    fig, axes = plt.subplots(1, len(kpis), figsize=figsize)
    if len(kpis) == 1:
        axes = [axes]
    for ax, (label, value) in zip(axes, kpis.items()):
        ax.text(0.5, 0.6, f"{value}", ha="center", va="center", fontsize=22, fontweight="bold")
        ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=11, color="dimgray")
        ax.axis("off")
        ax.set_facecolor("#f7f7f9")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, name or "kpi_overview", save)
    return fig
