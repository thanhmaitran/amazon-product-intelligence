"""
machine_learning.py
--------------------
Phase 6 — Machine Learning Models.

Task 1 — Rating Classification
    Target: high_rated = rating >= 4.2 (binary)
    Models: Logistic Regression (baseline), Random Forest, Gradient Boosting
            (stand-in for XGBoost — `xgboost` isn't part of the core
            dependency footprint here; swap in XGBClassifier for a drop-in
            upgrade, same sklearn-style .fit/.predict_proba API)

Task 2 — Rating Regression
    Target: rating (continuous)
    Metrics: MAE, RMSE, R²
    Feature importance: model-native importances, or SHAP if installed
    (falls back to sklearn's permutation_importance otherwise)

IMPORTANT — leakage guard: `popularity_score` is EXCLUDED from the feature
set because it's engineered as `rating * log1p(rating_count)` in
feature_engineering.py, i.e. it's a direct function of the target and would
leak it into both tasks.

Run directly:
    python src/machine_learning.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda import load_features
from src.visualization import save_fig, set_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RATING_THRESHOLD = 4.2
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "discount_percentage", "discounted_price", "review_length",
    "title_length", "description_length", "rating_count",
]
CATEGORICAL_FEATURES = ["main_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _prepare_xy(df: pd.DataFrame, target: str):
    data = df.dropna(subset=FEATURES + [target]).copy()
    X = data[FEATURES]
    y = data[target]
    return X, y


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


# --------------------------------------------------------------------------- #
# Task 1 — Classification
# --------------------------------------------------------------------------- #

def build_classification_target(df: pd.DataFrame, threshold: float = RATING_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    df["high_rated"] = (df["rating"] >= threshold).astype(int)
    return df


def train_classifiers(df: pd.DataFrame, save: bool = True) -> dict:
    df = build_classification_target(df)
    X, y = _prepare_xy(df, "high_rated")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        results[name] = {"pipeline": pipe, "metrics": metrics, "y_test": y_test, "y_proba": y_proba, "y_pred": y_pred}
        logger.info("[%s] %s", name, {k: round(v, 3) for k, v in metrics.items()})

    if save:
        _plot_classification_comparison(results)
        best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
        _plot_confusion_matrix(results[best_name]["y_test"], results[best_name]["y_pred"], best_name)
        _plot_roc_curves(results)

    return results


def _plot_classification_comparison(results: dict, save: bool = True):
    from src.visualization import plot_bar
    metric = "roc_auc"
    names = list(results.keys())
    scores = [results[n]["metrics"][metric] for n in names]
    plot_bar(names, scores, "Model Comparison — ROC-AUC (Rating Classification)",
              xlabel="ROC-AUC", horizontal=True, name="classification_model_comparison", save=save)


def _plot_confusion_matrix(y_test, y_pred, model_name: str, save: bool = True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not high-rated", "High-rated"],
                yticklabels=["Not high-rated", "High-rated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    save_fig(fig, "confusion_matrix_best_model", save)


def _plot_roc_curves(results: dict, save: bool = True):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Rating Classification")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "roc_curves", save)


# --------------------------------------------------------------------------- #
# Task 2 — Regression
# --------------------------------------------------------------------------- #

def train_regressors(df: pd.DataFrame, save: bool = True) -> dict:
    X, y = _prepare_xy(df, "rating")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
            "R2": r2_score(y_test, y_pred),
        }
        results[name] = {"pipeline": pipe, "metrics": metrics, "y_test": y_test, "y_pred": y_pred}
        logger.info("[%s] %s", name, {k: round(v, 3) for k, v in metrics.items()})

    if save:
        _plot_regression_comparison(results)
        best_name = max(results, key=lambda n: results[n]["metrics"]["R2"])
        _plot_predicted_vs_actual(results[best_name]["y_test"], results[best_name]["y_pred"], best_name)
        feature_importance(results[best_name]["pipeline"], X_train, save=save)

    return results


def _plot_regression_comparison(results: dict, save: bool = True):
    from src.visualization import plot_bar
    names = list(results.keys())
    scores = [results[n]["metrics"]["R2"] for n in names]
    plot_bar(names, scores, "Model Comparison — R² (Rating Regression)",
              xlabel="R²", horizontal=True, name="regression_model_comparison", save=save)


def _plot_predicted_vs_actual(y_test, y_pred, model_name: str, save: bool = True):
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.6)
    ax.set_xlabel("Actual rating")
    ax.set_ylabel("Predicted rating")
    ax.set_title(f"Predicted vs. Actual Rating — {model_name}")
    fig.tight_layout()
    save_fig(fig, "predicted_vs_actual_rating", save)


# --------------------------------------------------------------------------- #
# Explainability: SHAP if available, else permutation importance
# --------------------------------------------------------------------------- #

def _feature_names_out(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["prep"].get_feature_names_out())


def feature_importance(pipeline: Pipeline, X_sample: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    try:
        import shap
        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        X_trans = prep.transform(X_sample)
        explainer = shap.Explainer(model, X_trans)
        shap_values = explainer(X_trans)
        importances = np.abs(shap_values.values).mean(axis=0)
        names = _feature_names_out(pipeline)  # transformed (one-hot expanded) names
        method = "SHAP mean(|value|)"
    except ImportError:
        logger.warning(
            "`shap` not installed — falling back to sklearn permutation_importance. "
            "Run `pip install shap` for true SHAP explanations."
        )
        from sklearn.inspection import permutation_importance
        # permutation_importance operates on the *whole pipeline* with the
        # original (pre-one-hot) input, so importances line up with the raw
        # input columns, not the transformed/expanded feature names.
        result = permutation_importance(
            pipeline, X_sample, pipeline.predict(X_sample),
            n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
        )
        importances = result.importances_mean
        names = list(X_sample.columns)  # raw input column names
        method = "Permutation importance"

    imp_df = pd.DataFrame({"feature": names, "importance": importances}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    if save:
        from src.visualization import plot_bar
        top = imp_df.head(15)
        plot_bar(top["feature"], top["importance"], f"Feature Importance ({method})",
                  xlabel="Importance", name="feature_importance", save=save)

    return imp_df


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def run_ml(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    if df is None:
        df = load_features()

    classification_results = train_classifiers(df, save=save)
    regression_results = train_regressors(df, save=save)

    return {"classification": classification_results, "regression": regression_results}


if __name__ == "__main__":
    results = run_ml()
    print("\n=== Classification (high_rated >= {:.1f}) ===".format(RATING_THRESHOLD))
    for name, res in results["classification"].items():
        print(f"{name}: {res['metrics']}")
    print("\n=== Regression (rating) ===")
    for name, res in results["regression"].items():
        print(f"{name}: {res['metrics']}")
