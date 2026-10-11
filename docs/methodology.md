# Methodology

This document describes the full data science methodology behind the *Amazon
Product Intelligence* project, from raw data to deployed dashboard. It is
written to be read alongside the corresponding notebooks in `notebooks/` and
implementation in `src/`.

## 1. Dataset

The raw dataset (`data/raw/amazon.csv`) contains 1,465 Amazon India product
listings scraped at the product level — one row per product, with all of that
product's scraped customer reviews concatenated into single `review_title` /
`review_content` fields (comma-joined), rather than one row per individual
review. Key fields: `product_id`, `product_name`, `category` (a pipe-delimited
hierarchy, e.g. `Electronics|Mobiles&Accessories|...`), `discounted_price`,
`actual_price`, `discount_percentage`, `rating`, `rating_count`,
`about_product`, and reviewer/review metadata.

Two data-quality issues were identified and corrected during Phase 1 (see
§2): 114 duplicate `product_id` rows, and a formatting artifact in
`review_content` that embedded raw image URLs inline with review text (traced
back to how the source scrape joined fields — see §4 for how this was
discovered).

## 2. Data Cleaning (Phase 1 — `src/preprocessing.py`)

1. **Deduplication.** Rows are deduplicated on `product_id`, keeping the
   first occurrence. This removed 114 rows (1,465 → 1,351).
2. **Numeric parsing.** `discounted_price` / `actual_price` are stripped of
   the `₹` symbol and thousands separators and cast to float
   (`utils.parse_currency`). `rating_count` is parsed the same way
   (`utils.parse_count`). `discount_percentage` is converted from a `"64%"`
   string to a `0.64` float (`utils.parse_percentage`).
3. **Category hierarchy.** The pipe-delimited `category` field is split into
   `main_category` (first segment), `sub_category` (second segment), and
   `product_type` (last segment). Segment names are converted from
   concatenated CamelCase (e.g. `Computers&Accessories`, `USBCables`) to
   readable text via two regex passes: a lower/digit→upper transition
   (`fastCharging` → `fast Charging`) and an acronym-run→new-word transition
   (`USBCables` → `USB Cables`), plus `&` → ` & ` spacing.
4. **Text cleaning.** `about_product`, `review_title`, and `review_content`
   are each cleaned into a matching `*_clean` column: lowercased, with URLs,
   HTML tags, emoji, and punctuation stripped (punctuation is replaced with a
   space, not deleted, to avoid merging adjacent tokens — e.g. `X/XsMax/Xr`
   stays three tokens instead of collapsing into `xxsmaxxr`), followed by
   stopword removal and lemmatization (via NLTK when available; a small
   built-in stopword list is used as a fallback with lemmatization skipped
   when NLTK isn't installed, so the pipeline never hard-fails on a missing
   optional dependency).

## 3. Feature Engineering (Phase 1 — `src/feature_engineering.py`)

- `discount_amount = actual_price − discounted_price`
- `review_length`, `title_length`, `description_length` — word counts of the
  respective raw text fields
- `popularity_score = rating × log1p(rating_count)` — a single score
  balancing rating quality against review volume
- `price_bucket` (Budget / Mid / Premium) and `discount_bucket` (Low / Medium
  / High) — tertile bins of `discounted_price` and `discount_percentage`

**Leakage note:** `popularity_score` is a direct function of `rating`, so it
is excluded from the feature set in every predictive model in Phase 6 (see
§6) to avoid leaking the target.

## 4. Exploratory Data Analysis (Phase 2 — `src/eda.py`)

EDA is organized into seven sections — dataset overview, category, pricing,
rating, discount, popular-products, and correlation analysis — producing 18
charts in `outputs/figures/`. All charts are built on shared primitives in
`src/visualization.py`, including a treemap implemented from scratch with the
classic squarified-treemap algorithm (no `squarify` dependency required).

It was during this phase's NLP follow-through (topic modelling, §5) that the
image-URL contamination in `review_content` was discovered: an early topic
model run surfaced a topic dominated by `https`, `jpg`, `webp`, and `media`
tokens. This was traced to the raw scrape's field-joining format and fixed by
adding URL stripping to `utils.basic_clean`, then re-running the Phase 1
pipeline — a concrete example of later-phase analysis catching an earlier-phase
data-quality issue.

## 5. Statistical Analysis (Phase 3 — `src/statistics.py`)

Five hypothesis tests are run via `scipy.stats`, each reported with H0, H1,
test statistic, p-value, and an interpretation at α = 0.05:

| # | Question | Test |
|---|---|---|
| 1 | Does rating differ across categories? | Kruskal-Wallis |
| 2 | Does discount % differ across categories? | One-way ANOVA |
| 3 | Is price associated with rating? | Spearman correlation |
| 4 | Is discount associated with popularity? | Pearson correlation |
| 5 | Does review length differ by rating group? | Mann-Whitney U |

All five reject H0 at α = 0.05 (see `docs/results.md` for statistics and
interpretation). With n≈1,300, small effect sizes can reach significance, so
results are read alongside effect sizes, not p-values alone.

## 6. NLP Pipeline (Phase 4 — `src/nlp.py`)

- **N-gram frequency** (unigrams/bigrams/trigrams) via `CountVectorizer`.
- **Word cloud** via the `wordcloud` package when installed; otherwise a
  top-terms bar chart is substituted automatically.
- **TF-IDF keywords per category** via `TfidfVectorizer`, ranked by mean
  TF-IDF score within each category's reviews.
- **Topic modelling**: 6 topics via NMF (`sklearn.decomposition.NMF`) over a
  TF-IDF matrix of `review_content_clean` — chosen over LDA+gensim to avoid
  an extra dependency, with comparable topic coherence at this corpus size.

## 7. Sentiment Analysis (Phase 5 — `src/sentiment.py`)

A rule-based baseline scores each product's concatenated review text with
VADER (`vaderSentiment`) when installed; otherwise a small built-in
positive/negative lexicon produces a comparable compound-style score as a
fallback. Scores are bucketed into positive / neutral / negative at ±0.05. An
optional DistilBERT transformer comparison (`transformers` pipeline) runs
when the package and model weights are available, with an agreement-rate
comparison against the rule-based labels.

## 8. Machine Learning (Phase 6 — `src/machine_learning.py`)

**Task 1 — Classification:** target `high_rated = rating ≥ 4.2`; features are
`discount_percentage`, `discounted_price`, `review_length`, `title_length`,
`description_length`, `rating_count`, and one-hot-encoded `main_category`.
Logistic Regression (baseline, with `class_weight="balanced"`), Random
Forest, and Gradient Boosting (a scikit-learn stand-in for XGBoost) are
compared by accuracy/precision/recall/F1/ROC-AUC on a stratified 80/20 split.

**Task 2 — Regression:** same features, target `rating`; Linear Regression,
Random Forest, and Gradient Boosting compared by MAE/RMSE/R².

**Explainability:** SHAP (`shap.Explainer`) when installed; otherwise
scikit-learn's `permutation_importance` on the fitted pipeline, using the
original (pre-one-hot) feature names so importances remain interpretable.

## 9. Recommendation System (Phase 7 — `src/recommendation.py`)

A content-based recommender computes TF-IDF vectors over
`product_name + about_product_clean` and ranks candidates by cosine
similarity (`TfidfRecommender`). An optional embedding-based recommender
(`sentence-transformers`, `all-MiniLM-L6-v2`) implements the same interface
and is used automatically when the package and model weights are available.
Offline evaluation uses **category-precision@k**: the fraction of a sampled
product's top-k recommendations sharing its `main_category`, as a proxy for
topical relevance in the absence of labeled relevance judgments.

## 10. Dashboard (Phase 8 — `dashboard/streamlit_app.py`)

A three-page Streamlit app exposes the above pipeline interactively:
Business Dashboard (KPIs + charts, filterable by category/price/rating),
Review Explorer (search + per-product review/sentiment inspection), and
Recommendation (the TF-IDF recommender with a product picker). The app's data
logic (filtering, KPI computation, sentiment scoring) is implemented as plain
functions independent of Streamlit's UI calls, so it is testable via
`python dashboard/streamlit_app.py` without a running server.

## Reproducibility & Graceful Degradation

Every phase past Phase 1 depends on at least one optional third-party
package that may be unavailable in a given environment (no network access,
or a deliberately minimal install). Each such dependency has a documented,
functioning fallback (see the "Optional Dependencies" table in the main
`README.md`), so `python src/<module>.py` and every notebook run end-to-end
regardless of which optional packages are installed — with a log message
noting when and why a fallback was used.
