# Results

All figures referenced below are in `outputs/figures/`; all numbers are
computed directly from `data/processed/products_features.csv` (1,351
products, post-deduplication) by the corresponding `src/` module — see
`docs/methodology.md` for how each was produced.

## Which product categories receive the highest customer ratings?

| Category | Avg. rating | # Products |
|---|---|---|
| Office Products | 4.31 | 31 |
| Toys & Games | 4.30 | 1 |
| Home Improvement | 4.25 | 2 |
| Computers & Accessories | 4.15 | 375 |
| Electronics | 4.08 | 490 |
| Home & Kitchen | 4.04 | 448 |
| Health & Personal Care | 4.00 | 1 |
| Musical Instruments | 3.90 | 2 |
| Car & Motorbike | 3.80 | 1 |

The three highest-rated categories each have very few listings (1–31
products), so their averages are noisy and shouldn't be over-read. Among the
three categories with substantial sample sizes (Computers & Accessories,
Electronics, Home & Kitchen — 375/490/448 products respectively),
**Computers & Accessories rates highest (4.15)**, and a Kruskal-Wallis test
confirms rating differs significantly across categories overall
(H = 57.53, p ≈ 2×10⁻¹², see `03_statistics.ipynb`).

## Does a larger discount correspond to higher popularity or more reviews?

**No — if anything, the opposite.** Discount percentage correlates weakly
**negatively** with both rating (Pearson-adjacent Spearman check: r = −0.16 in
the full correlation matrix) and `popularity_score` (Pearson r = −0.165,
p ≈ 9.9×10⁻¹⁰, statistically significant — see Test 4 in `03_statistics.ipynb`).
Discount strategy also varies sharply by category (ANOVA F = 62.28,
p ≈ 1×10⁻³⁷): Home Improvement discounts the most on average (58%), while
Office Products (12%) and Toys & Games (0%) discount the least. Practically,
heavier discounting in this dataset looks more like a lever used on
lower-momentum products than a driver of popularity.

## Which keywords dominate positive reviews?

Across all reviews, the most frequent terms are dominated by satisfaction and
usability language: **good** (5,911 mentions), **product** (3,644),
**quality** (2,224), **use** (1,648), **price** (1,264), **like** (1,146),
**cable** (1,027), **phone** (1,012), **easy** (922). This tracks with the
dataset's heavily positive sentiment skew (97.8% of products score as
"positive" on the rule-based baseline — see below).

TF-IDF keywords per category surface more specific vocabulary once generic
sentiment words are down-weighted: **Computers & Accessories** →
cable/charging/mouse/fast; **Electronics** → tv/phone/watch/battery/sound/
camera; **Home & Kitchen** → water/easy/money/value; **Office Products** →
pen/pages/parker/ink/writing.

Topic modelling (6 NMF topics) recovers coherent sub-domains without manual
labelling: charging cables (cable/charging/fast/usb), phones & cameras
(phone/camera/battery/mobile), TVs (tv/remote/picture/sound), smartwatches
(watch/battery/smartwatch/calling), and audio/peripherals (sound/mouse/bass/
keyboard). An earlier pass surfaced a spurious topic dominated by leaked
image-URL fragments — a Phase 1 text-cleaning gap (see `methodology.md` §4)
found via this exact analysis and fixed at the source.

**Sentiment:** 97.8% of products score positive, 0.9% neutral, 1.3% negative
on the rule-based baseline (VADER, or the built-in fallback lexicon in this
environment — see the "Optional Dependencies" note). This skew is consistent
across every category (96–100% positive), matching the dataset's high average
rating (4.09/5) and the general pattern that visible Amazon reviews
over-represent satisfied customers.

## Which products are recommended for a given query?

For the query **"Wayona Nylon Braided USB"**, the TF-IDF content-based
recommender (`TfidfRecommender`) returns:

| Rank | Product | Category | Rating | Price | Similarity |
|---|---|---|---|---|---|
| 1 | Wayona Nylon Braided USB Syncing/Charging Cable (Pack of 2) | Computers & Accessories | 4.2 | ₹649 | 0.948 |
| 2 | Wayona Nylon Braided 3A Lightning to USB A Cable | Computers & Accessories | 4.2 | ₹399 | 0.916 |
| 3 | Wayona Nylon Braided (2 Pack) Lightning Fast USB Cable | Computers & Accessories | 4.2 | ₹649 | 0.915 |
| 4 | Wayona USB Nylon Braided Data Sync Cable | Computers & Accessories | 4.2 | ₹399 | 0.770 |
| 5 | Wayona Nylon Braided Lightning USB Data Sync Cable | Computers & Accessories | 4.2 | ₹399 | 0.743 |

All five results are close variants of the same brand/product family, with
similarity scores above 0.74 — a strong result given the recommender uses
only product name + description text. Evaluated offline across 200 sampled
products, **category-precision@5 = 0.94**: 94% of top-5 recommendations share
the queried product's category, a reasonable proxy for topical relevance
without labeled relevance data (see `methodology.md` §9 for the metric's
limitations — it will undervalue legitimate cross-category recommendations).

## Can rating be predicted from price, discount, category, and review text?

**Classification** (`high_rated` = rating ≥ 4.2): Random Forest is the
strongest model (ROC-AUC 0.775, accuracy 0.730, F1 0.670), ahead of Gradient
Boosting (ROC-AUC 0.760) and the Logistic Regression baseline (ROC-AUC 0.670).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.637 | 0.626 | 0.536 | 0.578 | 0.670 |
| Random Forest | **0.730** | **0.771** | 0.592 | **0.670** | **0.775** |
| Gradient Boosting | 0.715 | 0.745 | 0.584 | 0.655 | 0.760 |

**Regression** (predicting `rating` directly): all three models explain very
little variance (R² 0.04–0.11), though typical error is modest in absolute
terms (MAE 0.20–0.22 stars).

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.220 | 0.304 | 0.088 |
| Random Forest | **0.205** | **0.300** | **0.112** |
| Gradient Boosting | 0.221 | 0.312 | 0.041 |

**Interpretation:** this is an informative negative result. `rating_count`
and `discount_percentage` are the strongest features (permutation importance),
followed by `discounted_price` and `main_category`; text-length features
contribute least. The low R² indicates rating is driven mostly by factors
*not* captured here — actual product quality, brand reputation, individual
reviewer experience — rather than by pricing or discounting strategy.
`popularity_score` was deliberately excluded from all models (it's derived
directly from `rating` and would leak the target).

## Summary of statistical tests

| Question | Test | Statistic | p-value | Verdict (α=0.05) |
|---|---|---|---|---|
| Rating differs across categories? | Kruskal-Wallis | H = 57.53 | 1.98×10⁻¹² | Reject H0 |
| Discount % differs across categories? | ANOVA | F = 62.28 | 1.07×10⁻³⁷ | Reject H0 |
| Price associated with rating? | Spearman | ρ = 0.077 | 0.0047 | Reject H0 |
| Discount associated with popularity? | Pearson | r = −0.165 | 9.87×10⁻¹⁰ | Reject H0 |
| Review length differs by rating group? | Mann-Whitney U | U = 250,017.5 | 0.0011 | Reject H0 |

All five tests reach significance, but effect sizes for the correlation tests
(ρ = 0.077, r = −0.165) are small — with n≈1,300, statistical significance
here should not be read as strong practical significance.
