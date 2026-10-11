# Statistical Tests

```
Q1. Does rating differ between product categories?
    H0: The median rating is the same across all categories.
    H1: At least one category has a different median rating.
    Test: Kruskal-Wallis H-test | statistic=57.5282 | p-value=1.982e-12
    Verdict (alpha=0.05): Reject H0
    Interpretation: Median rating differs across at least one category pair.

Q2. Does discount percentage differ between product categories?
    H0: The mean discount percentage is the same across all categories.
    H1: At least one category has a different mean discount percentage.
    Test: One-way ANOVA | statistic=62.2811 | p-value=1.071e-37
    Verdict (alpha=0.05): Reject H0
    Interpretation: Mean discount percentage differs across at least one category pair.

Q3. Is product price associated with rating?
    H0: There is no monotonic association between price and rating (rho = 0).
    H1: There is a monotonic association between price and rating (rho != 0).
    Test: Spearman rank correlation | statistic=0.0768 | p-value=0.004731
    Verdict (alpha=0.05): Reject H0
    Interpretation: A statistically significant positive monotonic relationship (rho=0.077) exists between price and rating.

Q4. Is discount percentage associated with product popularity?
    H0: There is no linear association between discount percentage and popularity score (r = 0).
    H1: There is a linear association between discount percentage and popularity score (r != 0).
    Test: Pearson correlation | statistic=-0.1653 | p-value=9.873e-10
    Verdict (alpha=0.05): Reject H0
    Interpretation: A statistically significant negative linear relationship (r=-0.165) exists between discount percentage and popularity score.

Q5. Does review length differ between high-rated (>= 4.2) and lower-rated products?
    H0: Review length distributions are the same for high-rated and lower-rated products.
    H1: Review length distributions differ for high-rated and lower-rated products.
    Test: Mann-Whitney U test | statistic=250017.5000 | p-value=0.001051
    Verdict (alpha=0.05): Reject H0
    Interpretation: Review length distributions differ between high-rated (>= 4.2) and lower-rated products.

```
