# Analyzing Customer Behavior for E-commerce Insights

Take-home assignment for the **Intelligent Systems Services Engineer** role at Npontu Technologies.

**Author:** Kingsley Buabeng ([kbuabeng06@gmail.com](mailto:kbuabeng06@gmail.com))

## Objective

Analyze customer activity on a synthetic e-commerce platform to extract actionable business insights and build a churn prediction model, using a Ghana-market dataset generated with statistically realistic distributions rather than pure randomness.

## Repository Structure

| File | Description |
|---|---|
| `generate_data.py` | Generates the full synthetic dataset (4 linked tables) via parametric statistical simulation |
| `customers.csv` | 3,000 customers across all 16 Ghana regions / 59 towns, signup dates 2023-2026 |
| `sessions.csv` | 48,222 browsing sessions |
| `product_interactions.csv` | 227,248 view/cart-add/wishlist events across a 302-product, 10-category catalog |
| `purchases.csv` | 10,614 completed purchases |
| `ecommerce_dataset.xlsx` | All four tables combined into one workbook, for reviewers who prefer a single file |
| `analysis.ipynb` | **Main deliverable.** EDA, data cleaning, churn label construction, feature engineering, modeling, explainability, and business insights |
| `customer_features_scored.csv` | Per-customer feature table with model predictions, exported by `analysis.ipynb` for use by the Elasticsearch notebook |
| `elasticsearch_demo.ipynb` | Big-data-tool component — indexes the data into Elasticsearch and runs aggregation queries |
| `requirements.txt` | Python dependencies |
| `PROJECT_CONTEXT.md` | Running working log of decisions, findings, and methodology notes made throughout the project |

## Setup

```bash
pip install -r requirements.txt
```

To regenerate the dataset from scratch (optional — the CSVs are already included):

```bash
python generate_data.py
```

To run the main analysis notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
```

This takes a few minutes — the 227k-row groupbys, XGBoost cross-validation, and SHAP explainability are the slow steps, not a hang.

### Running the Elasticsearch component

Requires Docker.

```bash
docker run -d --name npontu-es -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.0

jupyter nbconvert --to notebook --execute --inplace elasticsearch_demo.ipynb
```

`analysis.ipynb` must be run first, since it produces `customer_features_scored.csv`.

## Methodology Summary

1. **Data generation** — four relationally-linked tables built with realistic distributions (Gamma/Beta/lognormal/Zipf as appropriate) rather than uniform randomness, with a deliberately planted churn-risk behavioral pattern: a subset of customers keep browsing heavily but stop converting specifically in the most recent period.

2. **EDA & cleaning** — structure, missingness, duplicate, and referential-integrity checks across all four tables before any modeling.

3. **Churn label construction** — churn = no purchase in the 90 days before the analysis date. Built with a strict `FEATURE_CUTOFF` boundary separating the data used for features from the data used for the label, after an early version was caught leaking (see below).

4. **Feature engineering** — seven feature groups (RFM, funnel/conversion rates, recency/momentum, temporal, demographic, device/category affinity, cross-table ratios), validated against the label before being trusted.

5. **Modeling** — Logistic Regression baseline and XGBoost, compared via 5-fold cross-validation (ROC-AUC, PR-AUC), with SHAP explainability.

6. **Big data tool** — Elasticsearch, chosen because the actual need is fast aggregation over a growing 227k+ row behavioral log, not event streaming (Kafka) or infrastructure monitoring (Grafana). One aggregation is cross-checked against the equivalent pandas computation as a rigor check — both agree exactly.

## A Note on Rigor: Two Rounds of Leakage, Caught and Fixed

The most important methodological finding in this project came from validating results rather than trusting them:

- **Round 1:** an early version of the model scored a suspicious ROC-AUC of 1.000. Investigation showed `days_since_last_purchase` had been computed relative to the same date used to define the churn label — the feature *was* the label, algebraically. Fixed by introducing a `FEATURE_CUTOFF` boundary so features and label are built from calendar-disjoint windows.

- **Round 2:** once fixed, the recency/momentum features specifically engineered to catch the planted churn pattern turned out *not* to be the strongest predictors — because the planted behavior change lives inside the exact 90-day window reserved for the label, invisible to any properly leak-free feature set. This is reported directly in `analysis.ipynb` (sections 5, 6.1, 7, 9) rather than hidden, since a false "our engineered features definitely helped" claim would be a worse analytical failure than an honest negative result.

Final model: XGBoost, 5-fold CV ROC-AUC ≈ 0.68, PR-AUC ≈ 0.70, on a leak-free, properly validated feature set.

## Key Business Insights

- Plain purchase recency and frequency, correctly computed with a clean prediction cutoff, are the most reliable churn signals available in this dataset.
- 210 currently-active customers are flagged as high churn risk, representing GHS 51,346.91 in combined historical spend — the priority segment for retention campaigns.
- Regional and product-category churn-risk breakdowns (see `analysis.ipynb` section 8) indicate where retention efforts should be targeted first, rather than a one-size-fits-all national campaign.
