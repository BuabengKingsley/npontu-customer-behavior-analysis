# Npontu Assignment - Project Context

## Assignment
Analyzing Customer Behavior for E-commerce Insights (Npontu Technologies interview assignment)
Applicant: Kingsley Buabeng | Role: Intelligent Systems Services Engineer
Timeline: Friday Sep 4 - Monday Sep 7, 2026 (official brief: submit 3 days after job application)

## Official Requirements (from assignment sheet)
1. EDA + cleaning (missing values, erroneous entries, duplicates)
2. Feature engineering (time-based, demographic, purchase-pattern features) + normalize/standardize
3. At least one predictive model (churn / recommendations / sales forecasting) + cross-validation + metrics
4. Big data tool (Kafka/Grafana/Elasticsearch or similar) to manage/process data, with justification
5. Actionable business insights + visualizations
6. Documented report/presentation, well-commented notebook code
Evaluated on: analytical rigor, feature engineering creativity, tool utilization relevance,
modeling proficiency, insight quality, presentation/documentation clarity.

## Approach: Parametric Statistical Simulation
Using NumPy/pandas to generate a realistic synthetic dataset with statistical distributions matched
to how each variable behaves in real life (not pure randomness), plus deliberately planted behavioral
patterns (e.g., high browsing + low conversion = churn risk signal) so a supervised classification
model can learn something meaningful later.

## Planned Schema (4 tables, linked by customer_id)
1. **customers** (DONE - see customers.csv) - customer_id (PK), age, gender, region, city, signup_date, tenure_days
   - age: normal distribution (mean 34, std 10)
   - region: all 16 Ghana regions, weighted by approximate population share (Greater Accra/Ashanti heaviest)
   - city: 59 real towns across those 16 regions, weighted within each region (regional capital heaviest)
   - signup_date/tenure: exponential distribution (most customers recent, smaller loyal tail),
     capped at ~3.7 years so the earliest signups land in Jan 2023 - dataset spans 2023-2026
   - 3% missing region+city values together (simulates guest checkout gaps)

2. **sessions** (DONE - see sessions.csv) - session_id (PK), customer_id (FK), session_date,
   session_duration_min, pages_viewed, device_type (48,222 rows, 1-110 sessions/customer)
   - session count, duration, pages_viewed all scale with a hidden per-customer `browsing_intensity`
     trait (not exported) - correlation between browsing_intensity and n_sessions is 0.57
   - device_type: ~65/28/7% Mobile/Desktop/Tablet, each customer has a dominant device (85% of
     their sessions) with occasional cross-device use
   - churn-risk pattern is planted via a hidden `conversion_propensity` trait computed alongside
     `browsing_intensity`, deliberately anti-correlated for a "high browse, rarely buys" segment
     (~18% of customers, mean conversion 0.07) vs a "power buyer" segment that also browses a lot
     but converts well (~6% of customers, mean conversion 0.66) vs everyone else (mean 0.29).
     Neither trait is written to any CSV - they'll drive purchase frequency/amount when the
     `purchases` table is built, so the churn signal emerges from joining tables, not a leaked label.

3. **product_interactions** (DONE - see product_interactions.csv) - interaction_id (PK), customer_id (FK),
   product_id, product_name, product_category, interaction_type (view/cart_add/wishlist),
   interaction_date (227,248 rows)
   - product catalog: 302 products across 10 categories (Fashion & Apparel/Phones & Tablets heaviest,
     matching a Jumia-style Ghanaian market mix), Zipf-distributed popularity within each category
     so a handful of "hit" products dominate interaction volume
   - product_name: real-sounding, category-appropriate item names (e.g. "Kente Print Shirt",
     "Samsung Galaxy A14", "Rice (5kg Bag)") mapped from a curated per-category name list so
     interactions/purchases show what was actually browsed/bought, not just an abstract product_id
   - interaction volume per customer scales with total pages_viewed from sessions.csv
   - interaction_type mix ~75% view / 17% cart_add / 8% wishlist overall, but cart_add rate is
     modulated by the hidden `conversion_propensity` trait: churn-risk segment carts at ~12%,
     baseline ~19%, power-buyer segment ~30% - a real funnel-level signal, not just raw activity

4. **purchases** (DONE - see purchases.csv) - purchase_id (PK), customer_id (FK), product_id,
   product_name, purchase_amount, purchase_date, payment_method (10,614 rows, drawn from cart_add events)
   - purchase_amount: category-specific lognormal price bands in GHS (Fashion ~120 median,
     Phones & Tablets ~900, Electronics ~600, down to Books & Stationery ~40), capped at 15,000
   - payment_method: Mobile Money 55% / Cash on Delivery 25% / Card 14% / Bank Transfer 5%
   - IMPORTANT TUNING NOTE: churn-risk customers actually generate ~2x MORE cart_adds than
     baseline in absolute terms (their sheer browsing volume outweighs their lower per-interaction
     rate), so a naive purchase-probability model gave them *more* lifetime purchases than
     baseline and a LOWER 90-day churn rate - the opposite of the intended pattern. Fixed by
     making their cart_add-to-purchase conversion crater specifically for RECENT (last 90 days)
     cart_adds (x0.12 multiplier), representing "still browsing, stopped buying recently" -
     this is what makes the churn label detectable from recency/rate features, not raw activity.

## Verified Churn Pattern (as of 2026-09-06, full pipeline run incl. product names + 2023-2026 dates)
Using churn = no purchase in last 90 days:
- Churn-risk segment (~18% of customers): 85.2% churned
- Baseline segment (~76% of customers): 56.3% churned
- Power-buyer segment (~6% of customers): 4.7% churned
- Overall churn rate: 58.6% (workable ~59/41 split for supervised classification)
Key modeling implication: raw purchase COUNT does not separate churn-risk from baseline
(2.15 vs 2.26) - the signal is in RECENCY and in rate/ratio features (cart_add rate, sessions
per purchase, days since last purchase vs days since last session). Feature engineering must
compute these ratios explicitly; a model fed only raw counts will miss the pattern entirely.
This is deliberate and should be called out in the report as a feature engineering insight.

## Full Phased Plan
- Phase 1 (Fri): Data generation DONE + EDA + cleaning DONE (analysis.ipynb sections 1-3)
- Phase 2 (Sat AM): Feature engineering DONE (analysis.ipynb section 4, 7 feature groups) +
  big data tool DONE (elasticsearch_demo.ipynb, justification below)
- Phase 3 (Sat PM-Sun AM): Predictive modeling DONE - CHURN PREDICTION as target (supervised classification)
  - Churn label = no purchase in last 90 days (business rule), built leak-free (see below)
  - Logistic Regression baseline + XGBoost, 5-fold CV, ROC-AUC/PR-AUC/classification report/confusion matrix
- Phase 4 (Sun PM): Insights + visualizations DONE (analysis.ipynb sections 5, 7, 8) - business
  recommendations DONE (section 9)
- Phase 5 (Sun night-Mon): NEXT - written report/presentation, GitHub push, PDF with repo link, final submission

## CRITICAL FINDING: Two Rounds of Leakage Discovery During Modeling (2026-09-06)
This is the single most important thing to carry into the written report - it's the strongest
"analytical rigor" evidence in the whole submission, and it changes the headline result from
"we found the perfect churn predictor" (a red flag, not an achievement) to an honest, moderate,
well-validated one.

**Round 1 - hard leakage (in analysis.ipynb section 3.1):** The first version computed
`days_since_last_purchase` relative to `AS_OF`, and the churn label was ALSO
`days_since_last_purchase > 90` relative to `AS_OF` - i.e. the feature WAS the label,
algebraically. Symptom: naive Logistic Regression scored a suspicious ROC-AUC of 1.000.
**Fix:** introduced `FEATURE_CUTOFF = AS_OF - 90 days`. Every feature uses only data
`<= FEATURE_CUTOFF`; the label uses only purchase activity in `(FEATURE_CUTOFF, AS_OF]`.
Also required excluding customers who signed up after FEATURE_CUTOFF (tenure_days < 90, n=627)
since they have zero pre-cutoff history to build features from at all - not just the ones with
no purchase yet, as an earlier looser version of the exclusion rule assumed.

**Round 2 - a subtler, more interesting finding after the fix:** Group C features
(`conversion_rate_delta`, `activity_momentum`, `browse_buy_gap_days`, `stale_cart_adds`) were
built specifically to catch the planted "high browse, recently stopped buying" pattern (see the
purchases-table tuning note below: the generator collapses cart-add-to-purchase conversion
specifically for cart-adds in the LAST 90 DAYS before AS_OF). Once features were correctly
restricted to `<= FEATURE_CUTOFF`, that exact planted mechanism became invisible to the
features by construction - it lives entirely inside the label window. Result: Group C features
score weakest in the section-5 correlation table, the naive-vs-full CV comparison (section 6.1)
shows no meaningful lift (ROC-AUC -0.009, PR-AUC +0.006, both within 1 CV std), and none of the
four Group-C signature features make the SHAP top-10 (section 7). Plain pre-cutoff recency/
frequency (`recent_purchases`, `total_purchases`, `total_spend`, `view_to_cart_rate`) do the
real work instead. This is reported directly in the notebook (sections 5, 6.1, 7, 9) as a
genuine methodological insight, not smoothed over - "the fancy features didn't help, and here's
the structural reason why" is a stronger analytical-rigor artifact than a false "they helped."

**Final model performance (post-fix, honest numbers):** XGBoost on the full feature set,
5-fold CV: ROC-AUC ≈ 0.68, PR-AUC ≈ 0.70. Modeling-eligible population: 2,373 of 3,000 customers
(627 excluded for insufficient pre-cutoff history). Churn rate in the modeling set: 55.2%.
Revenue-at-risk (currently-active customers flagged >=0.5 predicted churn probability): 210
customers, GHS 51,346.91 combined historical spend.

## Deliverables Required by Assignment
- Synthetic dataset + generation script
- Jupyter notebook (EDA, cleaning, feature engineering, modeling) - well-commented
- Big data tool component + justification (Kafka/Elasticsearch/Grafana)
- Visualizations
- Written report/presentation
- GitHub repo link inside a PDF document (per Npontu's submission instructions)

## Key Decisions Already Made
- Model type: Supervised learning, classification (churn prediction chosen over recommendations/forecasting)
- Dataset size: 3,000 customers
- Big data tool: Elasticsearch, chosen and justified in elasticsearch_demo.ipynb - the real need
  is fast aggregation over a growing 227k+ row behavioral log, not streaming (Kafka) or infra
  monitoring (Grafana). Verified with a live local Docker instance (see below), not just claimed.
- Random seed 42 used throughout for data-generation reproducibility

## Big Data Tool: Elasticsearch (DONE, elasticsearch_demo.ipynb)
Local single-node Elasticsearch 8.15.0 via Docker (container name `npontu-es`, ports 9200/9300,
security disabled for local demo). Two indices: `customers_scored` (2,373 docs, the final scored
feature table exported from analysis.ipynb as customer_features_scored.csv) and
`product_interactions` (227,248 docs, the full raw interaction log). Three aggregation queries
demonstrated: (1) a recent-vs-historical cart-add count split, cross-checked against the
pandas-computed equivalent in analysis.ipynb - both methods agree exactly (8,534 recent /
15,537 historical); (2) avg predicted churn probability by region (terms+avg agg); (3)
interaction-type mix by product category (nested terms agg). Elasticsearch python client must
be pinned to 8.x (`pip install "elasticsearch>=8.15,<9"`) to match the 8.15.0 server - the
default pip install pulls 9.x, which fails with a version-compatibility header error.
NOTE: Docker Desktop does not survive a machine/session restart - if `docker ps` can't reach
the daemon, relaunch Docker Desktop, then `docker start npontu-es` (container persists, don't
recreate it) and wait ~30-60s for ES to accept connections before re-running the notebook.

## Files in this folder
- generate_data.py - full 4-table generation script (customers, sessions, product_interactions, purchases)
- customers.csv, sessions.csv, product_interactions.csv, purchases.csv - generated data (see schema above)
- ecommerce_dataset.xlsx - all 4 tables combined as separate sheets, for reviewers who prefer one file
- analysis.ipynb - MAIN DELIVERABLE. EDA, cleaning, churn label construction (with the leakage
  fix), 7 feature groups, feature validation, Logistic Regression + XGBoost modeling with CV,
  SHAP explainability, business insights/visualizations, conclusions. Runs end-to-end with
  `jupyter nbconvert --to notebook --execute --inplace analysis.ipynb` (takes a few minutes;
  SHAP + XGBoost + 227k-row groupbys are the slow parts, not a hang).
- customer_features_scored.csv - exported by analysis.ipynb's last code cell (2,373 rows), the
  bridge file elasticsearch_demo.ipynb consumes. NOTE: contains the literal string "None" for
  customers with no favorite_category/dominant_device - read with `keep_default_na=False` or
  pandas will silently convert it back to NaN.
- elasticsearch_demo.ipynb - big data tool component, see above. Requires the `npontu-es` Docker
  container running before executing.

## Next Immediate Step
Phases 1-4 are DONE (data generation, EDA/cleaning, feature engineering, modeling, explainability,
business insights, big data tool - all in analysis.ipynb + elasticsearch_demo.ipynb, both execute
cleanly with zero errors). Move to Phase 5: write the report/presentation (should foreground the
two-round leakage story above as the main analytical-rigor narrative, not bury it), push to
GitHub, and produce the final PDF with the repo link per Npontu's submission instructions.
