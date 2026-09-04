# Npontu Assignment - Project Context

## Assignment
Analyzing Customer Behavior for E-commerce Insights (Data Science role assignment)
Applicant: Kingsley Buabeng | Role: Intelligent Systems and Security Officer (Data Science & Analytics)
Timeline: Friday Sep 4 - Monday Sep 7, 2026

## Approach: Parametric Statistical Simulation
Using NumPy/pandas to generate a realistic synthetic dataset with statistical distributions matched
to how each variable behaves in real life (not pure randomness), plus deliberately planted behavioral
patterns (e.g., high browsing + low conversion = churn risk signal) so a supervised classification
model can learn something meaningful later.

## Planned Schema (4 tables, linked by customer_id)
1. **customers** (DONE - see customers.csv) - customer_id (PK), age, gender, location, signup_date, tenure_days
   - age: normal distribution (mean 34, std 10)
   - location: weighted categorical (Accra heaviest)
   - signup_date/tenure: exponential distribution (most customers recent, smaller loyal tail)
   - 3% missing location values (simulates guest checkout gaps)

2. **sessions** (NOT YET BUILT) - session_id (PK), customer_id (FK), session_date, session_duration_min,
   pages_viewed, device_type
   - This is where we plant the churn-risk pattern: high session activity + low purchase conversion

3. **product_interactions** (NOT YET BUILT) - interaction_id (PK), customer_id (FK), product_id,
   product_category, interaction_type (view/cart_add/wishlist), interaction_date

4. **purchases** (NOT YET BUILT) - purchase_id (PK), customer_id (FK), product_id, purchase_amount,
   purchase_date, payment_method

## Full Phased Plan
- Phase 1 (Fri): Data generation (IN PROGRESS) + EDA + cleaning
- Phase 2 (Sat AM): Feature engineering + big data tool selection (leaning Elasticsearch, justify choice)
- Phase 3 (Sat PM-Sun AM): Predictive modeling - CHURN PREDICTION chosen as target (supervised classification)
  - Churn label defined by business rule (e.g., no purchase in last 90 days = churned)
  - Cross-validation, metrics: precision/recall/F1/ROC-AUC
- Phase 4 (Sun PM): Insights + visualizations, business recommendations
- Phase 5 (Sun night-Mon): Documentation, GitHub push, PDF report with repo link, final submission

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
- Big data tool: leaning Elasticsearch (fast aggregation/querying) - final justification pending
- Random seed 42 used throughout for reproducibility

## Files in this folder
- generate_data.py - customer table generation script (run so far)
- customers.csv - generated output (3000 rows)

## Next Immediate Step
Build the sessions table generation code, following the same parametric approach, with the
churn-risk pattern deliberately built in (high activity + low conversion).
