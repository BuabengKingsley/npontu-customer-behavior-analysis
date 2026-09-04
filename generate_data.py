import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 3000
today = datetime(2026, 9, 4)

# ---------- 1. CUSTOMERS TABLE ----------
regions = ['Accra', 'Kumasi', 'Takoradi', 'Tamale', 'Cape Coast', 'Ho']
genders = ['Male', 'Female', 'Other']

customer_ids = [f"CUST_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
signup_dates = [today - timedelta(days=int(np.random.exponential(scale=400))) for _ in range(N_CUSTOMERS)]
signup_dates = [max(d, today - timedelta(days=1000)) for d in signup_dates]  # cap tenure at ~2.7 years

customers = pd.DataFrame({
    'customer_id': customer_ids,
    'age': np.clip(np.random.normal(34, 10, N_CUSTOMERS).astype(int), 18, 70),
    'gender': np.random.choice(genders, N_CUSTOMERS, p=[0.48, 0.48, 0.04]),
    'location': np.random.choice(regions, N_CUSTOMERS, p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]),
    'signup_date': signup_dates,
})
customers['tenure_days'] = (today - customers['signup_date']).dt.days

# introduce some missing values deliberately (realistic: guest signups missing location)
missing_idx = customers.sample(frac=0.03, random_state=1).index
customers.loc[missing_idx, 'location'] = np.nan

print("Customers table:", customers.shape)
customers.to_csv('customers.csv', index=False)
customers.head()
