import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 3000
today = datetime(2026, 9, 4)

# ---------- 1. CUSTOMERS TABLE ----------
# All 16 administrative regions of Ghana, each with a handful of real towns/cities,
# weighted roughly by population share (region-level) and by town prominence within
# each region (e.g. regional capital gets the largest share). This is intentionally
# more granular than "just the big 4-5 cities" so the synthetic data reflects a
# nationwide e-commerce customer base rather than an urban-only slice.
GHANA_REGIONS = {
    'Greater Accra': {'weight': 0.21, 'towns': {
        'Accra': 0.40, 'Tema': 0.20, 'Madina': 0.15, 'Ashaiman': 0.10, 'Teshie': 0.08, 'Dansoman': 0.07}},
    'Ashanti': {'weight': 0.20, 'towns': {
        'Kumasi': 0.45, 'Obuasi': 0.15, 'Ejisu': 0.10, 'Konongo': 0.10, 'Mampong': 0.10, 'Bekwai': 0.10}},
    'Eastern': {'weight': 0.10, 'towns': {
        'Koforidua': 0.35, 'Akim Oda': 0.20, 'Nkawkaw': 0.15, 'Akosombo': 0.15, 'Suhum': 0.15}},
    'Central': {'weight': 0.08, 'towns': {
        'Cape Coast': 0.30, 'Kasoa': 0.25, 'Winneba': 0.15, 'Elmina': 0.15, 'Swedru': 0.15}},
    'Western': {'weight': 0.07, 'towns': {
        'Takoradi': 0.35, 'Sekondi': 0.25, 'Tarkwa': 0.20, 'Axim': 0.20}},
    'Volta': {'weight': 0.06, 'towns': {
        'Ho': 0.35, 'Hohoe': 0.20, 'Keta': 0.15, 'Aflao': 0.15, 'Kpando': 0.15}},
    'Northern': {'weight': 0.06, 'towns': {
        'Tamale': 0.50, 'Yendi': 0.25, 'Savelugu': 0.25}},
    'Upper East': {'weight': 0.035, 'towns': {
        'Bolgatanga': 0.45, 'Bawku': 0.30, 'Navrongo': 0.25}},
    'Bono': {'weight': 0.04, 'towns': {
        'Sunyani': 0.50, 'Berekum': 0.30, 'Dormaa Ahenkro': 0.20}},
    'Bono East': {'weight': 0.03, 'towns': {
        'Techiman': 0.55, 'Kintampo': 0.25, 'Atebubu': 0.20}},
    'Western North': {'weight': 0.025, 'towns': {
        'Sefwi Wiawso': 0.40, 'Bibiani': 0.35, 'Enchi': 0.25}},
    'Ahafo': {'weight': 0.02, 'towns': {
        'Goaso': 0.55, 'Kenyasi': 0.25, 'Hwidiem': 0.20}},
    'Upper West': {'weight': 0.02, 'towns': {
        'Wa': 0.55, 'Lawra': 0.25, 'Jirapa': 0.20}},
    'Oti': {'weight': 0.02, 'towns': {
        'Dambai': 0.35, 'Nkwanta': 0.35, 'Jasikan': 0.30}},
    'Savannah': {'weight': 0.015, 'towns': {
        'Damongo': 0.55, 'Bole': 0.45}},
    'North East': {'weight': 0.015, 'towns': {
        'Nalerigu': 0.50, 'Walewale': 0.50}},
}
region_names = list(GHANA_REGIONS.keys())
region_weights = np.array([GHANA_REGIONS[r]['weight'] for r in region_names])
region_weights = region_weights / region_weights.sum()  # guard against float rounding

genders = ['Male', 'Female', 'Other']

customer_ids = [f"CUST_{i:05d}" for i in range(1, N_CUSTOMERS + 1)]
signup_dates = [today - timedelta(days=int(np.random.exponential(scale=400))) for _ in range(N_CUSTOMERS)]
# Cap tenure at ~3.7 years so the earliest signups land in Jan 2023, giving the whole
# dataset (signups, sessions, interactions, purchases all derive their dates from this)
# genuine 2023-2026 spread rather than clustering in the most recent year.
signup_dates = [max(d, today - timedelta(days=1340)) for d in signup_dates]

sampled_regions = np.random.choice(region_names, N_CUSTOMERS, p=region_weights)
sampled_towns = [
    np.random.choice(list(GHANA_REGIONS[r]['towns'].keys()), p=list(GHANA_REGIONS[r]['towns'].values()))
    for r in sampled_regions
]

customers = pd.DataFrame({
    'customer_id': customer_ids,
    'age': np.clip(np.random.normal(34, 10, N_CUSTOMERS).astype(int), 18, 70),
    'gender': np.random.choice(genders, N_CUSTOMERS, p=[0.48, 0.48, 0.04]),
    'region': sampled_regions,
    'city': sampled_towns,
    'signup_date': signup_dates,
})
customers['tenure_days'] = (today - customers['signup_date']).dt.days

# introduce some missing values deliberately (realistic: guest signups missing location).
# Both region and city are blanked together since a guest checkout wouldn't partially
# know one but not the other.
missing_idx = customers.sample(frac=0.03, random_state=1).index
customers.loc[missing_idx, ['region', 'city']] = np.nan

print("Customers table:", customers.shape)
customers.to_csv('customers.csv', index=False)
customers.head()

# ---------- Hidden per-customer engagement traits (NOT exported) ----------
# These latent traits are what actually generate the churn-risk pattern: they drive
# observable session behavior now, and will drive purchase behavior once the
# `purchases` table is built later, so the correlation between "browses a lot" and
# "rarely buys" falls out naturally when the tables are joined - it isn't a label
# baked directly into any one table. Computed once, right after the customers table,
# so re-running this whole script top-to-bottom (fixed seed) always reproduces the
# same traits for the same customer_id, keeping later tables consistent with this one.
browsing_intensity = np.random.gamma(shape=2.0, scale=1.0, size=N_CUSTOMERS)  # heavy tail -> some power browsers

# Conversion propensity is deliberately anti-correlated with browsing intensity for a
# "high browse, rarely buys" churn-risk segment (~42% of the top browsers), while a
# smaller "power buyer" segment among the remaining top browsers converts even better
# than average. Everyone else gets an ordinary, unrelated conversion propensity.
base_conversion = np.random.beta(2, 5, N_CUSTOMERS)  # skewed low - most people don't convert often
top_browser = browsing_intensity > np.percentile(browsing_intensity, 70)
churn_risk_mask = top_browser & (np.random.rand(N_CUSTOMERS) < 0.6)
power_buyer_mask = top_browser & ~churn_risk_mask & (np.random.rand(N_CUSTOMERS) < 0.5)
conversion_propensity = np.select(
    [churn_risk_mask, power_buyer_mask],
    [base_conversion * 0.25, np.clip(base_conversion * 2.5, 0, 1)],
    default=base_conversion,
)

# ---------- 2. SESSIONS TABLE ----------
# Number of sessions scales with browsing intensity and with how long a customer has
# had an account to accumulate activity (capped so very old accounts don't dominate).
avg_sessions = (2 + browsing_intensity * 4) * np.clip(customers['tenure_days'].values / 200, 0.2, 3)
n_sessions = np.clip(np.random.poisson(avg_sessions), 1, None)  # every customer has >=1 session
total_sessions = int(n_sessions.sum())

# Roughly two-thirds mobile, consistent with a mobile-first Ghanaian e-commerce market;
# each customer has a dominant device but occasionally shows up on another one.
device_types = ['Mobile', 'Desktop', 'Tablet']
dominant_device = np.random.choice(device_types, N_CUSTOMERS, p=[0.65, 0.28, 0.07])

# Expand customer-level arrays to one row per session
cust_id_exp = np.repeat(customers['customer_id'].values, n_sessions)
signup_exp = np.repeat(customers['signup_date'].values, n_sessions)
tenure_exp = np.repeat(customers['tenure_days'].values, n_sessions)
browsing_exp = np.repeat(browsing_intensity, n_sessions)
dominant_device_exp = np.repeat(dominant_device, n_sessions)

# Session date: somewhere between signup and today, skewed toward more recent activity
recency_frac = np.random.beta(2, 1, total_sessions)
offset_days = (recency_frac * tenure_exp).astype(int)
session_date = pd.to_datetime(signup_exp) + pd.to_timedelta(offset_days, unit='D')

# Duration and pages viewed both scale with the customer's browsing intensity
session_duration_min = np.round(np.random.gamma(shape=2.0, scale=3 + browsing_exp * 4), 1)
session_duration_min = np.clip(session_duration_min, 0.5, 120)
pages_viewed = np.clip(np.random.poisson(2 + session_duration_min / 3 + browsing_exp * 1.5), 1, None)

# 85% of the time a session happens on the customer's usual device, otherwise another one
uses_dominant = np.random.rand(total_sessions) < 0.85
device_type = np.where(
    uses_dominant, dominant_device_exp,
    np.random.choice(device_types, total_sessions, p=[0.65, 0.28, 0.07]),
)

sessions = pd.DataFrame({
    'session_id': np.arange(1, total_sessions + 1),
    'customer_id': cust_id_exp,
    'session_date': session_date,
    'session_duration_min': session_duration_min,
    'pages_viewed': pages_viewed,
    'device_type': device_type,
}).sort_values(['customer_id', 'session_date']).reset_index(drop=True)
sessions['session_id'] = [f"SESS_{i:06d}" for i in range(1, len(sessions) + 1)]
sessions = sessions[['session_id', 'customer_id', 'session_date', 'session_duration_min', 'pages_viewed', 'device_type']]

print("Sessions table:", sessions.shape)
sessions.to_csv('sessions.csv', index=False)
sessions.head()

# ---------- Product catalog (kept in-script, reused by purchases later) ----------
# Category shares roughly reflect a Jumia-style Ghanaian e-commerce mix (fashion and
# phones dominate). Within each category, popularity follows a Zipf-like curve so a
# handful of products account for a disproportionate share of interactions - realistic
# "hit products" long-tail behavior rather than every product being equally likely.
CATEGORY_WEIGHTS = {
    'Fashion & Apparel': 0.22,
    'Phones & Tablets': 0.18,
    'Electronics': 0.15,
    'Home & Kitchen': 0.12,
    'Beauty & Personal Care': 0.10,
    'Health & Wellness': 0.07,
    'Sports & Outdoors': 0.06,
    'Baby & Kids': 0.05,
    'Books & Stationery': 0.03,
    'Groceries & Food': 0.02,
}
N_PRODUCTS = 300
category_names = list(CATEGORY_WEIGHTS.keys())
products_per_category = np.maximum(
    (np.array(list(CATEGORY_WEIGHTS.values())) * N_PRODUCTS).round().astype(int), 8
)

product_ids, product_categories, product_popularity = [], [], []
pid_counter = 1
for cat, n_prod in zip(category_names, products_per_category):
    ranks = np.arange(1, n_prod + 1)
    zipf_weights = (1 / ranks ** 0.8)
    zipf_weights = zipf_weights / zipf_weights.sum() * CATEGORY_WEIGHTS[cat]  # keep category's overall share
    for _ in range(n_prod):
        product_ids.append(f"PROD_{pid_counter:04d}")
        pid_counter += 1
    product_categories.extend([cat] * n_prod)
    product_popularity.extend(zipf_weights.tolist())

product_popularity = np.array(product_popularity)
product_popularity = product_popularity / product_popularity.sum()
product_category_map = dict(zip(product_ids, product_categories))

# Realistic, category-appropriate item names (Ghanaian e-commerce flavor) so interactions
# and purchases show what was actually browsed/bought, not just an abstract product_id.
# Several products within a category share a base name with different colors/sizes/models
# as distinct SKUs - normal for a real catalog - so names aren't required to be unique.
PRODUCT_NAMES = {
    'Fashion & Apparel': [
        'Ankara Print Dress', "Men's Slim Fit Jeans", 'Ladies Kaftan Gown', 'Cotton Polo Shirt',
        'Kente Print Shirt', "Women's Maxi Dress", "Men's Casual Blazer", 'Denim Jacket',
        'Ladies Handbag', 'Leather Sandals', 'Unisex Sneakers', 'Wax Print Skirt',
        "Men's Formal Suit", "Children's T-Shirt Set", 'Traditional Smock (Fugu)',
    ],
    'Phones & Tablets': [
        'Samsung Galaxy A14', 'Tecno Spark 10', 'Infinix Hot 30', 'iPhone 13', 'Itel Vision 3',
        'Samsung Galaxy Tab A8', 'Redmi Note 12', 'Huawei MatePad', 'Phone Charger (Type-C)',
        'Wireless Earbuds', 'Phone Screen Protector', 'Power Bank 20000mAh', 'Bluetooth Headset',
        'Universal Phone Case',
    ],
    'Electronics': [
        '43-inch LED TV', 'Bluetooth Speaker', 'Home Theater System', 'Rechargeable Fan',
        'Electric Iron', 'Microwave Oven', 'Blender (4L)', 'Laptop (Core i5)', 'Desktop Computer',
        'Air Conditioner (1.5HP)', 'Extension Socket', 'LED Bulb Pack', 'Digital Camera',
        'Gaming Console',
    ],
    'Home & Kitchen': [
        'Non-stick Cooking Pot Set', 'Bedsheet Set (Queen)', 'Kitchen Knife Set',
        'Storage Containers Set', 'Curtain Set', 'Dining Table Set', 'Mattress (Double)',
        'Rice Cooker', 'Gas Cooker (4-Burner)', 'Water Dispenser', 'Wall Clock',
        'Bathroom Towel Set', 'Sofa Set', 'Cutlery Set',
    ],
    'Beauty & Personal Care': [
        'Shea Butter Body Lotion', 'Black Soap Bar', 'Hair Relaxer Kit', 'Perfume (100ml)',
        'Facial Cleanser', 'Braided Hair Extension', 'Nail Polish Set', 'Makeup Kit',
        'Deodorant Roll-on', 'Sunscreen SPF50', 'Hair Dryer', 'Skin Toner', 'Lip Gloss Set',
        'Beard Grooming Kit',
    ],
    'Health & Wellness': [
        'Multivitamin Tablets', 'Digital Blood Pressure Monitor', 'First Aid Kit',
        'Hand Sanitizer (500ml)', 'Protein Powder', 'Face Mask Pack (50pcs)',
        'Digital Thermometer', 'Bathroom Weighing Scale', 'Vitamin C Supplements',
        'Herbal Tea Pack', 'Glucometer Kit', 'Reading Glasses',
    ],
    'Sports & Outdoors': [
        'Football (Size 5)', 'Yoga Mat', 'Running Shoes', 'Dumbbell Set', 'Camping Tent',
        'Cycling Helmet', 'Skipping Rope', 'Gym Gloves', 'Basketball', 'Hiking Backpack',
        'Water Bottle (1L)', 'Resistance Bands',
    ],
    'Baby & Kids': [
        'Baby Diapers Pack', 'Baby Stroller', 'Feeding Bottle Set', 'Kids Building Blocks',
        'Baby Carrier', 'Kids School Backpack', 'Baby Wipes (80pcs)', 'Toddler Shoes',
        'Baby Bath Tub', "Kids' Bicycle", 'Baby Monitor', 'Educational Toy Set',
    ],
    'Books & Stationery': [
        'Exercise Books Pack', 'Fiction Novel', 'Ballpoint Pens Pack', 'A4 Printing Paper Ream',
        'Mathematics Textbook', 'Hardcover Notebook', 'Highlighter Set', 'School Backpack',
        'Scientific Calculator', "Kids' Storybook",
    ],
    'Groceries & Food': [
        'Rice (5kg Bag)', 'Cooking Oil (3L)', 'Tomato Paste Carton', 'Milk Powder (400g)',
        'Sugar (2kg)', 'Spaghetti Pack', 'Canned Tuna', 'Bottled Water (Pack of 12)',
        'Local Spices Set', 'Garri (2kg)', 'Plantain Chips', 'Instant Noodles Pack',
    ],
}
product_names = [np.random.choice(PRODUCT_NAMES[cat]) for cat in product_categories]
product_name_map = dict(zip(product_ids, product_names))

# ---------- 3. PRODUCT_INTERACTIONS TABLE ----------
# Interaction volume per customer scales with how many pages they actually browsed in
# sessions.csv - a heavier browser generates more product-level view/cart/wishlist events.
pages_per_customer = sessions.groupby('customer_id')['pages_viewed'].sum() \
    .reindex(customers['customer_id']).fillna(0).values
n_interactions = np.clip(np.random.poisson(0.3 * pages_per_customer), 1, None)
total_interactions = int(n_interactions.sum())

cust_id_exp2 = np.repeat(customers['customer_id'].values, n_interactions)
signup_exp2 = np.repeat(customers['signup_date'].values, n_interactions)
tenure_exp2 = np.repeat(customers['tenure_days'].values, n_interactions)
conversion_exp2 = np.repeat(conversion_propensity, n_interactions)

recency_frac2 = np.random.beta(2, 1, total_interactions)
offset_days2 = (recency_frac2 * tenure_exp2).astype(int)
interaction_date = pd.to_datetime(signup_exp2) + pd.to_timedelta(offset_days2, unit='D')

products_sampled = np.random.choice(product_ids, size=total_interactions, p=product_popularity)
categories_sampled = pd.Series(products_sampled).map(product_category_map).values
names_sampled = pd.Series(products_sampled).map(product_name_map).values

# cart_add rate rises with the customer's hidden conversion propensity (the churn-risk
# segment rarely progresses past viewing; the power-buyer segment adds to cart often),
# wishlist stays a small constant slice, and view soaks up the remainder.
p_cart_add = np.clip(0.10 + 0.30 * conversion_exp2, 0, 0.5)
p_wishlist = np.full(total_interactions, 0.08)
u = np.random.rand(total_interactions)
interaction_type = np.where(
    u < p_wishlist, 'wishlist',
    np.where(u < p_wishlist + p_cart_add, 'cart_add', 'view'),
)

product_interactions = pd.DataFrame({
    'interaction_id': np.arange(1, total_interactions + 1),
    'customer_id': cust_id_exp2,
    'product_id': products_sampled,
    'product_name': names_sampled,
    'product_category': categories_sampled,
    'interaction_type': interaction_type,
    'interaction_date': interaction_date,
}).sort_values(['customer_id', 'interaction_date']).reset_index(drop=True)
product_interactions['interaction_id'] = [f"INT_{i:07d}" for i in range(1, len(product_interactions) + 1)]
product_interactions = product_interactions[
    ['interaction_id', 'customer_id', 'product_id', 'product_name', 'product_category', 'interaction_type', 'interaction_date']
]

print("Product interactions table:", product_interactions.shape)
product_interactions.to_csv('product_interactions.csv', index=False)
product_interactions.head()

# ---------- 4. PURCHASES TABLE ----------
# Purchases are drawn from the cart_add events above - not every cart add converts
# (industry-typical cart abandonment), and the conversion chance is further shaped by
# the same hidden `conversion_propensity` trait used throughout. This is the step that
# finally makes the churn-risk pattern (heavy browsing, rarely buying) observable: it
# only exists once this table is joined against sessions/product_interactions.
cart_adds = product_interactions[product_interactions['interaction_type'] == 'cart_add'].copy()
propensity_map = dict(zip(customers['customer_id'], conversion_propensity))
churn_risk_map = dict(zip(customers['customer_id'], churn_risk_mask))
cart_adds['conversion_propensity'] = cart_adds['customer_id'].map(propensity_map)
cart_adds['is_churn_risk'] = cart_adds['customer_id'].map(churn_risk_map)

base_purchase_prob = np.clip(0.10 + 0.55 * cart_adds['conversion_propensity'].values, 0, 0.9)

# Churn-risk customers browse and cart-add MORE in absolute terms than everyone else
# (their sheer activity volume outweighs their lower per-interaction conversion rate),
# so without this step they'd end up with *more* lifetime purchases than average and
# would never trip a "no purchase in 90 days" rule - the opposite of the intended
# signal. What actually makes them churn-risk is that they've largely stopped
# completing purchases recently even though they're still visibly active: cart-adds
# from the last 90 days convert at a fraction of their (already low) base rate, while
# older cart-adds convert normally. This is the "still visits, stopped buying" pattern
# a churn model should learn to catch from recency + activity features, not raw counts.
days_since_cart_add = (pd.Timestamp(today) - pd.to_datetime(cart_adds['interaction_date'])).dt.days.values
recent_dropoff = cart_adds['is_churn_risk'].values & (days_since_cart_add <= 90)
purchase_prob = np.where(recent_dropoff, base_purchase_prob * 0.12, base_purchase_prob)

converts = np.random.rand(len(cart_adds)) < purchase_prob
purchases_base = cart_adds.loc[converts].reset_index(drop=True)

# Purchase happens a few days after the cart-add, capped at "today"
delay_days = np.random.gamma(shape=1.5, scale=2.0, size=len(purchases_base)).astype(int)
purchase_date = pd.to_datetime(purchases_base['interaction_date']) + pd.to_timedelta(delay_days, unit='D')
purchase_date = purchase_date.clip(upper=pd.Timestamp(today))

# Category-level price bands in GHS (median, lognormal sigma) - wider spread for
# Phones & Tablets/Electronics since those categories span budget to premium items
CATEGORY_PRICE = {
    'Fashion & Apparel': (120, 0.6), 'Phones & Tablets': (900, 0.8), 'Electronics': (600, 0.8),
    'Home & Kitchen': (200, 0.6), 'Beauty & Personal Care': (60, 0.5), 'Health & Wellness': (70, 0.5),
    'Sports & Outdoors': (150, 0.6), 'Baby & Kids': (100, 0.5), 'Books & Stationery': (40, 0.4),
    'Groceries & Food': (80, 0.5),
}
medians = purchases_base['product_category'].map({k: v[0] for k, v in CATEGORY_PRICE.items()}).values
sigmas = purchases_base['product_category'].map({k: v[1] for k, v in CATEGORY_PRICE.items()}).values
purchase_amount = np.round(np.random.lognormal(mean=np.log(medians), sigma=sigmas), 2)
purchase_amount = np.clip(purchase_amount, 5, 15000)

# Mobile Money dominates in the Ghanaian market, followed by cash-on-delivery, then card/bank
payment_methods = ['Mobile Money', 'Cash on Delivery', 'Card', 'Bank Transfer']
payment_method = np.random.choice(payment_methods, len(purchases_base), p=[0.55, 0.25, 0.15, 0.05])

purchases = pd.DataFrame({
    'purchase_id': np.arange(1, len(purchases_base) + 1),
    'customer_id': purchases_base['customer_id'].values,
    'product_id': purchases_base['product_id'].values,
    'product_name': purchases_base['product_name'].values,
    'purchase_amount': purchase_amount,
    'purchase_date': purchase_date.values,
    'payment_method': payment_method,
}).sort_values(['customer_id', 'purchase_date']).reset_index(drop=True)
purchases['purchase_id'] = [f"PUR_{i:06d}" for i in range(1, len(purchases) + 1)]
purchases = purchases[
    ['purchase_id', 'customer_id', 'product_id', 'product_name', 'purchase_amount', 'purchase_date', 'payment_method']
]

print("Purchases table:", purchases.shape)
purchases.to_csv('purchases.csv', index=False)
purchases.head()

# ---------- Combined workbook ----------
# One .xlsx with all 4 tables as separate sheets, for easy review alongside the
# individual CSVs (which remain the notebook-friendly source of truth for analysis).
with pd.ExcelWriter('ecommerce_dataset.xlsx', engine='openpyxl') as writer:
    customers.to_excel(writer, sheet_name='Customers', index=False)
    sessions.to_excel(writer, sheet_name='Sessions', index=False)
    product_interactions.to_excel(writer, sheet_name='Product_Interactions', index=False)
    purchases.to_excel(writer, sheet_name='Purchases', index=False)

print("Combined workbook written: ecommerce_dataset.xlsx")
