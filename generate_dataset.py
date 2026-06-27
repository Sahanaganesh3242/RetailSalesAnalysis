"""
Generates a realistic synthetic retail sales dataset for analysis.
Run once to create data/retail_sales.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

categories = ['Electronics', 'Clothing', 'Groceries', 'Furniture', 'Toys', 'Beauty']
regions = ['North', 'South', 'East', 'West']
payment_modes = ['Credit Card', 'Debit Card', 'UPI', 'Cash']

n_rows = 3000
date_range = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')

rows = []
order_id = 1000

for _ in range(n_rows):
    date = np.random.choice(date_range)
    category = np.random.choice(categories, p=[0.2, 0.2, 0.25, 0.1, 0.1, 0.15])
    region = np.random.choice(regions)
    payment = np.random.choice(payment_modes, p=[0.35, 0.25, 0.3, 0.1])

    base_price = {
        'Electronics': 250, 'Clothing': 40, 'Groceries': 15,
        'Furniture': 300, 'Toys': 25, 'Beauty': 30
    }[category]

    # seasonal boost in Nov-Dec (holiday season)
    month = pd.Timestamp(date).month
    seasonal_factor = 1.4 if month in [11, 12] else 1.0

    unit_price = round(base_price * np.random.uniform(0.7, 1.3), 2)
    quantity = np.random.randint(1, 6)
    discount_pct = np.random.choice([0, 5, 10, 15, 20], p=[0.4, 0.2, 0.2, 0.1, 0.1])

    gross_amount = unit_price * quantity * seasonal_factor
    discount_amount = gross_amount * (discount_pct / 100)
    net_amount = round(gross_amount - discount_amount, 2)

    rows.append({
        'order_id': order_id,
        'order_date': pd.Timestamp(date).strftime('%Y-%m-%d'),
        'category': category,
        'region': region,
        'payment_mode': payment,
        'unit_price': unit_price,
        'quantity': quantity,
        'discount_pct': discount_pct,
        'net_amount': net_amount
    })
    order_id += 1

df = pd.DataFrame(rows)
df.sort_values('order_date', inplace=True)
df.to_csv('data/retail_sales.csv', index=False)
print(f"Generated {len(df)} rows -> data/retail_sales.csv")
print(df.head())
