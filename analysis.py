"""
Real-World Data Project — Retail Sales Analysis & Forecasting
================================================================
Domain: Retail
Goal: End-to-end EDA + visualization + predictive modeling on sales data.

Run: python3 analysis.py
Outputs: charts saved to images/, metrics printed to console
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv('data/retail_sales.csv', parse_dates=['order_date'])
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(df.dtypes)
print(df.describe())

# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

df['month'] = df['order_date'].dt.month
df['year'] = df['order_date'].dt.year
df['weekday'] = df['order_date'].dt.day_name()
df['year_month'] = df['order_date'].dt.to_period('M').astype(str)

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------

# 3a. Monthly revenue trend
monthly_rev = df.groupby('year_month')['net_amount'].sum().reset_index()
plt.figure()
plt.plot(monthly_rev['year_month'], monthly_rev['net_amount'], marker='o', color='#6a5cff')
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Revenue Trend (2023-2024)')
plt.ylabel('Revenue (₹)')
plt.tight_layout()
plt.savefig('images/monthly_revenue_trend.png', dpi=120)
plt.close()
print("\nSaved: images/monthly_revenue_trend.png")

# 3b. Revenue by category
cat_rev = df.groupby('category')['net_amount'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=cat_rev.values, y=cat_rev.index, palette='viridis')
plt.title('Total Revenue by Category')
plt.xlabel('Revenue (₹)')
plt.tight_layout()
plt.savefig('images/revenue_by_category.png', dpi=120)
plt.close()
print("Saved: images/revenue_by_category.png")

# 3c. Revenue by region
region_rev = df.groupby('region')['net_amount'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=region_rev.index, y=region_rev.values, palette='magma')
plt.title('Total Revenue by Region')
plt.ylabel('Revenue (₹)')
plt.tight_layout()
plt.savefig('images/revenue_by_region.png', dpi=120)
plt.close()
print("Saved: images/revenue_by_region.png")

# 3d. Payment mode distribution
plt.figure()
df['payment_mode'].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('Set2'))
plt.title('Orders by Payment Mode')
plt.ylabel('')
plt.tight_layout()
plt.savefig('images/payment_mode_distribution.png', dpi=120)
plt.close()
print("Saved: images/payment_mode_distribution.png")

# 3e. Discount impact on order value
plt.figure()
sns.boxplot(x='discount_pct', y='net_amount', data=df, palette='coolwarm')
plt.title('Order Value Distribution by Discount %')
plt.ylabel('Net Amount (₹)')
plt.tight_layout()
plt.savefig('images/discount_vs_amount.png', dpi=120)
plt.close()
print("Saved: images/discount_vs_amount.png")

# 3f. Weekday sales pattern
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_rev = df.groupby('weekday')['net_amount'].sum().reindex(weekday_order)
plt.figure()
sns.barplot(x=weekday_rev.index, y=weekday_rev.values, palette='crest')
plt.title('Revenue by Day of Week')
plt.ylabel('Revenue (₹)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('images/revenue_by_weekday.png', dpi=120)
plt.close()
print("Saved: images/revenue_by_weekday.png")

# ---------------------------------------------------------------
# 4. KEY INSIGHTS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
top_category = cat_rev.idxmax()
top_region = region_rev.idxmax()
total_revenue = df['net_amount'].sum()
avg_order_value = df['net_amount'].mean()
holiday_rev = df[df['month'].isin([11, 12])]['net_amount'].sum()
holiday_share = holiday_rev / total_revenue * 100

print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Average Order Value: ₹{avg_order_value:,.2f}")
print(f"Top-performing category: {top_category} (₹{cat_rev.max():,.2f})")
print(f"Top-performing region: {top_region} (₹{region_rev.max():,.2f})")
print(f"Nov-Dec holiday season share of revenue: {holiday_share:.1f}%")

# ---------------------------------------------------------------
# 5. PREDICTIVE MODEL — Predict net_amount from order features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("PREDICTIVE MODEL: Estimating Order Value")
print("=" * 60)

model_df = df.copy()
le_category = LabelEncoder()
le_region = LabelEncoder()
le_payment = LabelEncoder()

model_df['category_enc'] = le_category.fit_transform(model_df['category'])
model_df['region_enc'] = le_region.fit_transform(model_df['region'])
model_df['payment_enc'] = le_payment.fit_transform(model_df['payment_mode'])

features = ['category_enc', 'region_enc', 'payment_enc', 'unit_price', 'quantity', 'discount_pct', 'month']
X = model_df[features]
y = model_df['net_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=8)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Model: Random Forest Regressor")
print(f"Mean Absolute Error: ₹{mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Feature importance chart
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure()
sns.barplot(x=importances.values, y=importances.index, palette='flare')
plt.title('Feature Importance for Predicting Order Value')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=120)
plt.close()
print("Saved: images/feature_importance.png")

# Predicted vs Actual scatter
plt.figure()
plt.scatter(y_test, preds, alpha=0.4, color='#6a5cff')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Net Amount (₹)')
plt.ylabel('Predicted Net Amount (₹)')
plt.title(f'Predicted vs Actual Order Value (R² = {r2:.3f})')
plt.tight_layout()
plt.savefig('images/predicted_vs_actual.png', dpi=120)
plt.close()
print("Saved: images/predicted_vs_actual.png")

print("\nAnalysis complete. All charts saved to the images/ folder.")
