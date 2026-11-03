
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
df = pd.read_csv("RealEstate_Prices.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Clean column names
df.columns = df.columns.str.strip().str.lower()

# 4. Remove duplicate records
df.drop_duplicates(inplace=True)

# 5. Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 6. Convert categorical variables into numerical values
binary_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"yes": 1, "no": 0})

# Encode furnishing status
if "furnishingstatus" in df.columns:
    df = pd.get_dummies(
        df, columns=["furnishingstatus"], dtype=int
    )

# 7. Filter properties with valid prices
df = df[df["price"] > 0]

# 8. Detect and handle outliers using IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["price"] < lower) | (df["price"] > upper)]
print("\nNumber of price outliers:", len(outliers))

df["price"] = df["price"].clip(lower, upper)

# 9. Data aggregation
print("\nAverage Price by Bedrooms:")
print(df.groupby("bedrooms")["price"].mean())

print("\nAverage Price by Bathrooms:")
print(df.groupby("bathrooms")["price"].mean())

# 10. Data visualization
plt.figure(figsize=(7, 4))
sns.histplot(df["price"], kde=True)
plt.title("House Price Distribution")
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs House Price")
plt.show()

plt.figure(figsize=(7, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 11. Save cleaned dataset
df.to_csv("cleaned_real_estate.csv", index=False)

print("\nFinal Dataset Shape:", df.shape)
print("Data Wrangling Completed Successfully!")
