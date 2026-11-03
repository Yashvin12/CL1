
# PRACTICAL: DATA LOADING, STORAGE AND FILE FORMATS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load original dataset
df_original = pd.read_csv("sales_data.csv")

print(df_original.head())
print("Shape:", df_original.shape)
print("Columns:", df_original.columns.tolist())

# 2. Divide and save data in three formats
parts = np.array_split(df_original, 3)

parts[0].to_csv("sales_part1.csv", index=False)
parts[1].to_excel("sales_part2.xlsx", index=False)
parts[2].to_json("sales_part3.json", orient="records", indent=4)

# 3. Load data from CSV, Excel and JSON
csv_df = pd.read_csv("sales_part1.csv")
excel_df = pd.read_excel("sales_part2.xlsx")
json_df = pd.read_json("sales_part3.json")

# 4. Combine all datasets
df = pd.concat([csv_df, excel_df, json_df], ignore_index=True)

# 5. Explore dataset
print("\nCombined Shape:", df.shape)
print(df.info())
print("\nSummary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# 6. Data Cleaning
df.drop_duplicates(inplace=True)

# Remove extra spaces from text columns
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

df["Sale_Date"] = pd.to_datetime(df["Sale_Date"], errors="coerce")

# Convert numeric columns
num_cols = ["Sales_Amount", "Quantity_Sold",
            "Unit_Cost", "Unit_Price", "Discount"]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Fill missing values
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = ["Sales_Rep", "Region", "Product_Category",
            "Customer_Type", "Payment_Method", "Sales_Channel"]

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.dropna(subset=["Sale_Date", "Product_ID"], inplace=True)

# Remove invalid records
df = df[(df["Sales_Amount"] >= 0) &
        (df["Quantity_Sold"] > 0) &
        (df["Unit_Cost"] >= 0) &
        (df["Unit_Price"] >= 0)]

# 7. Data Transformation
df["Year"] = df["Sale_Date"].dt.year
df["Month"] = df["Sale_Date"].dt.month
df["Total_Cost"] = df["Quantity_Sold"] * df["Unit_Cost"]
df["Estimated_Profit"] = df["Sales_Amount"] - df["Total_Cost"]
df["Region_Sales_Rep"] = df["Region"] + "-" + df["Sales_Rep"]

# 8. Sales Analysis and Aggregation
print("\nTotal Sales:", df["Sales_Amount"].sum())
print("Average Sales:", df["Sales_Amount"].mean())
print("Total Quantity:", df["Quantity_Sold"].sum())
print("Total Profit:", df["Estimated_Profit"].sum())

category_sales = df.groupby("Product_Category")["Sales_Amount"].sum()
region_sales = df.groupby("Region")["Sales_Amount"].sum()
rep_sales = df.groupby("Sales_Rep")["Sales_Amount"].sum()
payment_sales = df.groupby("Payment_Method")["Sales_Amount"].sum()
channel_sales = df.groupby("Sales_Channel")["Sales_Amount"].sum()
monthly_sales = df.groupby(df["Sale_Date"].dt.to_period("M"))["Sales_Amount"].sum()

print("\nCategory Sales:\n", category_sales)
print("\nRegion Sales:\n", region_sales)
print("\nRepresentative Sales:\n", rep_sales)
print("\nPayment Sales:\n", payment_sales)
print("\nChannel Sales:\n", channel_sales)
print("\nMonthly Sales:\n", monthly_sales)

# 9. Data Visualization
sns.set_style("whitegrid")

# Sales by Product Category
category_sales.plot(kind="bar", figsize=(8, 4), title="Sales by Category")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# Regional Sales Distribution
region_sales.plot(kind="pie", autopct="%1.1f%%", figsize=(6, 6))
plt.title("Regional Sales Distribution")
plt.ylabel("")
plt.show()

# Sales Distribution by Category
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="Product_Category", y="Sales_Amount")
plt.title("Sales Distribution by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Sales Trend
monthly_sales.plot(kind="line", marker="o", figsize=(8, 4),
                   title="Monthly Sales Trend")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

# 10. Save cleaned data in three formats
df.to_csv("cleaned_sales_data.csv", index=False)
df.to_excel("cleaned_sales_data.xlsx", index=False)
df.to_json("cleaned_sales_data.json", orient="records", indent=4)

print("\nAll files saved successfully!")
print("PRACTICAL COMPLETED SUCCESSFULLY!")
