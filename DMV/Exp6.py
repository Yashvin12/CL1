
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load dataset
df = pd.read_csv("sales_data.csv")

# 3. Explore dataset
print("First 5 Records:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# 4. Clean dataset
df.drop_duplicates(inplace=True)
df.dropna(subset=["Region", "Sales_Amount",
                  "Product_Category"], inplace=True)

# 5. Aggregate sales by region
region_sales = df.groupby("Region")["Sales_Amount"].sum()
print("\nTotal Sales by Region:\n", region_sales)

# 6. Aggregate sales by product category
category_sales = df.groupby("Product_Category")["Sales_Amount"].sum()
print("\nTotal Sales by Product Category:\n", category_sales)

# 7. Aggregate sales by sales representative
rep_sales = df.groupby("Sales_Rep")["Sales_Amount"].sum()
print("\nTotal Sales by Sales Representative:\n", rep_sales)

# 8. Aggregate sales by sales channel
channel_sales = df.groupby("Sales_Channel")["Sales_Amount"].sum()
print("\nTotal Sales by Channel:\n", channel_sales)

# 9. Find top-performing region
print("\nTop Performing Region:", region_sales.idxmax())
print("Highest Sales:", region_sales.max())

# 10. Bar Chart - Region-wise Sales
region_sales.plot(kind="bar", figsize=(8, 4),
                  title="Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.tight_layout()
plt.show()

# 11. Pie Chart - Region-wise Sales Distribution
region_sales.plot(kind="pie", autopct="%1.1f%%",
                  figsize=(7, 7))
plt.title("Sales Distribution by Region")
plt.ylabel("")
plt.show()

# 12. Bar Chart - Product Category Sales
category_sales.plot(kind="bar", figsize=(8, 4),
                    title="Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Sales Amount")
plt.tight_layout()
plt.show()

# 13. Save aggregated results
region_sales.to_csv("region_sales.csv")
category_sales.to_csv("category_sales.csv")
rep_sales.to_csv("representative_sales.csv")
channel_sales.to_csv("channel_sales.csv")

print("\nData Aggregation Completed Successfully!")
