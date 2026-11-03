
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load and explore dataset
df = pd.read_csv("City_AQI.csv")

print(df.head())
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 3. Clean data
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df.drop_duplicates(inplace=True)

num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 4. AQI Category Distribution - Bar Chart
counts = df["aqi_category"].value_counts()

plt.figure(figsize=(8, 4))
plt.bar(counts.index, counts.values)
plt.title("AQI Category Distribution")
plt.xlabel("AQI Category")
plt.ylabel("Number of Cities")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 5. Top 10 Cities - Bar Chart
top = df.nlargest(10, "aqi_value")

plt.figure(figsize=(8, 4))
plt.bar(top["city"], top["aqi_value"])
plt.title("Top 10 Cities with Highest AQI")
plt.xticks(rotation=45)
plt.ylabel("AQI Value")
plt.tight_layout()
plt.show()

# 6. AQI Distribution - Histogram
plt.figure(figsize=(8, 4))
plt.hist(df["aqi_value"], bins=20, edgecolor="black")
plt.title("AQI Distribution")
plt.xlabel("AQI Value")
plt.ylabel("Frequency")
plt.show()

# 7. Average Pollutant AQI - Bar Chart
pollutants = ["co_aqi_value", "ozone_aqi_value",
              "no2_aqi_value", "pm2.5_aqi_value"]

plt.figure(figsize=(8, 4))
plt.bar(pollutants, df[pollutants].mean())
plt.title("Average Pollutant AQI")
plt.ylabel("Average AQI")
plt.show()

# 8. Pollutant Distribution - Box Plot
plt.figure(figsize=(8, 4))
plt.boxplot([df[c] for c in pollutants],
            labels=["CO", "Ozone", "NO2", "PM2.5"])
plt.title("Pollutant Distribution")
plt.ylabel("AQI Value")
plt.show()

# 9. PM2.5 vs Overall AQI - Scatter Plot
plt.figure(figsize=(8, 4))
plt.scatter(df["pm2.5_aqi_value"], df["aqi_value"], alpha=0.4)
plt.title("PM2.5 vs Overall AQI")
plt.xlabel("PM2.5 AQI")
plt.ylabel("Overall AQI")
plt.show()

# 10. AQI Category - Pie Chart
plt.figure(figsize=(7, 7))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
plt.title("AQI Category Percentage")
plt.show()

# 11. Statistical Analysis
print("\nAQI Summary:\n", df["aqi_value"].describe())

print("\nAverage AQI by Country:\n",
      df.groupby("country")["aqi_value"].mean()
      .sort_values(ascending=False).head(10))

# 12. Save cleaned dataset
df.to_csv("cleaned_city_aqi.csv", index=False)

print("\nData Visualization Completed Successfully!")
