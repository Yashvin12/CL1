# api keys b366b32a39141148ba56c9918cd5e2e9
# 93f48f9d3d0a7573758c764b7f469561
# 1. Import libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. API Configuration
API_KEY = input("Enter OpenWeatherMap API Key: ")
CITY = "Pune"
COUNTRY = "IN"

url = "https://api.openweathermap.org/data/2.5/forecast"
params = {"q": f"{CITY},{COUNTRY}", "appid": API_KEY, "units": "metric"}

# 3. Fetch Weather Data
response = requests.get(url, params=params, timeout=20)

if response.status_code != 200:
    raise SystemExit("API Error: " + response.text)

weather_data = response.json()
print("Weather data retrieved successfully!")
print("City:", weather_data["city"]["name"])

# 4. Extract Weather Attributes
records = []

for item in weather_data["list"]:
    records.append({
        "Date_Time": item["dt_txt"],
        "Temperature": item["main"]["temp"],
        "Feels_Like": item["main"]["feels_like"],
        "Min_Temperature": item["main"]["temp_min"],
        "Max_Temperature": item["main"]["temp_max"],
        "Humidity": item["main"]["humidity"],
        "Pressure": item["main"]["pressure"],
        "Wind_Speed": item["wind"]["speed"],
        "Wind_Direction": item["wind"].get("deg"),
        "Cloudiness": item["clouds"]["all"],
        "Weather": item["weather"][0]["description"],
        "Rainfall": item.get("rain", {}).get("3h", 0),
        "Probability_Of_Rain": item.get("pop", 0) * 100
    })

df = pd.DataFrame(records)

# 5. Explore and Clean Data
print("\nFirst 5 Records:\n", df.head())
print("\nShape:", df.shape)
print("\nDataset Information:")
df.info()
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary:\n", df.describe())

df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")

num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

df["Weather"] = df["Weather"].fillna("Unknown")
df.dropna(subset=["Date_Time"], inplace=True)
df.drop_duplicates(inplace=True)

# 6. Data Transformation
df["Date"] = df["Date_Time"].dt.date
df["Hour"] = df["Date_Time"].dt.hour

# 7. Descriptive Weather Analysis
print("\nAverage Temperature:", df["Temperature"].mean())
print("Maximum Temperature:", df["Temperature"].max())
print("Minimum Temperature:", df["Temperature"].min())
print("Average Humidity:", df["Humidity"].mean())
print("Average Wind Speed:", df["Wind_Speed"].mean())
print("Total Forecast Rainfall:", df["Rainfall"].sum())

# 8. Daily Data Aggregation
daily_data = df.groupby("Date").agg({
    "Temperature": ["mean", "min", "max"],
    "Humidity": "mean",
    "Wind_Speed": "mean",
    "Rainfall": "sum"
})

print("\nDaily Weather Summary:\n", daily_data)

# 9. Data Visualization
sns.set_style("whitegrid")

# Temperature over time
plt.figure(figsize=(9, 4))
plt.plot(df["Date_Time"], df["Temperature"], marker="o")
plt.title("Temperature Changes Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Daily average temperature
daily_temp = df.groupby("Date")["Temperature"].mean()
daily_temp.plot(kind="bar", figsize=(8, 4), title="Daily Average Temperature")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# Temperature vs Humidity
plt.figure(figsize=(7, 4))
plt.scatter(df["Temperature"], df["Humidity"])
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.show()

# Wind speed variation
plt.figure(figsize=(9, 4))
plt.plot(df["Date_Time"], df["Wind_Speed"], marker="o")
plt.title("Wind Speed Variation")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Daily rainfall
daily_rain = df.groupby("Date")["Rainfall"].sum()
daily_rain.plot(kind="bar", figsize=(8, 4), title="Daily Forecast Rainfall")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Weather Parameters Correlation")
plt.tight_layout()
plt.show()

# 10. Save Weather Data
df.to_csv("weather_data.csv", index=False)
daily_data.to_csv("daily_weather_summary.csv")

print("\nWeather data saved successfully!")
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
