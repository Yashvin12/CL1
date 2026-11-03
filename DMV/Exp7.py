
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 2. Load and explore dataset
df = pd.read_csv("Stock_Prices.csv")

print("First 5 Records:\n", df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# 3. Clean and prepare data
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["close"] = pd.to_numeric(df["close"])
df["volume"] = pd.to_numeric(df["volume"])

df.drop_duplicates(subset=["timestamp"], inplace=True)
df.sort_values("timestamp", inplace=True)
df.set_index("timestamp", inplace=True)

# Select AAPL stock
df = df[df["symbol"] == "AAPL"].copy()

price = df["close"]

# 4. Historical Stock Price
plt.figure(figsize=(10, 4))
plt.plot(price)
plt.title("Historical Stock Prices - AAPL")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.grid()
plt.show()

# 5. Moving Averages
df["MA_20"] = price.rolling(20).mean()
df["MA_50"] = price.rolling(50).mean()

plt.figure(figsize=(10, 4))
plt.plot(price, label="Closing Price")
plt.plot(df["MA_20"], label="20-Period Average")
plt.plot(df["MA_50"], label="50-Period Average")
plt.title("Stock Price Moving Averages")
plt.legend()
plt.grid()
plt.show()

# 6. Monthly Seasonality Analysis
monthly_returns = (price.pct_change() * 100).groupby(
    price.index.month
).mean()

plt.figure(figsize=(8, 4))
plt.bar(monthly_returns.index, monthly_returns.values)
plt.title("Average Monthly Stock Returns")
plt.xlabel("Month")
plt.ylabel("Average Return (%)")
plt.xticks(range(1, 13))
plt.show()

# 7. Trading Volume Analysis
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["volume"])
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid()
plt.show()

# Correlation Analysis
df["daily_return"] = price.pct_change()

print("\nCorrelation:\n",
      df[["close", "volume", "open", "high", "low"]].corr())

plt.figure(figsize=(7, 5))
plt.imshow(df[["close", "volume", "open", "high", "low"]].corr(),
           cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(5), ["Close", "Volume", "Open", "High", "Low"])
plt.yticks(range(5), ["Close", "Volume", "Open", "High", "Low"])
plt.title("Stock Data Correlation")
plt.show()

# 8. ARIMA Forecasting
model = ARIMA(price, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=10)
print("\nNext 10 Period Forecast:\n", forecast)

future_dates = pd.date_range(
    start=price.index[-1] + pd.Timedelta(hours=1),
    periods=10, freq="h"
)

plt.figure(figsize=(10, 4))
plt.plot(price.tail(100), label="Historical Price")
plt.plot(future_dates, forecast.values, label="Forecast")
plt.title("ARIMA Stock Price Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

print("\nStock Market Analysis Completed Successfully!")
