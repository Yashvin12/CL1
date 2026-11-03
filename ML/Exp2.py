
# ==========================================
# UBER FARE PREDICTION USING REGRESSION
# ==========================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Step 2: Load dataset
df = pd.read_csv("uber.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Step 3: Data preprocessing
# Remove unnecessary column
df.drop(columns=["Unnamed: 0", "key"], errors="ignore", inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Convert pickup_datetime to datetime
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
)

df.dropna(subset=["pickup_datetime"], inplace=True)

# Extract date and time features
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year

# Remove original datetime column
df.drop(columns=["pickup_datetime"], inplace=True)

print("\nAfter Preprocessing:")
print(df.head())

# Step 4: Remove invalid values
df = df[
    (df["fare_amount"] > 0) &
    (df["passenger_count"] >= 1) &
    (df["passenger_count"] <= 6) &
    (df["pickup_latitude"].between(-90, 90)) &
    (df["dropoff_latitude"].between(-90, 90)) &
    (df["pickup_longitude"].between(-180, 180)) &
    (df["dropoff_longitude"].between(-180, 180))
].copy()

# Step 5: Outlier detection using IQR
Q1 = df["fare_amount"].quantile(0.25)
Q3 = df["fare_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = max(0, Q1 - 1.5 * IQR)
upper_limit = Q3 + 1.5 * IQR

print("\nOutlier Limits:", lower_limit, upper_limit)

# Display boxplot before removing outliers
plt.figure(figsize=(7, 4))
sns.boxplot(x=df["fare_amount"])
plt.title("Fare Amount Outliers")
plt.show()

# Remove fare outliers
df = df[
    df["fare_amount"].between(lower_limit, upper_limit)
].copy()

print("Dataset after outlier removal:", df.shape)

# Step 6: Correlation analysis
plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

# Step 7: Select features and target
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]

# Step 8: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 9: Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Create regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000)
}

# Step 11: Train and evaluate models
results = []

for name, model in models.items():

    # Train model
    model.fit(X_train_scaled, y_train)

    # Predict fare
    y_pred = model.predict(X_test_scaled)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results.append([
        name, r2, mae, mse, rmse
    ])

    print("\n", name)
    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))

# Step 12: Compare models
results_df = pd.DataFrame(
    results,
    columns=["Model", "R2 Score", "MAE", "MSE", "RMSE"]
)

print("\nMODEL COMPARISON:")
print(results_df.to_string(index=False))

# Step 13: Visualize model performance
plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Comparison of Regression Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.show()

# Step 14: Predict fare for a sample ride
sample_ride = pd.DataFrame([{
    "pickup_longitude": -73.985,
    "pickup_latitude": 40.748,
    "dropoff_longitude": -73.985,
    "dropoff_latitude": 40.758,
    "passenger_count": 2,
    "hour": 14,
    "day": 3,
    "month": 11,
    "year": 2026
}])

sample_scaled = scaler.transform(sample_ride)

# Use Linear Regression for prediction
prediction = models["Linear Regression"].predict(sample_scaled)

print("\nPredicted Uber Fare: $", round(prediction[0], 2))
