
# EXPERIMENT: DATA CLEANING AND PREPARATION
# Telecom Customer Churn Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD AND EXPLORE DATA
df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.head())
print("\nShape:", df.shape)
df.info()
print(df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# 2. REMOVE DUPLICATES
print("\nDuplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset="customerID", inplace=True)

# 3. FIX STRUCTURAL ERRORS
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check unique values
for col in df.select_dtypes(include="object"):
    print(col, ":", df[col].unique()[:10])

# 4. HANDLE MISSING VALUES
df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[
    df["tenure"] == 0, "TotalCharges"
].fillna(0)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# 5. DETECT AND HANDLE OUTLIERS USING IQR
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    print(f"Outliers in {col}:", ((df[col] < lower) | (df[col] > upper)).sum())
    df[col] = df[col].clip(lower, upper)

# 6. FEATURE ENGINEERING
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df["Tenure_Group"] = pd.cut(
    df["tenure"], [-1, 12, 24, 48, 72],
    labels=["New", "Medium", "Long", "Loyal"]
)

df["Average_Monthly_Charge"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0
)

# 7. FINAL VALIDATION
print("\nFinal Dataset:\n", df.head())
print("Final Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nChurn Distribution:\n", df["Churn"].value_counts())


# 8. VISUALIZATION

# Graph 1: Churn Distribution
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

# Graph 2: Monthly Charges
sns.boxplot(data=df, x="MonthlyCharges")
plt.title("Monthly Charges After Outlier Treatment")
plt.show()

# Graph 3: Churn by Tenure Group
sns.countplot(data=df, x="Tenure_Group", hue="Churn")
plt.title("Customer Churn by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Number of Customers")
plt.show()


# 9. SAVE CLEANED DATA
df.to_csv("cleaned_telco_churn.csv", index=False)
print("\nData Cleaning and Preparation Completed!")
