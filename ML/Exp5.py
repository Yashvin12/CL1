
# ==========================================
# RANDOM FOREST CLASSIFIER - CAR EVALUATION
# ==========================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Load dataset
df = pd.read_csv("car_evaluation.csv")

print("First five rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Step 3: Data preprocessing
# Remove missing values
df.dropna(inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

print("\nDataset after preprocessing:")
print(df.info())

# Step 4: Encode categorical data
# Convert text values into numerical values

encoders = {}

for column in df.columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column])

    encoders[column] = le

print("\nEncoded Dataset:")
print(df.head())

# Step 5: Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 7: Create Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 8: Train the model
model.fit(X_train, y_train)

print("\nRandom Forest Model Training Completed!")

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoders["class"].classes_
))

# Step 11: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoders["class"].classes_,
    yticklabels=encoders["class"].classes_
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Step 12: Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(
    kind="bar",
    title="Feature Importance in Random Forest",
    figsize=(8, 5)
)

plt.ylabel("Importance")
plt.show()

# Step 13: Predict a sample car
sample_car = pd.DataFrame(
    [[2, 2, 1, 2, 1, 2]],
    columns=X.columns
)

prediction = model.predict(sample_car)

predicted_class = encoders["class"].inverse_transform(prediction)

print("\nPredicted Car Safety/Acceptability:", predicted_class[0])
