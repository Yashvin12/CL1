
# ==========================================
# HANDWRITTEN DIGIT CLASSIFICATION USING SVM
# ==========================================

# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 2: Load handwritten digits dataset
digits = load_digits()

X = digits.data
y = digits.target

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)
print("Digit Classes:", np.unique(y))

# Step 3: Display sample handwritten images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for ax, image, label in zip(
    axes.ravel(), digits.images[:10], digits.target[:10]
):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Digit: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Step 4: Preprocess the data
# Normalize pixel values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 6: Create SVM classifier
model = SVC(kernel="rbf", C=10, gamma="scale")

# Step 7: Train the SVM model
model.fit(X_train, y_train)

print("\nSVM Model Training Completed!")

# Step 8: Predict test data
y_pred = model.predict(X_test)

# Step 9: Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("SVM Confusion Matrix")
plt.show()

# Step 11: Display actual vs predicted digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for ax, index in zip(axes.ravel(), range(10)):
    ax.imshow(digits.images[X_test.shape[0] * 0 + 
                            np.where(
                                (digits.target == y_test[index])
                            )[0][0]], cmap="gray")
    ax.set_title(f"Actual: {y_test[index]}")
    ax.axis("off")

plt.tight_layout()
plt.show()

# Step 12: Predict a sample digit
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nActual Digit:", y_test[0])
print("Predicted Digit:", prediction[0])
