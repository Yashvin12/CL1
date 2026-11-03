
# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the wine dataset
url = "https://media.geeksforgeeks.org/wp-content/uploads/Wine.csv"
df = pd.read_csv(url)

print("First five rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Step 3: Separate features and target
X = df.drop("Customer_Segment", axis=1)
y = df["Customer_Segment"]

print("\nFeatures:", X.shape)
print("\nWine Classes:", y.unique())

# Step 4: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Create a new dataframe
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Wine_Class"] = y.values

print("\nTransformed Data:")
print(pca_df.head())

# Step 7: Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Captured:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Step 8: Visualize PCA results
plt.figure(figsize=(8, 6))

for wine_class in sorted(y.unique()):
    data = pca_df[pca_df["Wine_Class"] == wine_class]

    plt.scatter(
        data["PC1"],
        data["PC2"],
        label=f"Class {wine_class}",
        alpha=0.8
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.legend()
plt.grid(True)
plt.show()
