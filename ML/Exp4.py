
# ==========================================
# K-MEANS CLUSTERING ON IRIS DATASET
# ==========================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 2: Load Iris dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("First five rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Step 3: Preprocessing
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 4: Elbow Method
wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Step 5: Apply K-Means with optimal K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Add cluster labels
df["Cluster"] = y_kmeans

print("\nClustered Data:")
print(df.head())

# Display number of points in each cluster
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

# Step 7: Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 2],
    X_scaled[:, 3],
    c=y_kmeans,
    cmap='viridis',
    s=50
)

# Plot cluster centroids
centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 2],
    centers[:, 3],
    color='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.show()

# Step 8: Display cluster centers
print("\nCluster Centroids:")
print(kmeans.cluster_centers_)

print("\nOptimal Number of Clusters: 3")
