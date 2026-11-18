
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances
)

st.set_page_config(page_title="Customer Clustering", layout="wide")

st.title("Customer Segmentation using Clustering")
st.write("Comparison of K-Means and K-Medoids Algorithms")

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

st.subheader("1. Dataset Overview")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

# Preprocessing
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select number of clusters
st.sidebar.header("Clustering Settings")
k = st.sidebar.slider("Select Number of Clusters", 2, 10, 6)

# K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# K-Medoids
def k_medoids(X, k, max_iter=100):
    distances = pairwise_distances(X)
    np.random.seed(42)

    medoids = np.random.choice(len(X), k, replace=False)

    for _ in range(max_iter):
        labels = np.argmin(distances[:, medoids], axis=1)
        new_medoids = medoids.copy()

        for i in range(k):
            points = np.where(labels == i)[0]

            if len(points) > 0:
                d = distances[np.ix_(points, points)]
                new_medoids[i] = points[np.argmin(d.sum(axis=1))]

        if np.array_equal(medoids, new_medoids):
            break

        medoids = new_medoids

    labels = np.argmin(distances[:, medoids], axis=1)
    return labels, medoids

kmedoids_labels, medoid_indices = k_medoids(X_scaled, k)

# Evaluation
def evaluate(labels):
    return [
        silhouette_score(X_scaled, labels),
        davies_bouldin_score(X_scaled, labels),
        calinski_harabasz_score(X_scaled, labels)
    ]

km_metrics = evaluate(kmeans_labels)
kmed_metrics = evaluate(kmedoids_labels)

# Elbow Method
st.subheader("2. Elbow Method")

wcss = []

for i in range(1, 11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method")
st.pyplot(fig)

# Silhouette Analysis
st.subheader("3. Silhouette Score Analysis")

scores = []

for i in range(2, 11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

fig, ax = plt.subplots()
ax.plot(range(2, 11), scores, marker='o')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Analysis")
st.pyplot(fig)

# K-Means Visualization
st.subheader("4. K-Means Clustering")

fig, ax = plt.subplots()

ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=kmeans_labels,
    cmap='rainbow'
)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

ax.scatter(
    centroids[:, 1],
    centroids[:, 2],
    marker='X',
    color='black',
    s=200,
    label='Centroids'
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.legend()
st.pyplot(fig)

# K-Medoids Visualization
st.subheader("5. K-Medoids Clustering")

fig, ax = plt.subplots()

ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=kmedoids_labels,
    cmap='rainbow'
)

medoids = X.iloc[medoid_indices]

ax.scatter(
    medoids['Annual Income (k$)'],
    medoids['Spending Score (1-100)'],
    marker='X',
    color='black',
    s=200,
    label='Medoids'
)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.legend()
st.pyplot(fig)

# Evaluation Comparison
st.subheader("6. Algorithm Comparison")

comparison = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Index'
    ],
    'K-Means': km_metrics,
    'K-Medoids': kmed_metrics
})

st.dataframe(comparison.round(4))

st.subheader("7. Conclusion")

if km_metrics[0] > kmed_metrics[0]:
    st.write("K-Means achieved a higher Silhouette Score.")
else:
    st.write("K-Medoids achieved a higher Silhouette Score.")

st.write("The best algorithm is determined by comparing all three evaluation metrics.")
