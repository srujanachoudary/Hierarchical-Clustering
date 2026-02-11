import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("📰 Financial News Clustering using Hierarchical Clustering")

# ---------------------------------------------------
# Load data + train clustering + compute centroids
# ---------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("all-data.csv", encoding="latin1", header=None)
    df.columns = ["Sentiment","News"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(df["News"])
    X_dense = X.toarray()

    # Hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    clusters = hc.fit_predict(X_dense)

    # Compute centroids manually (needed for prediction)
    centroids = []
    for i in range(3):
        centroids.append(X_dense[clusters == i].mean(axis=0))
    centroids = np.array(centroids)

    return df, tfidf, X_dense, centroids

df, tfidf, X_dense, centroids = load_model()

# ---------------------------------------------------
# 📊 Show Dendrogram
# ---------------------------------------------------
st.subheader("📊 Hierarchical Clustering Dendrogram")

sample_data = X_dense[:500]   # small sample for speed
linked = linkage(sample_data, method='ward')

fig, ax = plt.subplots(figsize=(10,5))
dendrogram(linked, truncate_mode='level', p=5, ax=ax)

# horizontal cut line
ax.axhline(y=15, linestyle='--')

st.pyplot(fig)

# ---------------------------------------------------
# 🔍 News Cluster Prediction
# ---------------------------------------------------
st.subheader("🔍 Predict Cluster for New News")

user_input = st.text_area("Enter Financial News")

if st.button("Predict Cluster"):
    if user_input.strip() == "":
        st.warning("Please enter news text")
    else:
        new_vec = tfidf.transform([user_input]).toarray()
        distances = euclidean_distances(new_vec, centroids)
        predicted_cluster = distances.argmin()

        st.success(f"Predicted Cluster: {predicted_cluster}")

        if predicted_cluster == 0:
            st.write("📈 Topic: Stock Market / Company Performance")
        elif predicted_cluster == 1:
            st.write("🏛️ Topic: Economy / Government / Policy")
        else:
            st.write("🤝 Topic: Mergers / Acquisitions / Deals")
