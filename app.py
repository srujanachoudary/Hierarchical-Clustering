import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

# ---------------------------
# Load dataset and train model
# ---------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("all-data.csv", encoding="latin1", header=None)
    df.columns = ["Sentiment","News"]

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    X = tfidf.fit_transform(df["News"])
    X_dense = X.toarray()

    # Hierarchical Clustering
    hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    clusters = hc.fit_predict(X_dense)

    return tfidf, X_dense, clusters

tfidf, X_dense, clusters = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📰 Financial News Clustering App")
st.write("Enter a news headline and see which cluster it belongs to")

user_input = st.text_area("Enter Financial News:")

# ---------------------------
# Predict cluster for new text
# ---------------------------
if st.button("Predict Cluster"):
    if user_input.strip() == "":
        st.warning("Please enter some news text")
    else:
        new_vec = tfidf.transform([user_input]).toarray()

        # Find closest cluster using distance
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(new_vec, X_dense)
        closest_index = distances.argmin()
        predicted_cluster = clusters[closest_index]

        st.success(f"Predicted Cluster: {predicted_cluster}")

        if predicted_cluster == 0:
            st.write("📈 Topic: Stock Market / Company Performance")
        elif predicted_cluster == 1:
            st.write("🏛️ Topic: Economy / Government / Policy")
        else:
            st.write("🤝 Topic: Mergers / Acquisitions / Deals")
