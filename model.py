import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load data
df = pd.read_csv("../destinasi-wisata/src/data/destinasi-wisata-YKSM.csv")

# Gabungkan fitur-fitur yang relevan untuk content-based (bisa ditambah sesuai kebutuhan)
df['features'] = (
    df['Category'].astype(str) + ' ' +
    df['City'].astype(str) + ' ' +
    df['Outdoor/Indoor'].astype(str) + ' ' +
    df['Description'].astype(str) + ' ' +
    df['Price'].astype(str)
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['features'])

# Simpan model dan matrix untuk digunakan pada aplikasi rekomendasi
import pickle
with open("content_based_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("content_based_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
df.to_csv("content_based_index.csv", index=False)