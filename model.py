import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data (pastikan path sesuai struktur app.py)
df = pd.read_csv("src/data/destinasi-wisata-YKSM.csv")

# Pastikan kolom numerik dan data konsisten seperti di app.py
df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Konversi koordinat ke desimal
df["lat_decimal"] = df["Lat"] / 1e7
df["long_decimal"] = df["Long"] / 1e7

# Drop baris dengan koordinat tidak valid (sama seperti app.py)
df = df.dropna(subset=["lat_decimal", "long_decimal"])
df = df[
    (df["lat_decimal"] > -10) & (df["lat_decimal"] < 0) &
    (df["long_decimal"] > 100) & (df["long_decimal"] < 120)
]

# Gabungkan fitur-fitur yang relevan untuk content-based
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
with open("src/data/content_based_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("src/data/content_based_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
df.to_csv("src/data/content_based_index.csv", index=False)