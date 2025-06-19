import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os
import pickle
from sklearn.metrics.pairwise import linear_kernel

# Load data
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data/destinasi-wisata-YKSM.csv")
TFIDF_PATH = os.path.join(BASE_DIR, "data/content_based_tfidf.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "data/content_based_matrix.pkl")
csv_path = os.path.join(os.path.dirname(__file__), "data/destinasi-wisata-YKSM.csv")
df = pd.read_csv(csv_path)

# Pastikan kolom numerik
df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Konversi koordinat ke desimal
df["lat_decimal"] = df["Lat"] / 1e7
df["long_decimal"] = df["Long"] / 1e7

# Drop baris dengan koordinat tidak valid
df = df.dropna(subset=["lat_decimal", "long_decimal"])
df = df[
    (df["lat_decimal"] > -10) & (df["lat_decimal"] < 0) &
    (df["long_decimal"] > 100) & (df["long_decimal"] < 120)
]

# Streamlit page config
st.set_page_config(page_title="Dashboard Destinasi Wisata Yogyakarta", layout="wide")

# Navigasi halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Rekomendasi"])

if page == "Dashboard":
    st.markdown("<h1 style='text-align: center;'>WISATAKU</h1>", unsafe_allow_html=True,)
    st.markdown("<h2 style='text-align: center;'>Rekomendasi Destinasi Wisata D.I.Y dan Semarang</h2>", unsafe_allow_html=True)

    # Filter di halaman utama
    st.markdown("### Filter Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.multiselect("Kategori", options=sorted(df["Category"].dropna().unique()), default=None)
    with col2:
        city = st.multiselect("Kota", options=sorted(df["City"].dropna().unique()), default=None)
    with col3:
        indoor_outdoor = st.multiselect("Outdoor/Indoor", options=sorted(df["Outdoor/Indoor"].dropna().unique()), default=None)

    # Handler jika tidak ada filter dipilih
    if (not category) and (not city) and (not indoor_outdoor):
        st.warning("Tidak ada filter dipilih. Menampilkan semua data.")
        filtered_df = df.copy()
    else:
        filtered_df = df.copy()
        if category:
            filtered_df = filtered_df[filtered_df["Category"].isin(category)]
        if city:
            filtered_df = filtered_df[filtered_df["City"].isin(city)]
        if indoor_outdoor:
            filtered_df = filtered_df[filtered_df["Outdoor/Indoor"].isin(indoor_outdoor)]

    # Statistik Persebaran Data
    st.subheader("Statistik Persebaran Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Destinasi", len(filtered_df))
    col2.metric("Rata-rata Rating", f"{filtered_df['Rating'].mean():.2f}" if not filtered_df.empty else "-")
    col3.metric("Harga Tiket Min", f"Rp{filtered_df['Price'].min():,.0f}" if not filtered_df.empty else "-")
    col4.metric("Harga Tiket Max", f"Rp{filtered_df['Price'].max():,.0f}" if not filtered_df.empty else "-")

    # Visualisasi Data
    st.markdown("### Distribusi Kategori")
    if not filtered_df.empty:
        fig_cat = px.histogram(filtered_df, x="Category", color="Category", title="Jumlah Destinasi per Kategori")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan pada grafik kategori.")

    st.markdown("### Distribusi Rating")
    if not filtered_df.empty:
        fig_rating = px.histogram(filtered_df, x="Rating", nbins=20, title="Distribusi Rating Destinasi")
        st.plotly_chart(fig_rating, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan pada grafik rating.")

    st.markdown("### Distribusi Harga Tiket")
    if not filtered_df.empty:
        fig_price = px.histogram(filtered_df, x="Price", nbins=20, title="Distribusi Harga Tiket")
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Tidak ada data untuk ditampilkan pada grafik harga.")
        
    # Tabel Data
    st.markdown("## Data Destinasi Wisata")
    st.dataframe(filtered_df.reset_index(drop=True))

    # Map Interaktif di tengah
    st.markdown("## Peta Persebaran Destinasi Wisata")
    if not filtered_df.empty:
        map_col1, map_col2, map_col3 = st.columns([1, 6, 1])
        with map_col2:
            m = folium.Map(
                location=[filtered_df["lat_decimal"].mean(), filtered_df["long_decimal"].mean()],
                zoom_start=11
            )
            for _, row in filtered_df.iterrows():
                folium.Marker(
                    location=[row["lat_decimal"], row["long_decimal"]],
                    popup=f"<b>{row['Place_Name']}</b><br>Rating: {row['Rating']}<br>Harga: Rp{row['Price']:,.0f}",
                    tooltip=row["Place_Name"],
                    icon=folium.Icon(color="blue" if row["Outdoor/Indoor"] == "Outdoor" else "green")
                ).add_to(m)
            st_folium(m, use_container_width=True, height=500)
    else:
        st.info("Tidak ada data untuk ditampilkan pada peta.")


    @st.cache_resource
    def load_content_based():
        df = pd.read_csv(DATA_PATH)
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
        return df, tfidf, tfidf_matrix

    df_cb, tfidf, tfidf_matrix = load_content_based()

    def get_content_based_recommendations(place_name, top_n=10):
        idx = df_cb[df_cb['Place_Name'].str.lower() == place_name.lower()].index
        if len(idx) == 0:
            return []
        idx = idx[0]
        cosine_similarities = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
        results = []
        for i in related_docs_indices:
            results.append({
                "Place_Name": df_cb.iloc[i]["Place_Name"],
                "Category": df_cb.iloc[i]["Category"],
                "City": df_cb.iloc[i]["City"],
                "Rating": df_cb.iloc[i]["Rating"],
                "Rating_Count": df_cb.iloc[i].get("Rating_Count", 0),
                "Price": df_cb.iloc[i]["Price"],
                "Description": df_cb.iloc[i]["Description"],
                "Score": cosine_similarities[i]
            })
        return results

    def get_hybrid_recommendations(place_name, top_n=5):
        # Ambil 10 teratas content-based
        content_recs = get_content_based_recommendations(place_name, top_n=10)
        # Urutkan berdasarkan rating dan rating_count (popularity)
        content_recs = sorted(
            content_recs,
            key=lambda x: (x["Rating"], x.get("Rating_Count", 0)),
            reverse=True
        )
        return content_recs[:top_n]

    # --- Halaman Rekomendasi Hybrid ---
    st.title("Rekomendasi Destinasi Wisata (Hybrid Filtering)")
    st.write("Pilih destinasi yang Anda sukai, lalu dapatkan rekomendasi destinasi serupa dan populer:")

    place_options = df_cb["Place_Name"].sort_values().unique()
    selected_place = st.selectbox("Pilih Destinasi", place_options)

    if selected_place:
        rekomendasi = get_hybrid_recommendations(selected_place, top_n=5)
        if rekomendasi:
            st.markdown(f"#### Rekomendasi mirip dan populer dengan **{selected_place}**:")
            for rec in rekomendasi:
                st.markdown(f"**{rec['Place_Name']}**")
                st.write(f"Kategori: {rec['Category']}, Kota: {rec['City']}, Rating: {rec['Rating']} ({rec.get('Rating_Count', 0)} ulasan), Harga: Rp{rec['Price']:,.0f}")
                st.write(rec['Description'])
                st.write(f"Skor kemiripan: {rec['Score']:.3f}")
                st.markdown("---")
        else:
            st.info("Tidak ditemukan rekomendasi serupa.")