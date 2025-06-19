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

st.set_page_config(page_title="Dashboard Destinasi Wisata Yogyakarta", layout="wide")

page = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Rekomendasi"])

if page == "Dashboard":
    st.markdown("<h2 style='text-align: center;'>WISATAKU</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Rekomendasi Destinasi Wisata D.I.Y dan Semarang</h4>", unsafe_allow_html=True)

    with st.expander("Filter Data", expanded=True):
        category = st.multiselect("Kategori", options=sorted(df["Category"].dropna().unique()))
        city = st.multiselect("Kota", options=sorted(df["City"].dropna().unique()))
        indoor_outdoor = st.multiselect("Outdoor/Indoor", options=sorted(df["Outdoor/Indoor"].dropna().unique()))

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

    st.subheader("Statistik Persebaran Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Destinasi", len(filtered_df))
        st.metric("Harga Tiket Min", f"Rp{filtered_df['Price'].min():,.0f}" if not filtered_df.empty else "-")
    with col2:
        st.metric("Rata-rata Rating", f"{filtered_df['Rating'].mean():.2f}" if not filtered_df.empty else "-")
        st.metric("Harga Tiket Max", f"Rp{filtered_df['Price'].max():,.0f}" if not filtered_df.empty else "-")

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

    st.markdown("## Data Destinasi Wisata")
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

    st.markdown("## Peta Persebaran Destinasi Wisata")
    if not filtered_df.empty:
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
        st_folium(m, use_container_width=True, height=400)
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

def get_content_based_recommendations_by_filter(df, kota, harga, cuaca):
    filtered = df.copy()
    filtered.loc[filtered["Place_Name"].str.lower().str.contains("alun-alun", na=False), "Outdoor/Indoor"] = "Outdoor"

    if kota:
        filtered = filtered[filtered["City"] == kota]
    if harga:
        if harga == "Murah (< Rp20.000)":
            filtered = filtered[filtered["Price"] < 20000]
        elif harga == "Sedang (Rp20.000 - Rp50.000)":
            filtered = filtered[(filtered["Price"] >= 20000) & (filtered["Price"] <= 50000)]
        elif harga == "Mahal (> Rp50.000)":
            filtered = filtered[filtered["Price"] > 50000]
    if cuaca and cuaca != "Semua":
        if cuaca == "Hujan":
            filtered = filtered[filtered["Outdoor/Indoor"] == "Indoor"]
        elif cuaca == "Cerah":
            filtered = filtered[filtered["Outdoor/Indoor"] == "Outdoor"]
        elif cuaca == "Berawan":
            pass

    if filtered.empty:
        return pd.DataFrame()
    anchor = filtered.iloc[0]["Place_Name"]
    idx = df[df['Place_Name'] == anchor].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    results = []
    for i in sorted_indices:
        row = df.iloc[i].copy()
        if "alun-alun" in str(row["Place_Name"]).lower():
            row["Outdoor/Indoor"] = "Outdoor"
        if kota and row["City"] != kota:
            continue
        if harga:
            if harga == "Murah (< Rp20.000)" and row["Price"] >= 20000:
                continue
            if harga == "Sedang (Rp20.000 - Rp50.000)" and not (20000 <= row["Price"] <= 50000):
                continue
            if harga == "Mahal (> Rp50.000)" and row["Price"] <= 50000:
                continue
        if cuaca and cuaca != "Semua":
            if cuaca == "Hujan" and row["Outdoor/Indoor"] != "Indoor":
                continue
            if cuaca == "Cerah" and row["Outdoor/Indoor"] != "Outdoor":
                continue
        results.append({
            "Nama Tempat": row["Place_Name"],
            "Kota": row["City"],
            "Outdoor/Indoor": row["Outdoor/Indoor"],
            "Harga": f"Rp{row['Price']:,.0f}",
            "Rating": row["Rating"],
            "Jumlah Ulasan": row.get("Rating_Count", 0),
            "Skor Kemiripan": f"{cosine_similarities[i]:.3f}"
        })
    return pd.DataFrame(results)

if page == "Rekomendasi":
    st.markdown("<h2 style='text-align: center;'>Rekomendasi Destinasi Wisata</h2>", unsafe_allow_html=True)
    st.write("Pilih preferensi Anda:")

    # Input filter dalam satu baris (desktop)
    col1, col2, col3 = st.columns(3)
    with col1:
        kota = st.selectbox("Kota", options=[""] + sorted(df_cb["City"].dropna().unique().tolist()), key="kota")
    with col2:
        harga = st.selectbox("Harga", options=["", "Murah (< Rp20.000)", "Sedang (Rp20.000 - Rp50.000)", "Mahal (> Rp50.000)"], key="harga")
    with col3:
        cuaca = st.selectbox("Cuaca Saat Ini", options=["Semua", "Cerah", "Hujan", "Berawan"], key="cuaca")

    if st.button("Rekomendasikan"):
        rekomendasi_df = get_content_based_recommendations_by_filter(df_cb, kota, harga, cuaca)
        if not rekomendasi_df.empty:
            st.markdown("#### Tabel Rekomendasi Destinasi Wisata:")
            st.dataframe(rekomendasi_df, use_container_width=True)
            st.caption(f"Menampilkan {len(rekomendasi_df)} rekomendasi")
        else:
            st.info("Tidak ada destinasi yang sesuai filter Anda.")