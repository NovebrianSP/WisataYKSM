import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os

# Load data
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
    st.header("Rekomendasi Destinasi Wisata Yogyakarta dan Semarang")

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

elif page == "Rekomendasi":
    st.title("Rekomendasi Destinasi Wisata")
    st.markdown("Berikut adalah 5 destinasi wisata dengan rating tertinggi:")

    rekomendasi = df.sort_values(by=["Rating", "Rating_Count"], ascending=[False, False]).head(5)
    for i, row in rekomendasi.iterrows():
        st.markdown(f"### {row['Place_Name']}")
        st.write(f"**Kategori:** {row['Category']}")
        st.write(f"**Kota:** {row['City']}")
        st.write(f"**Rating:** {row['Rating']} ({row['Rating_Count']} ulasan)")
        st.write(f"**Harga Tiket:** Rp{row['Price']:,.0f}")
        st.write(row['Description'])
        st.markdown("---")