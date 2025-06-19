import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Load data
DATA_PATH = "../data/destinasi-wisata-YKSM.csv"
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Dashboard Destinasi Wisata Yogyakarta", layout="wide")
st.title("Dashboard Persebaran Destinasi Wisata Yogyakarta")

# Sidebar filter
with st.sidebar:
    st.header("Filter Data")
    category = st.multiselect("Kategori", options=df["Category"].unique(), default=list(df["Category"].unique()))
    city = st.multiselect("Kota", options=df["City"].unique(), default=list(df["City"].unique()))
    indoor_outdoor = st.multiselect("Outdoor/Indoor", options=df["Outdoor/Indoor"].unique(), default=list(df["Outdoor/Indoor"].unique()))

# Filter data
filtered_df = df[
    df["Category"].isin(category) &
    df["City"].isin(city) &
    df["Outdoor/Indoor"].isin(indoor_outdoor)
]

# Persebaran Data
st.subheader("Statistik Persebaran Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Destinasi", len(filtered_df))
col2.metric("Rata-rata Rating", f"{filtered_df['Rating'].mean():.2f}")
col3.metric("Harga Tiket Min", f"Rp{filtered_df['Price'].min():,.0f}")
col4.metric("Harga Tiket Max", f"Rp{filtered_df['Price'].max():,.0f}")

# Visualisasi Data
st.markdown("### Distribusi Kategori")
fig_cat = px.histogram(filtered_df, x="Category", color="Category", title="Jumlah Destinasi per Kategori")
st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("### Distribusi Rating")
fig_rating = px.histogram(filtered_df, x="Rating", nbins=20, title="Distribusi Rating Destinasi")
st.plotly_chart(fig_rating, use_container_width=True)

st.markdown("### Distribusi Harga Tiket")
fig_price = px.histogram(filtered_df, x="Price", nbins=20, title="Distribusi Harga Tiket")
st.plotly_chart(fig_price, use_container_width=True)

# Map Interaktif
st.markdown("## Peta Persebaran Destinasi Wisata")
m = folium.Map(location=[filtered_df["Lat"].mean()/1e7, filtered_df["Long"].mean()/1e7], zoom_start=12)

for _, row in filtered_df.iterrows():
    folium.Marker(
        location=[row["Lat"]/1e7, row["Long"]/1e7],
        popup=f"<b>{row['Place_Name']}</b><br>Rating: {row['Rating']}<br>Harga: Rp{row['Price']:,.0f}",
        tooltip=row["Place_Name"],
        icon=folium.Icon(color="blue" if row["Outdoor/Indoor"] == "Outdoor" else "green")
    ).add_to(m)

st_folium(m, width=900, height=500)

# Tabel Data
st.markdown("## Data Destinasi Wisata")
st.dataframe(filtered_df.reset_index(drop=True))