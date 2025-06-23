import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os
import pickle
from sklearn.metrics.pairwise import linear_kernel
from user_db import create_user_table, add_user, get_user, user_exists, hash_password, save_user_preference
from user_db import get_similar_users, get_all_user_preferences
create_user_table()

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

if st.session_state.get("logged_in"):
    allowed_pages = ["Dashboard", "Rekomendasi"]
else:
    allowed_pages = ["Login", "Register", "Dashboard", "Rekomendasi"]

page = st.sidebar.selectbox(
    "Pilih Halaman", 
    allowed_pages
)

if page == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = get_user(username, hash_password(password))
        if user:
            st.success(f"Selamat datang, {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()  # Redirect ke Dashboard
        else:
            st.error("Username atau password salah.")
    st.stop()

if page == "Register":
    st.title("Register")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if not email or not username or not password:
            st.warning("Semua field harus diisi.")
        elif user_exists(username, email):
            st.error("Username atau email sudah terdaftar.")
        else:
            add_user(email, username, hash_password(password))
            st.success("Registrasi berhasil! Silakan login.")
    st.stop()

if page == "Dashboard":
    if st.session_state.get("logged_in"):
        with st.sidebar:
            st.success(f"👋 Selamat Datang {st.session_state['username']}")
        # Dropdown user di kanan atas
        col1, col2 = st.columns([8, 4])
        with col2:
            with st.expander(st.session_state['username']):
                if st.button("Logout"):
                    st.session_state["logged_in"] = False
                    st.session_state["username"] = ""
                    st.rerun()
                    
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
        st.plotly_chart(fig_cat, use_container_width=True, config={'staticPlot': True})
    else:
        st.info("Tidak ada data untuk ditampilkan pada grafik kategori.")

    st.markdown("### Distribusi Rating")
    if not filtered_df.empty:
        fig_rating = px.histogram(filtered_df, x="Rating", nbins=20, title="Distribusi Rating Destinasi")
        st.plotly_chart(fig_rating, use_container_width=True, config={'staticPlot': True})
    else:
        st.info("Tidak ada data untuk ditampilkan pada grafik rating.")

    st.markdown("### Distribusi Harga Tiket")
    if not filtered_df.empty:
        fig_price = px.histogram(filtered_df, x="Price", nbins=20, title="Distribusi Harga Tiket")
        st.plotly_chart(fig_price, use_container_width=True, config={'staticPlot': True})
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
        

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

@st.cache_resource
def load_content_based():
    df = pd.read_csv(DATA_PATH)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    return df, tfidf, tfidf_matrix

def get_hybrid_recommendations(df, username, kota, harga, cuaca):
    similar_users = get_similar_users(username)
    prefs_df = get_all_user_preferences()
    collaborative_ids = []
    if similar_users:
        prefs_df = prefs_df[prefs_df['username'].isin(similar_users)]
        filter_kota = prefs_df['kota'].mode()[0] if not prefs_df['kota'].mode().empty else kota
        filter_harga = prefs_df['harga'].mode()[0] if not prefs_df['harga'].mode().empty else harga
        filter_cuaca = prefs_df['cuaca'].mode()[0] if not prefs_df['cuaca'].mode().empty else cuaca
        collaborative = df.copy()
        if filter_kota:
            collaborative = collaborative[collaborative['City'] == filter_kota]
        # --- Perbaikan di bawah ini ---
        if filter_harga:
            if filter_harga == "Murah (< Rp20.000)":
                collaborative = collaborative[collaborative["Price"] < 20000]
            elif filter_harga == "Sedang (Rp20.000 - Rp50.000)":
                collaborative = collaborative[(collaborative["Price"] >= 20000) & (collaborative["Price"] <= 50000)]
            elif filter_harga == "Mahal (> Rp50.000)":
                collaborative = collaborative[collaborative["Price"] > 50000]
        # --- End Perbaikan ---
        if filter_cuaca and filter_cuaca != "Semua":
            collaborative = collaborative[collaborative['Cuaca'] == filter_cuaca]
        collaborative_ids = collaborative['Place_Name'].tolist()
    else:
        collaborative = pd.DataFrame()

    # Content-based: destinasi sesuai filter user saat ini
    content_based = get_content_based_recommendations_by_filter(df, kota, harga, cuaca)
    content_ids = content_based['Nama Tempat'].tolist() if not content_based.empty else []

    # Gabungkan hasil collaborative dan content-based (prioritaskan yang muncul di kedua)
    hybrid_ids = []
    for id_ in collaborative_ids:
        if id_ in content_ids:
            hybrid_ids.append(id_)
    for id_ in collaborative_ids:
        if id_ not in hybrid_ids:
            hybrid_ids.append(id_)
    for id_ in content_ids:
        if id_ not in hybrid_ids:
            hybrid_ids.append(id_)

    # Ambil data destinasi berdasarkan urutan hybrid_ids
    hybrid_df = df[df['Place_Name'].isin(hybrid_ids)]
    hybrid_df['hybrid_rank'] = hybrid_df['Place_Name'].apply(lambda x: hybrid_ids.index(x))
    hybrid_df = hybrid_df.sort_values('hybrid_rank').drop('hybrid_rank', axis=1)
    return hybrid_df.head(10)

LOKASI_KOORDINAT = {
    "Kota Yogyakarta": (-7.801194, 110.364917),
    "Sleman": (-7.718817, 110.357563),
    "Bantul": (-7.887924, 110.328797),
    "Kulon Progo": (-7.824034, 110.164776),
    "Gunungkidul": (-8.030520, 110.616892),
    "Kota Semarang": (-7.005145, 110.438125),
    "Kabupaten Semarang": (-7.268218, 110.404556)
}

KOTA_LOKASI_MAP = {
    "Kota Yogyakarta": ["Kota Yogyakarta", "Sleman", "Bantul", "Kulon Progo", "Gunungkidul"],
    "Sleman": ["Sleman", "Kota Yogyakarta", "Bantul", "Kulon Progo", "Gunungkidul"],
    "Bantul": ["Bantul", "Kota Yogyakarta", "Sleman", "Kulon Progo", "Gunungkidul"],
    "Kulon Progo": ["Kulon Progo", "Kota Yogyakarta", "Sleman", "Bantul", "Gunungkidul"],
    "Gunungkidul": ["Gunungkidul", "Kota Yogyakarta", "Sleman", "Bantul", "Kulon Progo"],
    "Kota Semarang": ["Kota Semarang", "Kabupaten Semarang"],
    "Kabupaten Semarang": ["Kabupaten Semarang", "Kota Semarang"]
}

df_cb, tfidf, tfidf_matrix = load_content_based()

def get_content_based_recommendations_by_filter(df, kota, harga, cuaca, rating=None):
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
    if rating:
        try:
            rating_val = float(rating)
            filtered = filtered[filtered["Rating"] >= rating_val]
        except:
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
        if rating:
            try:
                rating_val = float(rating)
                if row["Rating"] < rating_val:
                    continue
            except:
                pass
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kota = st.selectbox("Kota", options=[""] + sorted(df_cb["City"].dropna().unique().tolist()), key="kota")
    with col2:
        harga = st.selectbox("Harga", options=["", "Murah (< Rp20.000)", "Sedang (Rp20.000 - Rp50.000)", "Mahal (> Rp50.000)"], key="harga")
    with col3:
        cuaca = st.selectbox("Cuaca Saat Ini", options=["Semua", "Cerah", "Hujan", "Berawan"], key="cuaca")
    with col4:
        rating = st.selectbox("Rating Minimal", options=[""] + [str(r) for r in range(1, 6)], key="rating")  # 1-5

    # Lokasi terkini menyesuaikan kota
    if kota and kota in KOTA_LOKASI_MAP:
        lokasi_options = KOTA_LOKASI_MAP[kota]
    else:
        lokasi_options = list(LOKASI_KOORDINAT.keys())

    lokasi_terkini = st.selectbox(
        "Lokasi Terkini Anda",
        options=[""] + lokasi_options,
        key="lokasi_terkini"
    )

    if not st.session_state.get("logged_in"):
        st.warning("Anda belum login. Rekomendasi hanya didasarkan pada rating konten saja, dan tidak ada preferensi pengguna.")

    if st.button("Rekomendasikan"):
        # Pilih metode rekomendasi
        if st.session_state.get("logged_in"):
            similar_users = get_similar_users(st.session_state["username"])
            if similar_users:
                rekomendasi_df = get_hybrid_recommendations(df_cb, st.session_state["username"], kota, harga, cuaca, rating)
            else:
                rekomendasi_df = get_content_based_recommendations_by_filter(df_cb, kota, harga, cuaca, rating)
            # Simpan preferensi
            save_user_preference(
                st.session_state["username"],
                kota,
                harga,
                cuaca,
                lokasi_terkini
            )
        else:
            rekomendasi_df = get_content_based_recommendations_by_filter(df_cb, kota, harga, cuaca, rating)

        # Urutkan berdasarkan jarak jika lokasi terkini dipilih
        if lokasi_terkini and lokasi_terkini in LOKASI_KOORDINAT and not rekomendasi_df.empty:
            pusat_lat, pusat_lon = LOKASI_KOORDINAT[lokasi_terkini]
            # Pastikan kolom koordinat ada
            if "lat_decimal" in rekomendasi_df.columns and "long_decimal" in rekomendasi_df.columns:
                rekomendasi_df["Jarak (km)"] = rekomendasi_df.apply(
                    lambda row: haversine(
                        pusat_lat, pusat_lon,
                        row.get("lat_decimal", np.nan), row.get("long_decimal", np.nan)
                    ), axis=1
                )
                rekomendasi_df = rekomendasi_df.sort_values("Jarak (km)")

        # Tampilkan hasil rekomendasi
        if not rekomendasi_df.empty:
            st.markdown("#### Tabel Rekomendasi Destinasi Wisata:")
            st.dataframe(rekomendasi_df, use_container_width=True)
            st.caption(f"Menampilkan {len(rekomendasi_df)} rekomendasi")
        else:
            st.info("Tidak ada destinasi yang sesuai filter Anda.")