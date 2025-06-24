import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os
import pickle
import requests
from user_db import create_user_table, add_user, get_user, user_exists, hash_password, save_user_preference
from user_db import get_similar_users, get_all_user_preferences
from user_db import create_favorite_table, add_favorite, remove_favorite, is_favorite
create_favorite_table()
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
        harga = st.selectbox("Harga Tiket", options=["", "Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"])

    if (not category) and (not city) and (not indoor_outdoor) and (not harga):
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
        if harga:
            if harga == "Murah (<20000)":
                filtered_df = filtered_df[filtered_df["Price"] < 20000]
            elif harga == "Sedang (20000-50000)":
                filtered_df = filtered_df[(filtered_df["Price"] >= 20000) & (filtered_df["Price"] <= 50000)]
            elif harga == "Mahal (>50000)":
                filtered_df = filtered_df[filtered_df["Price"] > 50000]

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

CITY_OPTIONS = sorted(df["City"].dropna().unique())

LOKASI_KOORDINAT = {
    "Kota Yogyakarta": (-7.801194, 110.364917),
    "Sleman": (-7.718817, 110.357563),
    "Bantul": (-7.887924, 110.328797),
    "Kulon Progo": (-7.824034, 110.164776),
    "Gunungkidul": (-8.030520, 110.616892),
    "Kota Semarang": (-7.005145, 110.438125),
    "Kabupaten Semarang": (-7.268218, 110.404556),
    "Semarang": (-7.005145, 110.438125),
    "Yogyakarta": (-7.801194, 110.364917),
}

def get_open_meteo_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if "current_weather" in data:
            cw = data["current_weather"]
            return {
                "temperature": cw.get("temperature"),
                "windspeed": cw.get("windspeed"),
                "weathercode": cw.get("weathercode"),
                "desc": weather_code_to_desc(cw.get("weathercode"))
            }
    except Exception as e:
        return None
    return None

def weather_code_to_desc(code):
    mapping = {
        0: "Cerah",
        1: "Cerah Berawan",
        2: "Berawan",
        3: "Mendung",
        45: "Berkabut",
        48: "Berkabut",
        51: "Gerimis",
        53: "Gerimis",
        55: "Gerimis",
        56: "Gerimis Beku",
        57: "Gerimis Beku",
        61: "Hujan Ringan",
        63: "Hujan Sedang",
        65: "Hujan Lebat",
        66: "Hujan Beku",
        67: "Hujan Beku",
        71: "Salju Ringan",
        73: "Salju Sedang",
        75: "Salju Lebat",
        77: "Butiran Salju",
        80: "Hujan Lokal",
        81: "Hujan Lokal",
        82: "Hujan Lokal Lebat",
        85: "Salju Lokal",
        86: "Salju Lokal Lebat",
        95: "Badai Petir",
        96: "Badai Petir Hujan Es",
        99: "Badai Petir Hujan Es"
    }
    return mapping.get(code, "Tidak diketahui")

df_cb, tfidf, tfidf_matrix = load_content_based()

def get_content_based_recommendations_by_filter(df, lokasi_terkini, cuaca_user, kategori=None, harga=None):
    filtered = df.copy()
    filtered.loc[filtered["Place_Name"].str.lower().str.contains("alun-alun", na=False), "Outdoor/Indoor"] = "Outdoor"

    # Filter lokasi hanya berdasarkan city dari csv
    if lokasi_terkini:
        filtered = filtered[filtered["City"] == lokasi_terkini]

    # Filter kategori
    if kategori:
        filtered = filtered[filtered["Category"] == kategori]

    # Filter harga
    if harga:
        if harga == "Murah (<20000)":
            filtered = filtered[filtered["Price"] < 20000]
        elif harga == "Sedang (20000-50000)":
            filtered = filtered[(filtered["Price"] >= 20000) & (filtered["Price"] <= 50000)]
        elif harga == "Mahal (>50000)":
            filtered = filtered[filtered["Price"] > 50000]

    # Pastikan kolom Outdoor/Indoor tidak kosong
    filtered = filtered[filtered["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]

    # Filter outdoor/indoor sesuai cuaca
    if cuaca_user:
        if cuaca_user in ["Cerah", "Cerah Berawan", "Berawan"]:
            filtered = filtered[filtered["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]
        else:
            filtered = filtered[filtered["Outdoor/Indoor"] == "Indoor"]

    # Urutkan berdasarkan jarak ke pusat kota
    pusat_lat, pusat_lon = None, None
    if lokasi_terkini in LOKASI_KOORDINAT and not filtered.empty:
        pusat_lat, pusat_lon = LOKASI_KOORDINAT[lokasi_terkini]
    if pusat_lat is not None and pusat_lon is not None:
        filtered["Jarak (km)"] = filtered.apply(
            lambda row: haversine(
                pusat_lat, pusat_lon,
                row.get("lat_decimal", np.nan), row.get("long_decimal", np.nan)
            ), axis=1
        )
        filtered = filtered.sort_values("Jarak (km)")
    return filtered

def get_hybrid_recommendations(df, username, lokasi_terkini, cuaca_user, kategori=None, harga=None):
    similar_users = get_similar_users(username)
    prefs_df = get_all_user_preferences()
    collaborative_ids = []
    if similar_users:
        prefs_df = prefs_df[prefs_df['username'].isin(similar_users)]
        # Selalu gunakan lokasi_terkini dari input user untuk filter city
        filter_lokasi = lokasi_terkini
        filter_cuaca = prefs_df['cuaca'].mode()[0] if not prefs_df['cuaca'].mode().empty else cuaca_user
        filter_kategori = prefs_df['kategori'].mode()[0] if 'kategori' in prefs_df and not prefs_df['kategori'].mode().empty else kategori
        filter_harga = prefs_df['harga'].mode()[0] if 'harga' in prefs_df and not prefs_df['harga'].mode().empty else harga
        collaborative = df.copy()
        if filter_lokasi:
            collaborative = collaborative[collaborative['City'] == filter_lokasi]
        if filter_kategori:
            collaborative = collaborative[collaborative["Category"] == filter_kategori]
        if filter_harga:
            if filter_harga == "Murah (<20000)":
                collaborative = collaborative[collaborative["Price"] < 20000]
            elif filter_harga == "Sedang (20000-50000)":
                collaborative = collaborative[(collaborative["Price"] >= 20000) & (collaborative["Price"] <= 50000)]
            elif filter_harga == "Mahal (>50000)":
                collaborative = collaborative[collaborative["Price"] > 50000]
        collaborative = collaborative[collaborative["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]
        if filter_cuaca:
            if filter_cuaca in ["Cerah", "Cerah Berawan", "Berawan"]:
                collaborative = collaborative[collaborative["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]
            else:
                collaborative = collaborative[collaborative["Outdoor/Indoor"] == "Indoor"]
        collaborative_ids = collaborative['Place_Name'].tolist()
    else:
        collaborative = pd.DataFrame()

    # Content-based: destinasi sesuai filter user saat ini
    content_based = get_content_based_recommendations_by_filter(df, lokasi_terkini, cuaca_user, kategori, harga)
    content_ids = content_based['Place_Name'].tolist() if not content_based.empty else []

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

    # Urutkan berdasarkan jarak ke pusat kota
    pusat_lat, pusat_lon = None, None
    if lokasi_terkini in LOKASI_KOORDINAT and not hybrid_df.empty:
        pusat_lat, pusat_lon = LOKASI_KOORDINAT[lokasi_terkini]
    if pusat_lat is not None and pusat_lon is not None and not hybrid_df.empty:
        hybrid_df["Jarak (km)"] = hybrid_df.apply(
            lambda row: haversine(
                pusat_lat, pusat_lon,
                row.get("lat_decimal", np.nan), row.get("long_decimal", np.nan)
            ), axis=1
        )
        hybrid_df = hybrid_df.sort_values("Jarak (km)")
    return hybrid_df

# --- UI Bagian Rekomendasi ---
if page == "Rekomendasi":
    if st.session_state.get("logged_in"):
        with st.sidebar:
            st.success(f"👋 Selamat Datang {st.session_state['username']}")
        col1, col2 = st.columns([8, 4])
        with col2:
            with st.expander(st.session_state['username']):
                if st.button("Logout"):
                    st.session_state["logged_in"] = False
                    st.session_state["username"] = ""
                    st.rerun()
                    
    st.markdown("<h2 style='text-align: center;'>Rekomendasi Destinasi Wisata</h2>", unsafe_allow_html=True)
    st.write("Pilih preferensi Anda:")

    lokasi_options = [""] + CITY_OPTIONS
    kategori_options = [""] + sorted(df["Category"].dropna().unique())
    harga_options = ["", "Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"]

    lokasi_terkini = st.selectbox(
        "Lokasi Terkini Anda",
        options=lokasi_options,
        key="lokasi_terkini"
    )
    kategori = st.selectbox(
        "Kategori",
        options=kategori_options,
        key="kategori"
    )
    harga = st.selectbox(
        "Harga Tiket",
        options=harga_options,
        key="harga"
    )

    # Ambil cuaca otomatis dari lokasi user
    cuaca_user = None
    if lokasi_terkini in LOKASI_KOORDINAT:
        lat, lon = LOKASI_KOORDINAT[lokasi_terkini]
        cuaca_data = get_open_meteo_weather(lat, lon)
        if cuaca_data:
            cuaca_user = cuaca_data['desc']
            st.info(f"Cuaca di kota Anda ({lokasi_terkini}): **{cuaca_user}**")
        else:
            st.warning("Gagal mengambil data cuaca kota Anda. Filter cuaca dinonaktifkan.")

    if not st.session_state.get("logged_in"):
        st.warning("Anda belum login. Rekomendasi hanya didasarkan pada konten saja, dan tidak ada preferensi pengguna.")

    if st.button("Rekomendasikan"):
        if st.session_state.get("logged_in"):
            similar_users = get_similar_users(st.session_state["username"])
            if similar_users:
                rekomendasi_df = get_hybrid_recommendations(
                    df_cb, st.session_state["username"], lokasi_terkini, cuaca_user, kategori if kategori else None, harga if harga else None
                )
            else:
                rekomendasi_df = get_content_based_recommendations_by_filter(
                    df_cb, lokasi_terkini, cuaca_user, kategori if kategori else None, harga if harga else None
                )
            save_user_preference(
                st.session_state["username"],
                lokasi_terkini,
                harga,
                cuaca_user,
                kategori
            )
        else:
            rekomendasi_df = get_content_based_recommendations_by_filter(
                df_cb, lokasi_terkini, cuaca_user, kategori if kategori else None, harga if harga else None
            )

        # Fallback: jika hasil kosong, tampilkan pesan error
        if rekomendasi_df.empty:
            st.error("Tidak ada destinasi yang sesuai dengan filter lokasi, kategori, atau harga yang Anda pilih. Silakan coba filter lain.")

        if not rekomendasi_df.empty:
            st.markdown("#### Rekomendasi Destinasi Wisata Terdekat:")
            items = rekomendasi_df.reset_index(drop=True)
            n_cols = 5
            for i in range(0, len(items), n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i + j
                    if idx < len(items):
                        row = items.iloc[idx]
                        # Gunakan Place_Id jika ada, jika tidak gunakan Place_Name
                        place_id = str(row['Place_Id']) if 'Place_Id' in row else row['Place_Name']
                        is_fav = False
                        if st.session_state.get("logged_in"):
                            is_fav = is_favorite(st.session_state["username"], place_id)
                        star = "⭐" if is_fav else "☆"
                        btn_key = f"fav_{place_id}_{i}_{j}"
                        with cols[j]:
                            # Tombol bintang
                            if st.session_state.get("logged_in"):
                                if st.button(star, key=btn_key):
                                    if is_fav:
                                        remove_favorite(st.session_state["username"], place_id)
                                        st.success(f"{row['Place_Name']} dihapus dari favorit.")
                                    else:
                                        add_favorite(st.session_state["username"], place_id)
                                        st.success(f"{row['Place_Name']} ditambahkan ke favorit.")
                                    st.experimental_rerun()
                            st.markdown(
                                f"""
                                <div style="border:1px solid #5555; border-radius:8px; padding:10px; margin-bottom:10px; background:#000000">
                                    <b>{row['Place_Name']}</b><br>
                                    <i>{row['City']}</i><br>
                                    <span>Kategori: {row['Category']}</span><br>
                                    <span>Harga: Rp{row['Price']:,.0f}</span><br>
                                    <span>Rating: {row['Rating']}</span><br>
                                    <span>Outdoor/Indoor: {row['Outdoor/Indoor']}</span><br>
                                    {"<span style='color:gold;'>⭐ Anda telah menambah ini ke daftar favorit anda</span>" if is_fav else ""}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            st.caption(f"Menampilkan {len(rekomendasi_df)} rekomendasi di sekitar {lokasi_terkini}")
        else:
            st.info("Tidak ada destinasi yang sesuai filter Anda.")