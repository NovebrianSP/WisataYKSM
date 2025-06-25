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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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

def rebuild_tfidf_model(df):
    """Rebuild the TF-IDF model using the current dataframe"""
    # Ensure Description column exists and has no NaN values
    df = df.copy()
    
    # Create combined text field for TF-IDF
    if "Description" in df.columns:
        # Clean and combine with other fields for better matching
        df["combined_text"] = df["Place_Name"] + " " + df["Category"].fillna("") + " " + df["Description"].fillna("")
    else:
        # If no description, use place name and category
        df["combined_text"] = df["Place_Name"] + " " + df["Category"].fillna("")
    
    # Create and fit the TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])
    
    # Save the updated models
    try:
        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
        print("TF-IDF model rebuilt and saved successfully")
    except Exception as e:
        print(f"Error saving TF-IDF model: {e}")
    
    return df, tfidf, tfidf_matrix

@st.cache_resource
def load_content_based():
    df = pd.read_csv(DATA_PATH)
    
    try:
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
            
        # Check if dimensions match
        if len(df) != tfidf_matrix.shape[0]:
            print("Rebuilding TF-IDF model due to dimension mismatch")
            df, tfidf, tfidf_matrix = rebuild_tfidf_model(df)
    except (FileNotFoundError, EOFError, pickle.PickleError) as e:
        # If loading fails, rebuild the model
        print(f"Building new TF-IDF model: {e}")
        df, tfidf, tfidf_matrix = rebuild_tfidf_model(df)
    
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

def get_content_based_recommendations_by_text(df, tfidf, tfidf_matrix, user_text, lokasi_terkini=None, cuaca_user=None, harga=None, top_n=50):
    """Get content-based recommendations using text similarity with descriptions"""
    # If empty text, return filtered results without similarity calculation
    if not user_text.strip():
        temp_df = df.copy()
        temp_df['similarity_score'] = 0.5  # Neutral score
        
        # Apply filters
        if lokasi_terkini:
            temp_df = temp_df[temp_df["City"] == lokasi_terkini]
        
        if harga:
            if harga == "Murah (<20000)":
                temp_df = temp_df[temp_df["Price"] < 20000]
            elif harga == "Sedang (20000-50000)":
                temp_df = temp_df[(temp_df["Price"] >= 20000) & (temp_df["Price"] <= 50000)]
            elif harga == "Mahal (>50000)":
                temp_df = temp_df[temp_df["Price"] > 50000]
        
        if cuaca_user:
            temp_df = temp_df[temp_df["Outdoor/Indoor"].notna()]
            if cuaca_user in ["Cerah", "Cerah Berawan", "Berawan"]:
                temp_df = temp_df[temp_df["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]
            else:
                temp_df = temp_df[temp_df["Outdoor/Indoor"] == "Indoor"]
                
        return temp_df
    
    # If dataframe and TF-IDF matrix lengths don't match, rebuild the TF-IDF model
    if len(df) != tfidf_matrix.shape[0]:
        df, tfidf, tfidf_matrix = rebuild_tfidf_model(df)
    
    # Transform user text with TF-IDF vectorizer
    user_vector = tfidf.transform([user_text])
    
    # Calculate cosine similarity between user text and all descriptions
    sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Add similarity scores to dataframe
    temp_df = df.copy()
    temp_df['similarity_score'] = sim_scores
    
    # Filter by location if specified
    if lokasi_terkini:
        temp_df = temp_df[temp_df["City"] == lokasi_terkini]
    
    # Filter by price if specified
    if harga:
        if harga == "Murah (<20000)":
            temp_df = temp_df[temp_df["Price"] < 20000]
        elif harga == "Sedang (20000-50000)":
            temp_df = temp_df[(temp_df["Price"] >= 20000) & (temp_df["Price"] <= 50000)]
        elif harga == "Mahal (>50000)":
            temp_df = temp_df[temp_df["Price"] > 50000]
    
    # Filter by weather
    if cuaca_user:
        temp_df = temp_df[temp_df["Outdoor/Indoor"].notna()]
        if cuaca_user in ["Cerah", "Cerah Berawan", "Berawan"]:
            temp_df = temp_df[temp_df["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]
        else:
            temp_df = temp_df[temp_df["Outdoor/Indoor"] == "Indoor"]
    
    # Sort by similarity score and get top N
    temp_df = temp_df.sort_values('similarity_score', ascending=False).head(top_n)
    
    return temp_df

def get_hybrid_recommendations(df, tfidf, tfidf_matrix, username, lokasi_terkini, cuaca_user, user_text=None, harga=None):
    similar_users = get_similar_users(username)
    prefs_df = get_all_user_preferences()
    collaborative_ids = []
    
    if similar_users:
        prefs_df = prefs_df[prefs_df['username'].isin(similar_users)]
        # Always use current location from user input for city filter
        filter_lokasi = lokasi_terkini
        filter_cuaca = prefs_df['cuaca'].mode()[0] if not prefs_df['cuaca'].mode().empty else cuaca_user
        filter_text = prefs_df['kategori'].mode()[0] if 'kategori' in prefs_df and not prefs_df['kategori'].mode().empty else user_text
        filter_harga = prefs_df['harga'].mode()[0] if 'harga' in prefs_df and not prefs_df['harga'].mode().empty else harga
        
        # Get collaborative filtering results
        collaborative = get_content_based_recommendations_by_text(
            df, tfidf, tfidf_matrix, filter_text if filter_text else "", 
            filter_lokasi, filter_cuaca, filter_harga
        )
        collaborative_ids = collaborative['Place_Name'].tolist() if not collaborative.empty else []
    else:
        collaborative = pd.DataFrame()

    # Content-based: destinations based on current user filters
    content_based = get_content_based_recommendations_by_text(
        df, tfidf, tfidf_matrix, user_text if user_text else "", 
        lokasi_terkini, cuaca_user, harga
    )
    content_ids = content_based['Place_Name'].tolist() if not content_based.empty else []

    # Combine collaborative and content-based results (prioritize those appearing in both)
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

    # Get destination data based on hybrid_ids order
    hybrid_df = df[df['Place_Name'].isin(hybrid_ids)]
    
    # Sort by similarity if content_based has similarity scores
    if 'similarity_score' in content_based.columns:
        hybrid_df = hybrid_df.merge(
            content_based[['Place_Name', 'similarity_score']], 
            on='Place_Name', how='left'
        )
        hybrid_df = hybrid_df.sort_values('similarity_score', ascending=False)

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
    harga_options = ["", "Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"]

    lokasi_terkini = st.selectbox(
        "Lokasi Terkini Anda",
        options=lokasi_options,
        key="lokasi_terkini"
    )
    
    search_text = st.text_input(
        "Masukkan kata kunci tentang tempat wisata yang Anda cari:",
        placeholder="Contoh: pantai, museum, sejarah, pemandangan indah, wisata alam...",
        key="search_text"
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
                    df_cb, tfidf, tfidf_matrix, st.session_state["username"], 
                    lokasi_terkini, cuaca_user, search_text if search_text else None, 
                    harga if harga else None
                )
            else:
                rekomendasi_df = get_content_based_recommendations_by_text(
                    df_cb, tfidf, tfidf_matrix, search_text if search_text else "", 
                    lokasi_terkini, cuaca_user, harga if harga else None
                )
            save_user_preference(
                st.session_state["username"],
                lokasi_terkini,
                harga,
                cuaca_user,
                search_text  # Save search text instead of kategori
            )
        else:
            rekomendasi_df = get_content_based_recommendations_by_text(
                df_cb, tfidf, tfidf_matrix, search_text if search_text else "", 
                lokasi_terkini, cuaca_user, harga if harga else None
            )

        # Fallback: jika hasil kosong, tampilkan pesan error
        if rekomendasi_df.empty:
            st.error("Tidak ada destinasi yang sesuai dengan filter lokasi, kata kunci, atau harga yang Anda pilih. Silakan coba filter lain.")

        if not rekomendasi_df.empty:
            # Filter to only show recommendations with meaningful similarity scores
            if 'similarity_score' in rekomendasi_df.columns:
                # Keep only recommendations with similarity above threshold
                filtered_rekomendasi_df = rekomendasi_df[rekomendasi_df['similarity_score'] > 0.05]
                
                # If we have matches after filtering, use those
                if not filtered_rekomendasi_df.empty:
                    rekomendasi_df = filtered_rekomendasi_df
                    st.success(f"Ditemukan {len(rekomendasi_df)} destinasi yang cocok dengan kata kunci Anda.")
                else:
                    st.warning("Tidak ada destinasi yang sangat cocok dengan kata kunci Anda. Menampilkan semua hasil.")
            
            st.markdown("#### Rekomendasi Destinasi Wisata:")
            items = rekomendasi_df.reset_index(drop=True)
            n_cols = 3  # Reduce from 5 to 3 for better readability
            for i in range(0, len(items), n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i + j
                    if idx < len(items):
                        row = items.iloc[idx]
                        with cols[j]:
                            # Get description if available
                            description = ""
                            if "Description" in row and pd.notna(row["Description"]):
                                description = str(row["Description"])
                            elif "combined_text" in row and pd.notna(row["combined_text"]):
                                description = str(row["combined_text"])
                            
                            # Completely strip all HTML tags and special characters
                            if description:
                                import re
                                # First, remove all HTML-like tags
                                description = re.sub(r'<[^>]*>', '', description)
                                # Then remove any special characters that might be problematic
                                description = re.sub(r'[^\w\s.,;:!?\-]', ' ', description)
                                # Clean up multiple spaces
                                description = re.sub(r'\s+', ' ', description).strip()
                                
                            # Truncate description for display
                            short_desc = ""
                            if description:
                                short_desc = (description[:100] + "...") if len(description) > 100 else description
                            
                            # Add similarity score if available
                            similarity_info = ""
                            if 'similarity_score' in row and row['similarity_score'] > 0.05:
                                match_percent = int(min(row['similarity_score'] * 100, 100))
                                similarity_info = f"Kecocokan: <span style='color:#4CAF50; font-weight:bold;'>{match_percent}%</span>"
                            
                            # Create a Streamlit card instead of HTML for more reliable rendering
                            st.markdown(f"##### {row['Place_Name']}")
                            st.markdown(f"*{row['City']}*")
                            st.write(f"**Kategori:** {row['Category']}")
                            st.write(f"**Harga:** Rp{row['Price']:,.0f}")
                            st.write(f"**Rating:** {row['Rating']}")
                            st.write(f"**Outdoor/Indoor:** {row['Outdoor/Indoor']}")
                            if similarity_info:
                                st.markdown(similarity_info, unsafe_allow_html=True)
                            if short_desc:
                                st.info(short_desc)
                            
                            # Add expander for full description
                            if description and len(description) > 100:
                                with st.expander("Lihat deskripsi lengkap"):
                                    st.write(description)
                            
                            # Add a separator
                            st.markdown("---")
                            
            st.caption(f"Menampilkan {len(rekomendasi_df)} rekomendasi di {lokasi_terkini if lokasi_terkini else 'semua lokasi'}")
        else:
            st.info("Tidak ada destinasi yang sesuai filter Anda.")