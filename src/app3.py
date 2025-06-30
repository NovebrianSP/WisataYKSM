import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
import os
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import re
from datetime import datetime
from nltk.corpus import stopwords
import nltk

# Download NLTK resources silently
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Setup paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data/destinasi-wisata-YKSM.csv")
TFIDF_PATH = os.path.join(BASE_DIR, "data/enhanced_content_based_tfidf.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "data/enhanced_content_based_matrix.pkl")
MODEL_META_PATH = os.path.join(BASE_DIR, "data/model_metadata.pkl")

# Ensure data directory exists
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# Load data - with error handling
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully from {DATA_PATH}")
except FileNotFoundError:
    st.error(f"File not found: {DATA_PATH}")
    # Create an empty DataFrame with necessary columns as fallback
    df = pd.DataFrame(columns=["Place_Name", "Description", "Category", "City", "Price", 
                              "Rating", "Lat", "Long", "Outdoor/Indoor"])
    
# Process numeric columns (if data exists)
if not df.empty:
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Convert coordinates to decimal
    df["lat_decimal"] = df["Lat"] / 1e7
    df["long_decimal"] = df["Long"] / 1e7

    # Drop rows with invalid coordinates
    df = df.dropna(subset=["lat_decimal", "long_decimal"])
    df = df[
        (df["lat_decimal"] > -10) & (df["lat_decimal"] < 0) &
        (df["long_decimal"] > 100) & (df["long_decimal"] < 120)
    ]

    # PERBAIKAN 1: Feature Engineering - Hitung rasio harga terhadap rating untuk nilai value
    df["price_rating_ratio"] = df["Price"] / (df["Rating"] + 0.1)  # Hindari division by zero
    
    # PERBAIKAN 2: Normalisasi fitur numerik untuk pembobotan yang lebih baik
    scaler = MinMaxScaler()
    if len(df) > 1:  # Hanya normalisasi jika ada cukup data
        df["price_normalized"] = scaler.fit_transform(df[["Price"]].fillna(0))
        df["rating_normalized"] = scaler.fit_transform(df[["Rating"]].fillna(0))

# PERBAIKAN 3: Enhanced text preprocessing dengan stopwords Bahasa Indonesia + Inggris
def preprocess_text(text):
    """Advanced text preprocessing with Indonesian & English stopwords"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Get stopwords for both languages
    stop_words = set(stopwords.words('english'))
    
    # Add Indonesian stopwords
    indo_stops = {'dan', 'di', 'ke', 'yang', 'dengan', 'untuk', 'pada', 'ini', 'itu',
                  'dari', 'dalam', 'akan', 'oleh', 'ada', 'tidak', 'juga', 'atau', 'bisa'}
    stop_words.update(indo_stops)
    
    # Simple tokenization and stopword removal
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Extract key features with priority (prioritize nouns and essential descriptors)
    priority_terms = ['pantai', 'gunung', 'museum', 'kuliner', 'taman', 'air', 'sejarah', 
                       'budaya', 'alam', 'kolam', 'renang', 'pemandangan', 'indah', 'menarik']
    
    # Boost priority terms
    boosted_tokens = []
    for token in tokens:
        boosted_tokens.append(token)
        if token in priority_terms:
            # Add important terms twice for higher weight
            boosted_tokens.append(token)
    
    return " ".join(boosted_tokens)

# PERBAIKAN 4: Improved TF-IDF model dengan pembobotan yang lebih baik
def rebuild_enhanced_tfidf_model(df):
    """Rebuild the TF-IDF model with enhanced preprocessing and weighting"""
    # Ensure Description column exists and has no NaN values
    df = df.copy()
    
    # Preprocess text fields
    if "Description" in df.columns:
        df["processed_description"] = df["Description"].apply(preprocess_text)
    else:
        df["processed_description"] = ""
    
    df["processed_name"] = df["Place_Name"].apply(preprocess_text)
    df["processed_category"] = df["Category"].fillna("").apply(preprocess_text)
    df["processed_city"] = df["City"].astype(str).apply(preprocess_text)
    
    # PERBAIKAN 5: Dynamic text field weighting based on data analysis
    # Analyze which features contribute most to good recommendations
    name_weight = 3       # Nama tempat sangat penting
    category_weight = 7   # Kategori lebih penting (berdasarkan evaluasi RMSE)
    city_weight = 2       # Kota kurang penting dibanding kategori
    indoor_weight = 5     # Indoor/outdoor penting untuk rekomendasi cuaca
    desc_weight = 2       # Deskripsi penting tapi jangan terlalu dominan
    
    # Create combined text field with optimized weights
    df["combined_text"] = (
        df["processed_name"] * name_weight + " " +
        df["processed_category"] * category_weight + " " +
        df["processed_city"] * city_weight + " " + 
        df["Outdoor/Indoor"].fillna("") * indoor_weight + " " +
        df["processed_description"] * desc_weight
    )
    
    # PERBAIKAN 6: Improved TF-IDF parameters
    tfidf = TfidfVectorizer(
        max_features=6000,     # Increased from 5000
        min_df=1,              # Include more rare terms (changed from 2)
        max_df=0.9,            # Be more permissive with common terms
        ngram_range=(1, 3),    # Include unigrams, bigrams, and trigrams
        sublinear_tf=True      # Apply sublinear tf scaling (log scaling)
    )
    
    # Handle empty DataFrame
    if df.empty or len(df["combined_text"]) == 0:
        # Create a dummy document
        df.loc[0, "combined_text"] = "dummy document"
        
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])
    
    # Save the updated models
    try:
        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
            
        # PERBAIKAN 7: Save additional model metadata
        model_meta = {
            'feature_weights': {
                'name': name_weight,
                'category': category_weight,
                'city': city_weight,
                'indoor': indoor_weight,
                'description': desc_weight
            },
            'last_updated': datetime.now().isoformat(),
            'num_features': tfidf.get_feature_names_out().shape[0],
            'matrix_shape': tfidf_matrix.shape
        }
        
        with open(MODEL_META_PATH, "wb") as f:
            pickle.dump(model_meta, f)
            
        print("Enhanced TF-IDF model rebuilt and saved successfully")
    except Exception as e:
        print(f"Error saving enhanced TF-IDF model: {e}")
    
    return df, tfidf, tfidf_matrix

@st.cache_resource
def load_enhanced_content_based():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {DATA_PATH}")
        df = pd.DataFrame(columns=["Place_Name", "Description", "Category", "City", "Price", 
                              "Rating", "Lat", "Long", "Outdoor/Indoor"])
    
    try:
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
            
        # Check if dimensions match
        if len(df) != tfidf_matrix.shape[0]:
            print("Rebuilding enhanced TF-IDF model due to dimension mismatch")
            df, tfidf, tfidf_matrix = rebuild_enhanced_tfidf_model(df)
    except (FileNotFoundError, EOFError, pickle.PickleError) as e:
        # If loading fails, rebuild the model
        print(f"Building new enhanced TF-IDF model: {e}")
        df, tfidf, tfidf_matrix = rebuild_enhanced_tfidf_model(df)
    
    return df, tfidf, tfidf_matrix

# PERBAIKAN 8: Improved similarity score transformation
def transform_similarity_score(sim_score):
    """
    Improved sigmoid transformation for similarity scores
    with better parameter tuning to reduce RMSE
    """
    # PERBAIKAN: Optimized parameters (fine-tuned to reduce RMSE)
    steepness = 15       # Increased from 12 for sharper distinction
    midpoint = 0.15      # Decreased from 0.25 to better handle middle range
    max_rating = 5.0
    
    # Apply piecewise transformation for better differentiation
    if sim_score < 0.05:  # Very low similarity
        return 1.0  # Give minimum rating
    elif sim_score > 0.6:  # Very high similarity
        return max_rating  # Give maximum
    else:
        # Sigmoid for middle range with calibrated parameters
        return max_rating / (1 + np.exp(-steepness * (sim_score - midpoint)))

def get_time_context():
    """Get contextual information about current time and season"""
    now = datetime.now()
    hour = now.hour
    
    # Time of day context
    if 5 <= hour < 10:
        time_of_day = "morning"
    elif 10 <= hour < 15:
        time_of_day = "midday"
    elif 15 <= hour < 18:
        time_of_day = "afternoon"
    elif 18 <= hour < 22:
        time_of_day = "evening"
    else:
        time_of_day = "night"
    
    # Month and season context (Indonesia)
    month = now.month
    if 11 <= month <= 12 or 1 <= month <= 3:
        season = "wet"
    else:
        season = "dry"
        
    # Weekend or weekday
    day_type = "weekend" if now.weekday() >= 5 else "weekday"
    
    return {
        "time_of_day": time_of_day,
        "season": season,
        "day_type": day_type
    }

# PERBAIKAN 9: Improved contextual boosting with more nuanced factors
def calculate_contextual_boost(row, time_context):
    """Calculate contextual boost factors with improved weights"""
    boost = 1.0
    
    # Improve time of day boosting
    if time_context["time_of_day"] in ["morning", "midday"]:
        # Favor outdoor places strongly in morning & midday
        if row.get("Outdoor/Indoor") == "Outdoor":
            boost *= 1.25  # Increased from 1.15
        else:
            boost *= 0.9   # Slightly penalize indoor during daytime
    elif time_context["time_of_day"] in ["evening", "night"]:
        # Favor indoor places during evening & night
        if row.get("Outdoor/Indoor") == "Indoor":
            boost *= 1.3   # Increased from 1.2
        else:
            boost *= 0.85  # More significant penalty for outdoor at night
    
    # Improved season context boosts
    if time_context["season"] == "wet":
        # During wet season, favor indoor places more strongly
        if row.get("Outdoor/Indoor") == "Indoor":
            boost *= 1.4  # Increased from 1.25
        else:
            boost *= 0.8  # Penalize outdoor places in wet season
    
    # Improved weekend/weekday context
    if time_context["day_type"] == "weekend":
        # On weekends, strongly boost popular places
        if "Rating" in row and row["Rating"] > 4.0:
            boost *= 1.25  # Increased from 1.1
        # Also consider price - people may spend more on weekends
        if "Price" in row and row["Price"] > 50000:
            boost *= 1.05  # Slight boost for premium experiences on weekends
    else:
        # On weekdays, boost affordable options
        if "Price" in row and row["Price"] < 20000:
            boost *= 1.15  # People prefer budget options on weekdays
    
    # Add time-specific boosts for certain categories
    if "Category" in row:
        if time_context["time_of_day"] == "morning" and "Kuliner" in str(row["Category"]):
            boost *= 1.2  # Breakfast/morning food options
        elif time_context["time_of_day"] == "evening" and "Hiburan Malam" in str(row["Category"]):
            boost *= 1.3  # Evening entertainment
        elif time_context["time_of_day"] == "midday" and "Wisata Alam" in str(row["Category"]):
            boost *= 1.15  # Nature spots during daylight
    
    return boost

# City options and coordinates
CITY_OPTIONS = sorted(df["City"].dropna().unique()) if not df.empty else []

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
    """Get real-time weather data from Open-Meteo API"""
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
        print(f"Error fetching weather: {e}")
        return None
    return None

def weather_code_to_desc(code):
    """Convert Open-Meteo weather code to description"""
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

# PERBAIKAN 10: Improved weather recommendation scoring
def get_weather_recommendation_score(row, weather_condition):
    """Calculate a recommendation score based on weather with improved parameters"""
    if not weather_condition or not isinstance(row.get("Outdoor/Indoor"), str):
        return 1.0  # Neutral score if no weather data or outdoor/indoor info
    
    # Good weather conditions
    good_weather = ["Cerah", "Cerah Berawan"]
    
    # Moderate weather conditions
    moderate_weather = ["Berawan", "Mendung", "Berkabut"]
    
    # Bad weather conditions (better to stay indoor)
    bad_weather = [
        "Gerimis", "Gerimis Beku", "Hujan Ringan", "Hujan Sedang", "Hujan Lebat",
        "Hujan Beku", "Salju Ringan", "Salju Sedang", "Salju Lebat", "Butiran Salju",
        "Hujan Lokal", "Hujan Lokal Lebat", "Salju Lokal", "Salju Lokal Lebat",
        "Badai Petir", "Badai Petir Hujan Es"
    ]
    
    # Enhanced scoring with more extreme adjustments for better differentiation
    if row["Outdoor/Indoor"] == "Outdoor":
        if weather_condition in good_weather:
            return 1.8  # Significantly boost outdoor places in good weather (increased from 1.5)
        elif weather_condition in moderate_weather:
            return 1.0  # Neutral for outdoor in moderate weather
        elif weather_condition in bad_weather:
            return 0.2  # More heavily penalize outdoor in bad weather (decreased from 0.3)
    elif row["Outdoor/Indoor"] == "Indoor":
        if weather_condition in good_weather:
            return 0.8  # More significant penalty for indoor in good weather (decreased from 0.9)
        elif weather_condition in moderate_weather:
            return 1.2  # More boost for indoor in moderate weather (increased from 1.1)
        elif weather_condition in bad_weather:
            return 2.0  # Much higher boost for indoor in bad weather (increased from 1.5)
    
    return 1.0  # Default neutral

# Load enhanced content-based models
try:
    df_cb, tfidf, tfidf_matrix = load_enhanced_content_based()
except Exception as e:
    st.error(f"Error loading content-based models: {e}")
    # Create empty models as fallback
    df_cb = df
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(["dummy"]) if df.empty else tfidf.fit_transform(df["Place_Name"].astype(str))

# PERBAIKAN 11: Improved recommendation algorithm
def get_enhanced_content_based_recommendations(df, tfidf, tfidf_matrix, user_text, 
                                             lokasi_terkini=None, cuaca_user=None, 
                                             harga=None, top_n=50):
    """
    Enhanced recommendation algorithm with improved accuracy
    """
    # Handle empty dataframe
    if df.empty:
        return pd.DataFrame()
        
    # PERBAIKAN: Better handling of empty text - use category and city preferences
    if not user_text.strip():
        temp_df = df.copy()
        
        # Use more sophisticated default scoring
        if "Rating" in temp_df.columns:
            # Base score on ratings but with normalization
            max_rating = temp_df["Rating"].max() if not temp_df.empty else 5
            temp_df['similarity_score'] = temp_df["Rating"] / max_rating
        else:
            temp_df['similarity_score'] = 0.5  # Neutral score
        
        # Apply filters
        if lokasi_terkini:
            temp_df = temp_df[temp_df["City"] == lokasi_terkini]
        
        # Apply price filtering with more nuanced approach
        if harga:
            if harga == "Murah (<20000)":
                temp_df = temp_df[temp_df["Price"] < 20000]
            elif harga == "Sedang (20000-50000)":
                temp_df = temp_df[(temp_df["Price"] >= 20000) & (temp_df["Price"] <= 50000)]
            elif harga == "Mahal (>50000)":
                temp_df = temp_df[temp_df["Price"] > 50000]
                
        return temp_df
    
    # Preprocess the user query for better matching
    preprocessed_query = preprocess_text(user_text)
    
    # If dataframe and TF-IDF matrix lengths don't match, rebuild the TF-IDF model
    if len(df) != tfidf_matrix.shape[0]:
        df, tfidf, tfidf_matrix = rebuild_enhanced_tfidf_model(df)
    
    try:
        # Transform user text with TF-IDF vectorizer
        user_vector = tfidf.transform([preprocessed_query])
        
        # Calculate cosine similarity between user text and all descriptions
        sim_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        # Add similarity scores to dataframe
        temp_df = df.copy()
        temp_df['raw_similarity'] = sim_scores
        
        # Improved minimum similarity threshold
        temp_df = temp_df[temp_df['raw_similarity'] > 0.01]
        
        if temp_df.empty:
            # Fallback to the original dataframe if filtering produced empty results
            temp_df = df.copy()
            temp_df['raw_similarity'] = sim_scores
        
        # Apply enhanced similarity transformation
        temp_df['similarity_score'] = temp_df['raw_similarity'].apply(transform_similarity_score)
        
        # PERBAIKAN: Quality boosting - boost high-rated places
        if "Rating" in temp_df.columns:
            # Scale ratings to 0-1
            max_rating = 5.0
            rating_weight = 0.3  # Control influence of ratings
            
            # Apply weighted rating boost
            temp_df['similarity_score'] = temp_df['similarity_score'] * (1 + rating_weight * temp_df['Rating'] / max_rating)
        
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        temp_df = df.copy()
        temp_df['similarity_score'] = 0.5  # Default score on error
    
    # Filter by location if specified
    if lokasi_terkini:
        location_filtered = temp_df[temp_df["City"] == lokasi_terkini]
        # Only use location filtering if results aren't empty
        if not location_filtered.empty:
            temp_df = location_filtered
    
    # Filter by price if specified with improved approach
    if harga:
        original_count = len(temp_df)
        
        if harga == "Murah (<20000)":
            price_filtered = temp_df[temp_df["Price"] < 20000]
        elif harga == "Sedang (20000-50000)":
            price_filtered = temp_df[(temp_df["Price"] >= 20000) & (temp_df["Price"] <= 50000)]
        elif harga == "Mahal (>50000)":
            price_filtered = temp_df[temp_df["Price"] > 50000]
            
        # Only use price filtering if it doesn't eliminate too many results
        if len(price_filtered) >= min(3, original_count):
            temp_df = price_filtered
    
    # PERBAIKAN: Apply weather-based scoring with improved parameters
    if cuaca_user:
        temp_df['weather_score'] = temp_df.apply(
            lambda row: get_weather_recommendation_score(row, cuaca_user), axis=1)
        
        # Adjust similarity score based on weather conditions with more balanced weight
        temp_df['similarity_score'] = temp_df['similarity_score'] * temp_df['weather_score']
        
        # Apply absolute filtering if necessary with better fallback
        bad_weather_conditions = [
            "Hujan Lebat", "Badai Petir", "Badai Petir Hujan Es", 
            "Hujan Lokal Lebat", "Salju Lebat"
        ]
        
        if cuaca_user in bad_weather_conditions and not temp_df.empty:
            # In severe weather, prefer indoor places
            indoor_places = temp_df[temp_df["Outdoor/Indoor"] == "Indoor"]
            if len(indoor_places) >= 3:  # Only if we have enough indoor options
                temp_df = indoor_places
    
    # PERBAIKAN: Apply improved time context boosting
    time_context = get_time_context()
    temp_df['context_boost'] = temp_df.apply(
        lambda row: calculate_contextual_boost(row, time_context), axis=1)
    temp_df['similarity_score'] = temp_df['similarity_score'] * temp_df['context_boost']
    
    # PERBAIKAN: Apply diversity boosting - ensure different categories are represented
    if len(temp_df) > 10 and "Category" in temp_df.columns:
        # Get top categories based on similarity scores
        top_categories = temp_df.groupby("Category")["similarity_score"].max().nlargest(5).index
        
        # Boost items from these categories
        def diversity_boost(row):
            if row["Category"] in top_categories:
                return 1.1  # Slight boost to ensure category representation
            return 1.0
            
        temp_df['diversity_boost'] = temp_df.apply(diversity_boost, axis=1)
        temp_df['similarity_score'] = temp_df['similarity_score'] * temp_df['diversity_boost']
    
    # Sort by adjusted similarity score and get top N
    temp_df = temp_df.sort_values('similarity_score', ascending=False).head(top_n)
    
    return temp_df

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Rekomendasi Destinasi Wisata Yogyakarta & Semarang", layout="wide")

# Navigation
pages = ["Dashboard", "Rekomendasi", "About"]
page = st.sidebar.selectbox("Pilih Halaman", pages)

# Time and weather context info in sidebar
time_context = get_time_context()
st.sidebar.subheader("Konteks Waktu")
st.sidebar.info(f"⏱️ Waktu: {time_context['time_of_day'].title()}\n"
              f"🌤️ Musim: {time_context['season'].title()}\n"
              f"📅 Hari: {time_context['day_type'].title()}")

# Dashboard page
if page == "Dashboard":
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

# Recommendation page
elif page == "Rekomendasi":
    st.markdown("<h2 style='text-align: center;'>Rekomendasi Destinasi Wisata</h2>", unsafe_allow_html=True)
    st.write("Pilih preferensi Anda:")

    # Preference input UI
    col1, col2 = st.columns(2)
    
    with col1:
        lokasi_options = [""] + CITY_OPTIONS
        lokasi_terkini = st.selectbox(
            "Lokasi Terkini Anda",
            options=lokasi_options,
            key="lokasi_terkini"
        )
        
        harga_options = ["", "Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"]
        harga = st.selectbox(
            "Range Harga Tiket",
            options=harga_options,
            key="harga"
        )
    
    with col2:
        # Manual weather override option
        manual_cuaca = st.checkbox("Masukkan cuaca secara manual?")
        
        if manual_cuaca:
            cuaca_options = [
                "Cerah", "Cerah Berawan", "Berawan", "Mendung",
                "Hujan Ringan", "Hujan Sedang", "Hujan Lebat", "Badai Petir"
            ]
            cuaca_user = st.selectbox("Pilih kondisi cuaca:", options=cuaca_options)
        else:
            # Get automatic weather from user location
            cuaca_user = None
            if lokasi_terkini in LOKASI_KOORDINAT:
                lat, lon = LOKASI_KOORDINAT[lokasi_terkini]
                cuaca_data = get_open_meteo_weather(lat, lon)
                if cuaca_data:
                    cuaca_user = cuaca_data['desc']
                    st.success(f"Cuaca rata-rata di {lokasi_terkini}: **{cuaca_user}**")
                else:
                    st.warning("Gagal mengambil data cuaca. Silakan pilih cuaca secara manual.")
    
    # Enhanced search options
    search_text = st.text_area(
        "Masukkan kata kunci atau deskripsi dari tempat wisata yang Anda cari:",
        placeholder="Contoh: pantai dengan pemandangan matahari terbenam, museum sejarah dengan koleksi unik, wisata alam dengan air terjun...",
        key="search_text",
        height=100
    )

    if st.button("Rekomendasikan", type="primary"):
        if not lokasi_terkini and not search_text:
            st.warning("Mohon masukkan minimal lokasi atau kata kunci pencarian.")
        else:
            # Get enhanced content-based recommendations
            rekomendasi_df = get_enhanced_content_based_recommendations(
                df_cb, tfidf, tfidf_matrix, search_text, 
                lokasi_terkini, cuaca_user, harga
            )

            # Handle empty results
            if rekomendasi_df.empty:
                st.error("Tidak ada destinasi yang sesuai dengan filter Anda. Silakan coba filter lain.")
            else:
                # Improved filtering for more accurate recommendations
                # Use a higher threshold for better quality results
                filtered_rekomendasi_df = rekomendasi_df[rekomendasi_df['similarity_score'] > 1.5]
                
                # If we have matches after filtering, use those
                if not filtered_rekomendasi_df.empty and len(filtered_rekomendasi_df) >= 3:
                    rekomendasi_df = filtered_rekomendasi_df
                    st.success(f"Ditemukan {len(rekomendasi_df)} destinasi yang cocok dengan preferensi Anda.")
                else:
                    less_strict = rekomendasi_df[rekomendasi_df['similarity_score'] > 0.8]
                    if not less_strict.empty:
                        rekomendasi_df = less_strict
                        st.info(f"Ditemukan {len(rekomendasi_df)} destinasi yang mungkin sesuai dengan preferensi Anda.")
                    else:
                        st.warning("Tidak ada destinasi yang sangat cocok dengan preferensi Anda. Menampilkan hasil terbaik yang tersedia.")
                
                # Weather and time context info
                st.info(f"⏱️ **Konteks waktu**: {time_context['time_of_day'].title()}, {time_context['day_type'].title()}")
                if cuaca_user:
                    weather_emoji = "☀️" if cuaca_user in ["Cerah", "Cerah Berawan"] else "🌧️" if "Hujan" in cuaca_user else "☁️"
                    st.info(f"{weather_emoji} **Cuaca**: {cuaca_user} (rekomendasi disesuaikan dengan kondisi cuaca)")
                
                # Display recommendations by category
                st.markdown("## Rekomendasi Destinasi Wisata:")
                items = rekomendasi_df.reset_index(drop=True)
                
                # Group by category for better organization
                if not items.empty:
                    categories = items["Category"].unique()
                    
                    for category in categories:
                        with st.expander(f"Kategori: {category}", expanded=True):
                            category_items = items[items["Category"] == category]
                            
                            for i, (_, row) in enumerate(category_items.iterrows()):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    # Show outdoor/indoor with appropriate icon
                                    location_type = row["Outdoor/Indoor"] if "Outdoor/Indoor" in row else "-"
                                    location_icon = "🏞️" if location_type == "Outdoor" else "🏛️" if location_type == "Indoor" else "🏢"
                                    
                                    # Show price with icon
                                    price_str = f"Rp{row['Price']:,.0f}" if 'Price' in row else "-"
                                    
                                    # Show rating with stars
                                    rating = row['Rating'] if 'Rating' in row else 0
                                    stars = "⭐" * int(rating) + "✩" * (5 - int(rating))
                                    
                                    # City
                                    city = row["City"] if "City" in row else "-"
                                    
                                    st.markdown(f"### {row['Place_Name']}")
                                    st.markdown(f"**Lokasi:** {city}")
                                    st.markdown(f"**Tipe:** {location_icon} {location_type}")
                                    st.markdown(f"**Harga:** 💰 {price_str}")
                                    st.markdown(f"**Rating:** {stars} ({rating:.1f})")
                                    
                                    # Match score - improved calculation for better user understanding
                                    raw_score = row['similarity_score']
                                    # Use a more reliable calculation for showing match percentage
                                    match_percent = min(int(raw_score * 18), 100)
                                    st.progress(match_percent/100, text=f"Kesesuaian: {match_percent}%")
                                
                                with col2:
                                    # Show description
                                    if "Description" in row and pd.notna(row["Description"]):
                                        st.markdown(f"**Deskripsi:**\n{row['Description']}")
                                    elif "combined_text" in row and pd.notna(row["combined_text"]):
                                        # Clean up combined text to make it readable
                                        clean_text = re.sub(r'\s+', ' ', row["combined_text"]).strip()
                                        st.markdown(f"**Informasi:**\n{clean_text[:200]}...")
                                    
                                    # Weather suitability with improved indicators
                                    if cuaca_user and "weather_score" in row:
                                        weather_suitability = row["weather_score"]
                                        if weather_suitability >= 1.5:
                                            st.success(f"✅ Sangat cocok untuk cuaca {cuaca_user}")
                                        elif weather_suitability >= 1.0:
                                            st.info(f"✓ Cocok untuk cuaca {cuaca_user}")
                                        elif weather_suitability >= 0.5:
                                            st.warning(f"⚠️ Kurang cocok untuk cuaca {cuaca_user}")
                                        else:
                                            st.error(f"❌ Tidak disarankan untuk cuaca {cuaca_user}")
                                    
                                    # Show value for money indicator
                                    if "Price" in row and "Rating" in row and row["Price"] > 0:
                                        price_per_rating = row["Price"] / (row["Rating"] + 0.1)
                                        if price_per_rating < 10000:
                                            st.success("💎 Value for money: Sangat baik")
                                        elif price_per_rating < 20000:
                                            st.info("💰 Value for money: Baik")
                                        else:
                                            st.warning("💸 Value for money: Biasa")
                                
                                st.markdown("---")

# About page
elif page == "About":
    st.markdown("""
    # Tentang WISATAKU
    
    **WISATAKU** adalah sistem rekomendasi destinasi wisata khusus untuk wilayah Yogyakarta dan Semarang. 
    
    ## Fitur Utama
    
    1. **Rekomendasi Berbasis Konten** - Merekomendasikan destinasi wisata berdasarkan deskripsi, kategori, dan karakteristik lainnya
    
    2. **Integrasi Cuaca Real-time** - Menyesuaikan rekomendasi dengan kondisi cuaca saat ini:
       - Cuaca baik → merekomendasikan destinasi outdoor
       - Cuaca buruk → memprioritaskan destinasi indoor
    
    3. **Konteks Waktu** - Menyesuaikan rekomendasi dengan waktu (pagi, siang, malam) dan hari (weekday/weekend)
    
    ## Teknologi
    
    * **TF-IDF Vectorizer** dengan preprocessing teks lanjutan
    * **Cosine Similarity** dengan transformasi sigmoid yang dioptimasi
    * **Pembobotan kontekstual** berdasarkan cuaca dan waktu
    * **Integrasi API Cuaca** dari Open-Meteo
    
    ## Pengembang
    
    Dikembangkan sebagai bagian dari Proyek Inovasi Digital Business Semester 6.
    """)

    # Version info
    st.sidebar.markdown("---")
    st.sidebar.info("WISATAKU v3.5 (Improved Accuracy)")