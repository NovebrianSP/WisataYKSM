import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🏛️ Tourism Recommendation System",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add JavaScript for device detection
st.markdown("""
<script>
function detectDevice() {
    const isMobile = window.innerWidth <= 768 || /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    return isMobile;
}

// Auto-adjust layout based on screen size
window.addEventListener('resize', function() {
    const isMobile = window.innerWidth <= 768;
    document.body.classList.toggle('mobile-view', isMobile);
});

// Initial detection
document.addEventListener('DOMContentLoaded', function() {
    const isMobile = detectDevice();
    document.body.classList.toggle('mobile-view', isMobile);
});
</script>
""", unsafe_allow_html=True)

# Custom CSS for responsive design
st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding: 1rem;
    }
    
    /* Headers - Responsive */
    .main-header {
        font-size: clamp(2rem, 5vw, 3rem);
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f77b4, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: clamp(1.2rem, 3vw, 1.5rem);
        color: #ff7f0e;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Cards - Responsive */
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .recommendation-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }
    
    .weather-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: clamp(1rem, 3vw, 1.5rem);
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(116, 185, 255, 0.2);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .weather-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: clamp(1.2rem, 3vw, 1.5rem);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: clamp(0.8rem, 2.5vw, 1.2rem);
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Responsive Layout */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        
        .stColumns > div {
            padding: 0.25rem !important;
        }
        
        .recommendation-card {
            margin: 0.5rem 0;
        }
        
        /* Hide complex charts on mobile */
        .plotly-graph-div {
            min-height: 300px !important;
        }
        
        /* Stack columns on mobile */
        .row-widget.stColumns {
            flex-direction: column;
        }
        
        .row-widget.stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            margin-bottom: 1rem;
        }
        
        .weather-card {
            padding: 0.8rem;
        }
        
        .metric-card {
            padding: 0.6rem;
        }
        
        /* Smaller text on very small screens */
        .stMarkdown {
            font-size: 0.9rem;
        }
        
        /* Compact metrics */
        [data-testid="metric-container"] {
            padding: 0.5rem;
        }
    }
    
    /* Tablet styles */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 0.75rem;
        }
        
        .recommendation-card {
            padding: 1.2rem;
        }
    }
    
    /* Large screen optimizations */
    @media (min-width: 1200px) {
        .main {
            max-width: 1200px;
            margin: 0 auto;
        }
    }
    
    /* Sidebar Responsive */
    .css-1d391kg {
        padding: 1rem;
    }
    
    @media (max-width: 768px) {
        .css-1d391kg {
            padding: 0.5rem;
        }
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
    }
    
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Warning Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f3f4 0%, #e8eaed 100%);
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Chart Container */
    .plotly-graph-div {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    /* Loading Spinner */
    .stSpinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        border-top: 3px solid #1f77b4;
    }
    
    /* Responsive Table */
    .dataframe {
        font-size: clamp(0.8rem, 2vw, 1rem);
    }
    
    @media (max-width: 768px) {
        .dataframe {
            font-size: 0.7rem;
        }
    }
    
    /* Container Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Manages SQLite database operations for user authentication and preferences"""
    
    def __init__(self, db_path="tourism_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                location TEXT NOT NULL,
                age INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create user ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                place_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, place_id)
            )
        ''')
        
        # Create user preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                preference_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, category)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, full_name, location, age):
        """Create new user account"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, location, age)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, location, age))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return True, user_id
        except sqlite3.IntegrityError as e:
            conn.close()
            if 'username' in str(e):
                return False, "Username sudah digunakan"
            elif 'email' in str(e):
                return False, "Email sudah terdaftar"
            else:
                return False, "Error saat membuat akun"
        except Exception as e:
            conn.close()
            return False, f"Error: {str(e)}"
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, username, email, full_name, location, age
                FROM users 
                WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            
            user = cursor.fetchone()
            
            if user:
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (user[0],))
                conn.commit()
                
                user_data = {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'full_name': user[3],
                    'location': user[4],
                    'age': user[5]
                }
                conn.close()
                return True, user_data
            else:
                conn.close()
                return False, "Username atau password salah"
                
        except Exception as e:
            conn.close()
            return False, f"Error: {str(e)}"
    
    def add_user_rating(self, user_id, place_id, rating):
        """Add or update user rating for a place"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_ratings (user_id, place_id, rating)
                VALUES (?, ?, ?)
            ''', (user_id, place_id, rating))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            conn.close()
            return False
    
    def get_user_ratings(self, user_id):
        """Get all ratings by a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT place_id, rating FROM user_ratings
                WHERE user_id = ?
            ''', (user_id,))
            
            ratings = cursor.fetchall()
            conn.close()
            
            return pd.DataFrame(ratings, columns=['Place_Id', 'Place_Ratings'])
        except Exception as e:
            conn.close()
            return pd.DataFrame()
    
    def get_all_user_ratings(self):
        """Get all user ratings for collaborative filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id as User_Id, place_id as Place_Id, rating as Place_Ratings
                FROM user_ratings
            ''')
            
            ratings = cursor.fetchall()
            conn.close()
            
            return pd.DataFrame(ratings, columns=['User_Id', 'Place_Id', 'Place_Ratings'])
        except Exception as e:
            conn.close()
            return pd.DataFrame()
    
    def update_user_preferences(self, user_id, category, preference_score=1.0):
        """Update user preferences based on ratings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences (user_id, category, preference_score)
                VALUES (?, ?, ?)
            ''', (user_id, category, preference_score))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            conn.close()
            return False
    
    def get_user_preferences(self, user_id):
        """Get user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT category, preference_score FROM user_preferences
                WHERE user_id = ?
                ORDER BY preference_score DESC
            ''', (user_id,))
            
            preferences = cursor.fetchall()
            conn.close()
            
            return [pref[0] for pref in preferences if pref[1] > 0.5]
        except Exception as e:
            conn.close()
            return []
    
    def get_user_by_id(self, user_id):
        """Get user information by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, email, full_name, location, age
                FROM users 
                WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            conn.close()
            
            if user:
                return {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'full_name': user[3],
                    'location': user[4],
                    'age': user[5]
                }
            return None
        except Exception as e:
            conn.close()
            return None

def show_auth_page():
    """Show authentication page (login/register)"""
    st.markdown('<h1 class="main-header">🏛️ Tourism Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize database
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Authentication tabs
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.markdown("### 🔑 Login ke Akun Anda")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Masukkan username Anda")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password Anda")
            
            login_btn = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
            
            if login_btn:
                if username and password:
                    success, result = st.session_state.db_manager.authenticate_user(username, password)
                    
                    if success:
                        st.session_state.user = result
                        st.session_state.authenticated = True
                        st.success(f"✅ Selamat datang, {result['full_name']}!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ Harap isi semua field")
    
    with tab2:
        st.markdown("### 📝 Buat Akun Baru")
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                reg_username = st.text_input("👤 Username", placeholder="Pilih username unik")
                reg_email = st.text_input("📧 Email", placeholder="alamat@email.com")
                reg_password = st.text_input("🔒 Password", type="password", placeholder="Minimal 6 karakter")
            
            with col2:
                reg_full_name = st.text_input("👨‍👩‍👧‍👦 Nama Lengkap", placeholder="Nama lengkap Anda")
                reg_location = st.selectbox("📍 Lokasi", [
                    "Jakarta, DKI Jakarta",
                    "Bandung, Jawa Barat",
                    "Surabaya, Jawa Timur",
                    "Yogyakarta, DIY",
                    "Semarang, Jawa Tengah",
                    "Bekasi, Jawa Barat",
                    "Bogor, Jawa Barat",
                    "Depok, Jawa Barat",
                    "Tangerang, Banten",
                    "Palembang, Sumatera Selatan"
                ])
                reg_age = st.number_input("🎂 Umur", min_value=13, max_value=100, value=25)
            
            register_btn = st.form_submit_button("✨ Daftar Akun", type="primary", use_container_width=True)
            
            if register_btn:
                if all([reg_username, reg_email, reg_password, reg_full_name, reg_location, reg_age]):
                    if len(reg_password) < 6:
                        st.error("❌ Password minimal 6 karakter")
                    elif "@" not in reg_email:
                        st.error("❌ Format email tidak valid")
                    else:
                        success, result = st.session_state.db_manager.create_user(
                            reg_username, reg_email, reg_password, reg_full_name, reg_location, reg_age
                        )
                        
                        if success:
                            st.success("✅ Akun berhasil dibuat! Silakan login.")
                        else:
                            st.error(f"❌ {result}")
                else:
                    st.warning("⚠️ Harap isi semua field")

class WeatherService:
    """Service untuk mengambil data cuaca dari Open-Meteo API"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
    
    def get_weather(self, latitude, longitude):
        """Ambil data cuaca berdasarkan koordinat"""
        try:
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'current_weather': 'true',
                'daily': 'precipitation_sum,temperature_2m_max,temperature_2m_min',
                'timezone': 'Asia/Jakarta'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current_weather = data.get('current_weather', {})
            daily = data.get('daily', {})
            
            weather_code = current_weather.get('weathercode', 0)
            temperature = current_weather.get('temperature', 25)
            precipitation = daily.get('precipitation_sum', [0])[0] if daily.get('precipitation_sum') else 0
            
            # Weather interpretation
            weather_desc = self._get_weather_description(weather_code)
            
            return {
                'temperature': temperature,
                'precipitation': precipitation,
                'is_sunny': weather_code in [0, 1, 2],
                'is_rainy': weather_code in [61, 63, 65, 80, 81, 82],
                'weather_code': weather_code,
                'description': weather_desc,
                'icon': self._get_weather_icon(weather_code)
            }
        except Exception as e:
            st.warning(f"Tidak dapat mengambil data cuaca: {e}")
            return {
                'temperature': 25,
                'precipitation': 0,
                'is_sunny': True,
                'is_rainy': False,
                'weather_code': 0,
                'description': 'Cerah',
                'icon': '☀️'
            }
    
    def _get_weather_description(self, code):
        """Convert weather code to description"""
        weather_codes = {
            0: "Cerah",
            1: "Sebagian Berawan",
            2: "Berawan",
            3: "Mendung",
            45: "Berkabut",
            48: "Berkabut Tebal",
            51: "Gerimis Ringan",
            53: "Gerimis Sedang",
            55: "Gerimis Lebat",
            61: "Hujan Ringan",
            63: "Hujan Sedang",
            65: "Hujan Lebat",
            80: "Hujan Deras",
            81: "Hujan Sangat Deras",
            82: "Hujan Ekstrem"
        }
        return weather_codes.get(code, "Tidak Diketahui")
    
    def _get_weather_icon(self, code):
        """Get emoji icon for weather"""
        if code in [0]:
            return "☀️"
        elif code in [1, 2]:
            return "⛅"
        elif code in [3]:
            return "☁️"
        elif code in [45, 48]:
            return "🌫️"
        elif code in [51, 53, 55, 61, 63, 65]:
            return "🌧️"
        elif code in [80, 81, 82]:
            return "⛈️"
        else:
            return "🌤️"

class ContentBasedFilter:
    """Content-based filtering berdasarkan fitur destinasi wisata"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_matrix = None
        self.places_df = None
        self.scaler = StandardScaler()
    
    def fit(self, places_df):
        """Train content-based model"""
        self.places_df = places_df.copy()
        
        # Combine text features
        text_features = (
            places_df['Place_Name'].fillna('') + ' ' +
            places_df['Description'].fillna('') + ' ' +
            places_df['Category'].fillna('') + ' ' +
            places_df['City'].fillna('')
        )
        
        # TF-IDF for text features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        # Numerical features
        numerical_features = places_df[['Price', 'Rating', 'Rating_Count']].fillna(0)
        numerical_scaled = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        self.feature_matrix = np.hstack([tfidf_matrix.toarray(), numerical_scaled])
        
        return self
    
    def recommend_by_category(self, preferred_categories, weather_condition, top_k=10):
        """Rekomendasi berdasarkan kategori dan cuaca"""
        filtered_places = self.places_df.copy()
        
        # Filter berdasarkan kategori
        if preferred_categories:
            filtered_places = filtered_places[
                filtered_places['Category'].isin(preferred_categories)
            ]
        
        # Filter berdasarkan cuaca
        if weather_condition['is_rainy']:
            indoor_places = filtered_places[filtered_places['Outdoor/Indoor'] == 'Indoor']
            if not indoor_places.empty:
                filtered_places = indoor_places
        elif weather_condition['temperature'] > 32:
            cool_places = filtered_places[
                (filtered_places['Category'].isin(['Bahari', 'Cagar Alam'])) |
                (filtered_places['Outdoor/Indoor'] == 'Indoor')
            ]
            if not cool_places.empty:
                filtered_places = cool_places
        
        return filtered_places.nlargest(top_k, 'Rating')

class CollaborativeFilter:
    """Collaborative filtering menggunakan matrix factorization dengan integrasi database"""
    
    def __init__(self, n_components=50, db_manager=None):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.db_manager = db_manager
        self.csv_ratings_df = None
    
    def fit(self, ratings_df, db_manager=None):
        """Train collaborative filtering model with both CSV and database data"""
        self.csv_ratings_df = ratings_df
        self.db_manager = db_manager
        
        # Combine CSV ratings with database ratings
        combined_ratings = ratings_df.copy()
        
        if db_manager:
            db_ratings = db_manager.get_all_user_ratings()
            if not db_ratings.empty:
                # Offset database user IDs to avoid conflicts with CSV data
                max_csv_user_id = ratings_df['User_Id'].max() if not ratings_df.empty else 0
                db_ratings['User_Id'] = db_ratings['User_Id'] + max_csv_user_id
                
                # Combine datasets
                combined_ratings = pd.concat([ratings_df, db_ratings], ignore_index=True)
        
        # Create user-item matrix
        self.user_item_matrix = combined_ratings.pivot_table(
            index='User_Id', 
            columns='Place_Id', 
            values='Place_Ratings',
            fill_value=0
        )
        
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.item_mapping = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        
        # Apply SVD
        matrix_values = self.user_item_matrix.values
        self.svd.fit(matrix_values)
        
        # Get user and item factors
        self.user_factors = self.svd.transform(matrix_values)
        self.item_factors = self.svd.components_.T
        
        return self
    
    def predict_rating(self, user_id, place_id):
        """Prediksi rating untuk user dan place tertentu"""
        try:
            user_idx = self.user_mapping.get(user_id)
            item_idx = self.item_mapping.get(place_id)
            
            if user_idx is None or item_idx is None:
                return 3.0
            
            predicted = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            return np.clip(predicted, 1, 5)
        except:
            return 3.0
    
    def recommend_for_user(self, user_id, places_df, top_k=10, exclude_rated=True):
        """Rekomendasi untuk user tertentu dengan integrasi database"""
        try:
            # Check if this is a database user (authenticated user)
            is_db_user = user_id > 1000  # Assuming DB users have high IDs
            
            if is_db_user:
                # Get actual database user ID
                actual_user_id = user_id - (self.csv_ratings_df['User_Id'].max() if not self.csv_ratings_df.empty else 0)
                
                # Get user ratings from database
                if self.db_manager:
                    user_ratings_df = self.db_manager.get_user_ratings(actual_user_id)
                    rated_places = set(user_ratings_df['Place_Id'].tolist()) if not user_ratings_df.empty else set()
                else:
                    rated_places = set()
            else:
                # CSV user
                user_idx = self.user_mapping.get(user_id)
                if user_idx is None:
                    return places_df.nlargest(top_k, 'Rating')
                
                # Get rated places for this user
                rated_places = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
            
            # Predict ratings for all places
            recommendations = []
            for place_id in places_df['Place_Id']:
                if exclude_rated and place_id in rated_places:
                    continue
                
                predicted_rating = self.predict_rating(user_id, place_id)
                place_info = places_df[places_df['Place_Id'] == place_id].iloc[0]
                
                recommendations.append({
                    'Place_Id': place_id,
                    'predicted_rating': predicted_rating,
                    'Place_Name': place_info['Place_Name'],
                    'Category': place_info['Category'],
                    'City': place_info['City'],
                    'Rating': place_info['Rating'],
                    'Price': place_info['Price']
                })
            
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            return pd.DataFrame(recommendations[:top_k])
        except Exception as e:
            return places_df.nlargest(top_k, 'Rating')

class LocationBasedFilter:
    """Filter berdasarkan lokasi geografis"""
    
    @staticmethod
    def calculate_distance(user_location, places_df):
        """Hitung jarak dari lokasi user ke destinasi wisata"""
        distances = []
        user_coords = (user_location['lat'], user_location['lng'])
        
        for _, place in places_df.iterrows():
            try:
                place_coords = (place['Lat'] / 10000000, place['Long'] / 10000000)
                distance = geodesic(user_coords, place_coords).kilometers
                distances.append(distance)
            except:
                distances.append(float('inf'))
        
        places_with_distance = places_df.copy()
        places_with_distance['distance_km'] = distances
        
        return places_with_distance

class HybridTourismRecommendationSystem:
    """Sistem rekomendasi hybrid untuk wisata dengan integrasi database"""
    
    def __init__(self, db_manager=None):
        self.weather_service = WeatherService()
        self.content_filter = ContentBasedFilter()
        self.collaborative_filter = CollaborativeFilter(db_manager=db_manager)
        self.location_filter = LocationBasedFilter()
        self.db_manager = db_manager
        
        self.users_df = None
        self.places_df = None
        self.ratings_df = None
        
        # Weights for hybrid approach
        self.weights = {
            'content': 0.3,
            'collaborative': 0.4,
            'popularity': 0.2,
            'distance': 0.1
        }
    
    def load_data(self, users_df, places_df, ratings_df):
        """Load dataset"""
        self.users_df = users_df
        self.places_df = places_df
        self.ratings_df = ratings_df
        return True
    
    def train_models(self):
        """Train semua model dengan data gabungan"""
        with st.spinner('🔄 Training machine learning models...'):
            self.content_filter.fit(self.places_df)
            self.collaborative_filter.fit(self.ratings_df, self.db_manager)
        return True
    
    def get_user_location(self, user_id, is_authenticated_user=False):
        """Dapatkan lokasi user dari data atau database"""
        try:
            if is_authenticated_user and self.db_manager:
                # Get from database
                user = self.db_manager.get_user_by_id(user_id)
                if user:
                    location_str = user['location']
                else:
                    return {'lat': -7.7956, 'lng': 110.3695}  # Default
            else:
                # Get from CSV
                user = self.users_df[self.users_df['User_Id'] == user_id].iloc[0]
                location_str = user['Location']
            
            location_mapping = {
                'Yogyakarta': {'lat': -7.7956, 'lng': 110.3695},
                'Semarang': {'lat': -6.9667, 'lng': 110.4167},
                'Jakarta': {'lat': -6.2088, 'lng': 106.8456},
                'Bandung': {'lat': -6.9175, 'lng': 107.6191},
                'Surabaya': {'lat': -7.2575, 'lng': 112.7521},
                'Bogor': {'lat': -6.5944, 'lng': 106.7892},
                'Bekasi': {'lat': -6.2383, 'lng': 106.9756},
                'Depok': {'lat': -6.4025, 'lng': 106.7942},
                'Tangerang': {'lat': -6.1783, 'lng': 106.6319},
                'Palembang': {'lat': -2.9761, 'lng': 104.7754}
            }
            
            for city in location_mapping:
                if city in location_str:
                    return location_mapping[city]
            
            return location_mapping['Yogyakarta']
        except:
            return {'lat': -7.7956, 'lng': 110.3695}
    
    def get_user_preferences(self, user_id, is_authenticated_user=False):
        """Dapatkan preferensi user berdasarkan riwayat rating"""
        try:
            if is_authenticated_user and self.db_manager:
                # Get from database preferences
                preferences = self.db_manager.get_user_preferences(user_id)
                if preferences:
                    return preferences
                
                # If no preferences, infer from ratings
                user_ratings_df = self.db_manager.get_user_ratings(user_id)
                if user_ratings_df.empty:
                    return []
                
                high_rated = user_ratings_df[user_ratings_df['Place_Ratings'] >= 4]
                if high_rated.empty:
                    return []
                
                liked_places = self.places_df[self.places_df['Place_Id'].isin(high_rated['Place_Id'])]
                preferred_categories = liked_places['Category'].unique().tolist()
                
                # Update preferences in database
                for category in preferred_categories:
                    self.db_manager.update_user_preferences(user_id, category, 1.0)
                
                return preferred_categories
            else:
                # Get from CSV
                user_ratings = self.ratings_df[self.ratings_df['User_Id'] == user_id]
                if user_ratings.empty:
                    return []
                
                high_rated = user_ratings[user_ratings['Place_Ratings'] >= 4]
                if high_rated.empty:
                    return []
                
                liked_places = self.places_df[self.places_df['Place_Id'].isin(high_rated['Place_Id'])]
                preferred_categories = liked_places['Category'].unique().tolist()
                
                return preferred_categories
        except:
            return []
    
    def recommend(self, user_id, top_k=10, max_distance_km=50, is_authenticated_user=False):
        """Generate hybrid recommendations"""
        # Get user location and weather
        user_location = self.get_user_location(user_id, is_authenticated_user)
        weather = self.weather_service.get_weather(user_location['lat'], user_location['lng'])
        
        # Get user preferences
        user_preferences = self.get_user_preferences(user_id, is_authenticated_user)
        
        # Content-based recommendations
        content_recs = self.content_filter.recommend_by_category(
            user_preferences, weather, top_k=top_k*2
        )
        
        # Collaborative filtering recommendations
        # For authenticated users, use offset user ID for collaborative filtering
        collab_user_id = user_id
        if is_authenticated_user:
            max_csv_user_id = self.ratings_df['User_Id'].max() if not self.ratings_df.empty else 0
            collab_user_id = user_id + max_csv_user_id
        
        collab_recs = self.collaborative_filter.recommend_for_user(
            collab_user_id, self.places_df, top_k=top_k*2
        )
        
        # Add location information
        places_with_distance = self.location_filter.calculate_distance(
            user_location, self.places_df
        )
        
        # Filter by distance
        nearby_places = places_with_distance[places_with_distance['distance_km'] <= max_distance_km]
        
        # Hybrid scoring
        recommendations = self._hybrid_score(
            content_recs, collab_recs, nearby_places, weather, top_k
        )
        
        return recommendations, weather, user_location, user_preferences
    
    def _hybrid_score(self, content_recs, collab_recs, nearby_places, weather, top_k):
        """Calculate hybrid score combining all methods"""
        scores = {}
        
        # Content-based scores
        for _, place in content_recs.iterrows():
            place_id = place['Place_Id']
            scores[place_id] = scores.get(place_id, 0) + self.weights['content']
        
        # Collaborative filtering scores
        if 'predicted_rating' in collab_recs.columns and not collab_recs.empty:
            max_pred_rating = collab_recs['predicted_rating'].max()
            for _, place in collab_recs.iterrows():
                place_id = place['Place_Id']
                normalized_score = place['predicted_rating'] / max_pred_rating
                scores[place_id] = scores.get(place_id, 0) + self.weights['collaborative'] * normalized_score
        
        # Popularity score
        for _, place in self.places_df.iterrows():
            place_id = place['Place_Id']
            popularity_score = (place['Rating'] / 5.0) * (np.log1p(place['Rating_Count']) / 10)
            scores[place_id] = scores.get(place_id, 0) + self.weights['popularity'] * popularity_score
        
        # Distance score
        if not nearby_places.empty:
            max_distance = nearby_places['distance_km'].max()
            for _, place in nearby_places.iterrows():
                place_id = place['Place_Id']
                distance_score = 1 - (place['distance_km'] / max_distance) if max_distance > 0 else 1
                scores[place_id] = scores.get(place_id, 0) + self.weights['distance'] * distance_score
        
        # Weather bonus
        for _, place in self.places_df.iterrows():
            place_id = place['Place_Id']
            if weather['is_rainy'] and place['Outdoor/Indoor'] == 'Indoor':
                scores[place_id] = scores.get(place_id, 0) + 0.1
            elif weather['is_sunny'] and place['Outdoor/Indoor'] == 'Outdoor':
                scores[place_id] = scores.get(place_id, 0) + 0.1
        
        # Sort by score and return top k
        sorted_places = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_place_ids = [place_id for place_id, score in sorted_places[:top_k]]
        
        # Get place details
        recommendations = self.places_df[self.places_df['Place_Id'].isin(top_place_ids)].copy()
        
        # Add scores and additional info
        score_dict = dict(sorted_places)
        recommendations['recommendation_score'] = recommendations['Place_Id'].map(score_dict)
        
        # Add distance info if available
        if not nearby_places.empty:
            distance_dict = nearby_places.set_index('Place_Id')['distance_km'].to_dict()
            recommendations['distance_km'] = recommendations['Place_Id'].map(distance_dict)
        
        # Sort by recommendation score
        recommendations = recommendations.sort_values('recommendation_score', ascending=False)
        
        return recommendations.head(top_k)

@st.cache_data
def load_datasets():
    """Load all datasets with caching"""
    try:
        users_df = pd.read_csv("dataset/user.csv")
        places_df = pd.read_csv("dataset/destinasi-wisata-YKSM.csv")
        ratings_df = pd.read_csv("dataset/tour_rating.csv")
        return users_df, places_df, ratings_df, True
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None, None, False

def create_responsive_recommendation_card(place, idx, weather, show_rating_option=False, db_manager=None, user_id=None):
    """Create a responsive recommendation card with rating option"""
    with st.container():
        # Mobile-first approach - single column on small screens
        if st.session_state.get('mobile_view', False):
            # Mobile layout
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>🏛️ {idx}. {place['Place_Name']}</h3>
                <p><strong>📍 {place['City']} | 🏷️ {place['Category']}</strong></p>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;">
                    <div>⭐ {place['Rating']}/5</div>
                    <div>💰 Rp {place['Price']:,}</div>
                    {'<div>🎯 ' + f"{(place['recommendation_score'] * 100):.1f}%" + '</div>' if 'recommendation_score' in place else ''}
                </div>
            """, unsafe_allow_html=True)
            
            # Description
            description = place['Description'][:150] + "..." if len(place['Description']) > 150 else place['Description']
            st.markdown(f"📝 {description}")
            
            # Tags and distance
            tags = f"🏠 {place['Outdoor/Indoor']}"
            if 'distance_km' in place and pd.notna(place['distance_km']):
                tags += f" | 📍 {place['distance_km']:.1f} km"
            st.markdown(tags)
            
            # Weather suitability
            if weather['is_rainy'] and place['Outdoor/Indoor'] == 'Indoor':
                st.success("☔ Cocok untuk cuaca hujan")
            elif weather['is_sunny'] and place['Outdoor/Indoor'] == 'Outdoor':
                st.success("☀️ Cocok untuk cuaca cerah")
            
            # Rating option for authenticated users
            if show_rating_option and db_manager and user_id:
                st.markdown("### ⭐ Beri Rating")
                user_rating = st.slider(
                    f"Rating untuk {place['Place_Name']}", 
                    1, 5, 3, 
                    key=f"rating_{place['Place_Id']}_{idx}"
                )
                
                if st.button(f"Submit Rating", key=f"submit_{place['Place_Id']}_{idx}"):
                    if db_manager.add_user_rating(user_id, place['Place_Id'], user_rating):
                        st.success("✅ Rating berhasil disimpan!")
                        
                        # Update user preferences based on rating
                        if user_rating >= 4:
                            db_manager.update_user_preferences(user_id, place['Category'], 1.0)
                        elif user_rating <= 2:
                            db_manager.update_user_preferences(user_id, place['Category'], 0.3)
                        
                        st.rerun()
                    else:
                        st.error("❌ Gagal menyimpan rating")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Desktop layout
            if show_rating_option and db_manager and user_id:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            else:
                col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {idx}. {place['Place_Name']}")
                st.markdown(f"**📍 {place['City']}** | **🏷️ {place['Category']}**")
                
                if len(place['Description']) > 200:
                    description = place['Description'][:200] + "..."
                else:
                    description = place['Description']
                st.markdown(f"📝 {description}")
                
                # Tags
                tags = f"🏠 {place['Outdoor/Indoor']}"
                if 'distance_km' in place and pd.notna(place['distance_km']):
                    tags += f" | 📍 {place['distance_km']:.1f} km"
                st.markdown(tags)
            
            with col2:
                st.metric("⭐ Rating", f"{place['Rating']}/5")
                st.metric("💰 Harga", f"Rp {place['Price']:,}")
            
            with col3:
                if 'recommendation_score' in place:
                    score_percent = (place['recommendation_score'] * 100)
                    st.metric("🎯 Score", f"{score_percent:.1f}%")
                
                # Weather suitability
                if weather['is_rainy'] and place['Outdoor/Indoor'] == 'Indoor':
                    st.success("☔ Cocok untuk cuaca hujan")
                elif weather['is_sunny'] and place['Outdoor/Indoor'] == 'Outdoor':
                    st.success("☀️ Cocok untuk cuaca cerah")
            
            # Rating column for authenticated users
            if show_rating_option and db_manager and user_id:
                with col4:
                    st.markdown("**⭐ Beri Rating**")
                    user_rating = st.slider(
                        "Rating", 1, 5, 3, 
                        key=f"rating_{place['Place_Id']}_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    if st.button("Submit", key=f"submit_{place['Place_Id']}_{idx}"):
                        if db_manager.add_user_rating(user_id, place['Place_Id'], user_rating):
                            st.success("✅ Tersimpan!")
                            
                            # Update user preferences based on rating
                            if user_rating >= 4:
                                db_manager.update_user_preferences(user_id, place['Category'], 1.0)
                            elif user_rating <= 2:
                                db_manager.update_user_preferences(user_id, place['Category'], 0.3)
                            
                            st.rerun()
                        else:
                            st.error("❌ Gagal!")
        
        st.markdown("---")

def detect_mobile_view():
    """Detect if user is on mobile device"""
    # Simple detection based on user agent or screen size
    # This is a basic implementation
    return st.session_state.get('mobile_view', False)

def create_map_visualization(recommendations, user_location):
    """Create interactive map with recommendations"""
    fig = go.Figure()
    
    # Add user location
    fig.add_trace(go.Scattermapbox(
        lat=[user_location['lat']],
        lon=[user_location['lng']],
        mode='markers',
        marker=dict(size=15, color='red'),
        text=['Your Location'],
        name='Your Location'
    ))
    
    # Add recommended places
    if not recommendations.empty:
        fig.add_trace(go.Scattermapbox(
            lat=recommendations['Lat'] / 10000000,
            lon=recommendations['Long'] / 10000000,
            mode='markers',
            marker=dict(size=12, color='blue'),
            text=recommendations['Place_Name'],
            name='Recommended Places'
        ))
    
    fig.update_layout(
        mapbox=dict(
            accesstoken='pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uz-VWAjzWjIlKm4eQ',
            style='open-street-map',
            center=dict(lat=user_location['lat'], lon=user_location['lng']),
            zoom=10
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def create_analytics_charts(places_df, ratings_df):
    """Create responsive analytics charts"""
    # Check if mobile view
    is_mobile = st.session_state.get('mobile_view', False)
    
    if is_mobile:
        # Mobile layout - stacked charts
        st.markdown("#### 📊 Distribusi Kategori Wisata")
        category_counts = places_df['Category'].value_counts()
        fig_cat = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Kategori Wisata",
            height=400
        )
        fig_cat.update_layout(font_size=10)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        st.markdown("#### ⭐ Distribusi Rating Destinasi")
        fig_rating = px.histogram(
            places_df,
            x='Rating',
            nbins=15,
            title="Rating Destinasi Wisata",
            height=350
        )
        fig_rating.update_layout(font_size=10)
        st.plotly_chart(fig_rating, use_container_width=True)
        
        st.markdown("#### 🏙️ Analisis Kota")
        city_stats = places_df.groupby('City').agg({
            'Rating': 'mean',
            'Price': 'mean',
            'Place_Id': 'count'
        }).reset_index()
        city_stats.columns = ['City', 'Avg_Rating', 'Avg_Price', 'Count']
        
        # Show top 10 cities for mobile
        city_stats_top = city_stats.nlargest(10, 'Count')
        fig_city = px.bar(
            city_stats_top,
            x='City',
            y='Count',
            title="Top 10 Kota dengan Destinasi Terbanyak",
            height=400
        )
        fig_city.update_layout(font_size=10, xaxis_tickangle=-45)
        st.plotly_chart(fig_city, use_container_width=True)
        
    else:
        # Desktop layout - side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = places_df['Category'].value_counts()
            fig_cat = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="📊 Distribusi Kategori Wisata"
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig_rating = px.histogram(
                places_df,
                x='Rating',
                nbins=20,
                title="⭐ Distribusi Rating Destinasi Wisata"
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        # City wise analysis
        city_stats = places_df.groupby('City').agg({
            'Rating': 'mean',
            'Price': 'mean',
            'Place_Id': 'count'
        }).reset_index()
        city_stats.columns = ['City', 'Avg_Rating', 'Avg_Price', 'Count']
        
        fig_city = px.scatter(
            city_stats,
            x='Avg_Price',
            y='Avg_Rating',
            size='Count',
            hover_name='City',
            title="🏙️ Analisis Kota: Harga vs Rating vs Jumlah Destinasi"
        )
        st.plotly_chart(fig_city, use_container_width=True)

def main():
    """Main Streamlit application dengan autentikasi"""
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Check authentication
    if not st.session_state.authenticated:
        show_auth_page()
        return
    
    # Authenticated user interface
    user = st.session_state.user
    db_manager = st.session_state.db_manager
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<h1 class="main-header">🏛️ Sistem Rekomendasi Wisata Hybrid</h1>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"**👋 Selamat datang, {user['full_name']}!**")
    with col3:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    st.markdown("---")
    
    # Load data
    users_df, places_df, ratings_df, success = load_datasets()
    
    if not success:
        st.error("❌ Gagal memuat dataset. Pastikan file dataset tersedia di folder 'dataset/'")
        st.info("📁 File yang dibutuhkan: user.csv, destinasi-wisata-YKSM.csv, tour_rating.csv")
        return
    
    # Initialize recommendation system
    if 'recommender' not in st.session_state or not hasattr(st.session_state, 'db_manager'):
        with st.spinner('🚀 Initializing recommendation system...'):
            recommender = HybridTourismRecommendationSystem(db_manager)
            recommender.load_data(users_df, places_df, ratings_df)
            recommender.train_models()
            st.session_state.recommender = recommender
            st.success("✅ System initialized successfully!")
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Pengaturan Rekomendasi")
    
    # User info in sidebar
    st.sidebar.markdown(f"""
    **� Profil Anda:**
    - **Nama:** {user['full_name']}
    - **Lokasi:** {user['location']}
    - **Umur:** {user['age']} tahun
    """)
    
    # View mode toggle
    view_mode = st.sidebar.radio(
        "� Mode Tampilan:",
        ["Desktop", "Mobile"],
        help="Pilih mode tampilan sesuai perangkat Anda"
    )
    st.session_state.mobile_view = (view_mode == "Mobile")
    
    # Parameters
    num_recommendations = st.sidebar.slider(
        "📝 Jumlah Rekomendasi:",
        min_value=3,
        max_value=15,
        value=8,
        help="Tentukan berapa banyak rekomendasi yang ingin ditampilkan"
    )
    
    max_distance = st.sidebar.slider(
        "📍 Radius Maksimum (km):",
        min_value=10,
        max_value=200,
        value=100,
        help="Jarak maksimum destinasi wisata dari lokasi user"
    )
    
    # Show user's rating history
    with st.sidebar.expander("📊 Riwayat Rating Anda"):
        user_ratings_df = db_manager.get_user_ratings(user['id'])
        if not user_ratings_df.empty:
            for _, rating in user_ratings_df.iterrows():
                place_info = places_df[places_df['Place_Id'] == rating['Place_Id']]
                if not place_info.empty:
                    place_name = place_info.iloc[0]['Place_Name']
                    st.markdown(f"⭐ {place_name}: {rating['Place_Ratings']}/5")
        else:
            st.markdown("Belum ada rating")
    
    # Main navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "📖 Pilih Halaman:",
        ["🎯 Rekomendasi", "📊 Dashboard", "📈 Analytics"],
        help="Navigasi antar halaman aplikasi"
    )
    
    if page == "📊 Dashboard":
        show_dashboard_page(places_df, ratings_df, users_df)
    elif page == "📈 Analytics":
        show_analytics_page(places_df, ratings_df, users_df)
    else:
        show_recommendation_page(places_df, ratings_df, users_df, db_manager, user, num_recommendations, max_distance)
def show_dashboard_page(places_df, ratings_df, users_df):
    """Halaman dashboard untuk menampilkan semua data destinasi wisata"""
    st.markdown('<h2 class="sub-header">📊 Dashboard Destinasi Wisata</h2>', unsafe_allow_html=True)
    
    # Filter options
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_cities = st.multiselect(
            "🏙️ Filter Kota:",
            options=sorted(places_df['City'].unique()),
            default=[]
        )
    
    with col2:
        selected_categories = st.multiselect(
            "🏷️ Filter Kategori:",
            options=sorted(places_df['Category'].unique()),
            default=[]
        )
    
    with col3:
        price_range = st.slider(
            "💰 Range Harga:",
            min_value=int(places_df['Price'].min()),
            max_value=int(places_df['Price'].max()),
            value=(int(places_df['Price'].min()), int(places_df['Price'].max()))
        )
    
    with col4:
        rating_filter = st.slider(
            "⭐ Rating Minimum:",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.1
        )
    
    # Search box
    search_query = st.text_input(
        "🔍 Cari destinasi:",
        placeholder="Ketik nama destinasi atau deskripsi..."
    )
    
    # Apply filters
    filtered_df = places_df.copy()
    
    if selected_cities:
        filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
    
    filtered_df = filtered_df[
        (filtered_df['Price'] >= price_range[0]) & 
        (filtered_df['Price'] <= price_range[1]) &
        (filtered_df['Rating'] >= rating_filter)
    ]
    
    if search_query:
        mask = (
            filtered_df['Place_Name'].str.contains(search_query, case=False, na=False) |
            filtered_df['Description'].str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Total Destinasi", len(filtered_df))
    
    with col2:
        if not filtered_df.empty:
            avg_rating = filtered_df['Rating'].mean()
            st.metric("⭐ Rating Rata-rata", f"{avg_rating:.2f}")
        else:
            st.metric("⭐ Rating Rata-rata", "N/A")
    
    with col3:
        if not filtered_df.empty:
            avg_price = filtered_df['Price'].mean()
            st.metric("💰 Harga Rata-rata", f"Rp {avg_price:,.0f}")
        else:
            st.metric("💰 Harga Rata-rata", "N/A")
    
    with col4:
        if not filtered_df.empty:
            unique_cities = filtered_df['City'].nunique()
            st.metric("🏙️ Jumlah Kota", unique_cities)
        else:
            st.metric("🏙️ Jumlah Kota", "0")
    
    st.markdown("---")
    
    # Data display
    if filtered_df.empty:
        st.warning("⚠️ Tidak ada destinasi yang sesuai dengan filter yang dipilih.")
    else:
        # Sort options
        sort_option = st.selectbox(
            "📊 Urutkan berdasarkan:",
            ["Rating (Tertinggi)", "Rating (Terendah)", "Harga (Termurah)", "Harga (Termahal)", "Nama (A-Z)"]
        )
        
        if sort_option == "Rating (Tertinggi)":
            filtered_df = filtered_df.sort_values('Rating', ascending=False)
        elif sort_option == "Rating (Terendah)":
            filtered_df = filtered_df.sort_values('Rating', ascending=True)
        elif sort_option == "Harga (Termurah)":
            filtered_df = filtered_df.sort_values('Price', ascending=True)
        elif sort_option == "Harga (Termahal)":
            filtered_df = filtered_df.sort_values('Price', ascending=False)
        else:
            filtered_df = filtered_df.sort_values('Place_Name', ascending=True)
        
        # Pagination
        items_per_page = st.selectbox("📄 Item per halaman:", [10, 20, 50, 100], index=1)
        total_pages = (len(filtered_df) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page_num = st.selectbox(f"📖 Halaman (1-{total_pages}):", range(1, total_pages + 1))
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_df = filtered_df.iloc[start_idx:end_idx]
        else:
            page_df = filtered_df
        
        # Display cards
        for idx, (_, place) in enumerate(page_df.iterrows(), 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {place['Place_Name']}")
                    st.markdown(f"**📍 {place['City']}** | **🏷️ {place['Category']}**")
                    
                    # Truncate description for dashboard view
                    description = place['Description'][:150] + "..." if len(place['Description']) > 150 else place['Description']
                    st.markdown(f"📝 {description}")
                    
                    st.markdown(f"🏠 {place['Outdoor/Indoor']}")
                
                with col2:
                    st.metric("⭐ Rating", f"{place['Rating']}/5")
                    st.metric("📊 Reviews", f"{place['Rating_Count']}")
                
                with col3:
                    st.metric("💰 Harga", f"Rp {place['Price']:,}")
                    
                    # Show detail button
                    if st.button(f"👁️ Detail", key=f"detail_{place['Place_Id']}"):
                        st.session_state[f"show_detail_{place['Place_Id']}"] = True
                
                # Detail modal
                if st.session_state.get(f"show_detail_{place['Place_Id']}", False):
                    with st.expander(f"📋 Detail {place['Place_Name']}", expanded=True):
                        st.markdown(f"**📝 Deskripsi Lengkap:**")
                        st.markdown(place['Description'])
                        
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.markdown(f"**📍 Koordinat:** {place['Lat']/10000000:.6f}, {place['Long']/10000000:.6f}")
                            st.markdown(f"**🏷️ Kategori:** {place['Category']}")
                        
                        with detail_col2:
                            st.markdown(f"**🏠 Jenis:** {place['Outdoor/Indoor']}")
                            st.markdown(f"**🆔 ID:** {place['Place_Id']}")
                        
                        if st.button(f"❌ Tutup Detail", key=f"close_detail_{place['Place_Id']}"):
                            st.session_state[f"show_detail_{place['Place_Id']}"] = False
                            st.rerun()
                
                st.markdown("---")

def show_recommendation_page(places_df, ratings_df, users_df, db_manager, user, num_recommendations, max_distance):
    """Halaman rekomendasi dengan input yang beragam"""
    st.markdown('<h2 class="sub-header">🎯 Sistem Rekomendasi Wisata</h2>', unsafe_allow_html=True)
    
    # Input section - simplified
    st.markdown("### 📝 Pengaturan Rekomendasi")
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 Lokasi & Radius**")
        selected_city = st.selectbox(
            "🏙️ Pilih Kota:",
            ["Semarang", "Yogyakarta"],
            help="Pilih kota sebagai pusat pencarian"
        )
        
        radius_km = st.slider(
            "📏 Radius Pencarian (km):",
            min_value=10,
            max_value=100,
            value=50,
            help="Jarak maksimum dari kota yang dipilih"
        )
        
        st.markdown("**💰 Rentang Harga**")
        price_max = st.number_input(
            "Harga Maksimum (Rp):",
            min_value=10000,
            max_value=int(places_df['Price'].max()),
            value=100000,
            step=10000
        )
    
    with col2:
        st.markdown("**🌤️ Pengaturan Cuaca**")
        weather_input_type = st.radio(
            "Pilih jenis input cuaca:",
            ["🌐 Otomatis (API)", "✋ Manual"],
            help="Pilih apakah ingin menggunakan data cuaca real-time atau input manual"
        )
        
        if weather_input_type == "✋ Manual":
            manual_weather = st.selectbox(
                "Kondisi Cuaca:",
                ["☀️ Cerah", "⛅ Berawan", "🌧️ Hujan"],
                help="Pilih kondisi cuaca untuk rekomendasi"
            )
        
        st.markdown("**📝 Deskripsi Singkat**")
        description_input = st.text_area(
            "Deskripsikan tempat yang diinginkan:",
            placeholder="Contoh: tempat tenang, alam, budaya, dll.",
            height=80,
            help="Jelaskan jenis destinasi wisata yang Anda cari"
        )
    
    # Simplified filters
    st.markdown("### 🎛️ Filter Sederhana")
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        min_rating = st.slider(
            "⭐ Rating Minimum:",
            min_value=0.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
            help="Rating minimum destinasi wisata"
        )
    
    with filter_col2:
        indoor_outdoor = st.selectbox(
            "🏠 Jenis Tempat:",
            ["Semua", "Indoor", "Outdoor"],
            help="Pilih jenis tempat berdasarkan lokasi"
        )
    
    # Generate recommendations button
    if st.button("🎯 Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner('🔍 Generating personalized recommendations...'):
            try:
                # Initialize recommendation system if not exists
                if 'recommender' not in st.session_state:
                    recommender = HybridTourismRecommendationSystem(db_manager)
                    recommender.load_data(users_df, places_df, ratings_df)
                    recommender.train_models()
                    st.session_state.recommender = recommender
                
                recommender = st.session_state.recommender
                
                # Get weather data
                city_coords = {
                    'Semarang': {'lat': -6.9667, 'lng': 110.4167},
                    'Yogyakarta': {'lat': -7.7956, 'lng': 110.3695}
                }
                user_location = city_coords[selected_city]
                
                if weather_input_type == "🌐 Otomatis (API)":
                    weather = recommender.weather_service.get_weather(
                        user_location['lat'], user_location['lng']
                    )
                else:
                    # Manual weather input
                    weather = {
                        'temperature': 28,  # Default temperature
                        'precipitation': 0 if "☀️" in manual_weather or "⛅" in manual_weather else 5,
                        'is_sunny': "☀️" in manual_weather,
                        'is_rainy': "🌧️" in manual_weather,
                        'description': manual_weather.split()[1] if len(manual_weather.split()) > 1 else manual_weather,
                        'icon': manual_weather.split()[0]
                    }
                
                # Generate recommendations with enhanced parameters
                recommendations = generate_enhanced_recommendations(
                    recommender=recommender,
                    user_id=user['id'],
                    user_location=user_location,
                    weather=weather,
                    places_df=places_df,
                    description_input=description_input,
                    indoor_outdoor=indoor_outdoor,
                    price_range=(0, price_max),  # Use 0 as minimum price
                    min_rating=min_rating,
                    radius_km=radius_km,
                    top_k=num_recommendations,
                    is_authenticated_user=True
                )
                
                # Store in session state
                st.session_state.recommendations = recommendations
                st.session_state.weather = weather
                st.session_state.user_location = user_location
                st.session_state.selected_city = selected_city
                
                st.success(f"✅ Berhasil generate {len(recommendations)} rekomendasi!")
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.exception(e)
    
    # Display results
    if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
        recommendations = st.session_state.recommendations
        weather = st.session_state.weather
        user_location = st.session_state.user_location
        selected_city = st.session_state.get('selected_city', 'Yogyakarta')
        
        st.markdown("---")
        st.markdown("## 🎯 Rekomendasi untuk Anda")
        
        # Weather info
        st.markdown(f"""
        <div class="weather-card">
            <h3>{weather['icon']} Cuaca di {selected_city}</h3>
            <p><strong>{weather['description']}</strong></p>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div>🌡️ {weather['temperature']}°C</div>
                <div>🌧️ {weather.get('precipitation', 0)} mm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        for idx, (_, place) in enumerate(recommendations.iterrows(), 1):
            create_responsive_recommendation_card(
                place, 
                idx, 
                weather, 
                show_rating_option=True,
                db_manager=db_manager,
                user_id=user['id']
            )
        
        # Map visualization
        st.markdown("### 🗺️ Peta Rekomendasi")
        if not recommendations.empty:
            map_fig = create_map_visualization(recommendations, user_location)
            st.plotly_chart(map_fig, use_container_width=True)

def generate_enhanced_recommendations(recommender, user_id, user_location, weather, places_df, 
                                   description_input, indoor_outdoor, 
                                   price_range, min_rating, radius_km, top_k, is_authenticated_user=False):
    """Generate enhanced recommendations with simplified filters"""
    
    # Start with all places
    filtered_places = places_df.copy()
    
    # Apply basic filters
    filtered_places = filtered_places[
        (filtered_places['Price'] >= price_range[0]) &
        (filtered_places['Price'] <= price_range[1]) &
        (filtered_places['Rating'] >= min_rating)
    ]
    
    # Indoor/Outdoor filter
    if indoor_outdoor != "Semua":
        filtered_places = filtered_places[filtered_places['Outdoor/Indoor'] == indoor_outdoor]
    
    # Location-based filtering (radius)
    if not filtered_places.empty:
        distances = []
        user_coords = (user_location['lat'], user_location['lng'])
        
        for _, place in filtered_places.iterrows():
            try:
                place_coords = (place['Lat'] / 10000000, place['Long'] / 10000000)
                distance = geodesic(user_coords, place_coords).kilometers
                distances.append(distance)
            except:
                distances.append(float('inf'))
        
        filtered_places = filtered_places.copy()
        filtered_places['distance_km'] = distances
        filtered_places = filtered_places[filtered_places['distance_km'] <= radius_km]
    
    if filtered_places.empty:
        return pd.DataFrame()
    
    # Simplified TF-IDF content matching
    content_scores = []
    if description_input.strip():
        tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Combine text features from places
        place_texts = (
            filtered_places['Place_Name'].fillna('') + ' ' +
            filtered_places['Description'].fillna('') + ' ' +
            filtered_places['Category'].fillna('')
        )
        
        try:
            # Add user description
            all_texts = place_texts.tolist() + [description_input]
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate similarity
            user_vector = tfidf_matrix[-1]
            place_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(user_vector, place_vectors).flatten()
            content_scores = similarities.tolist()
        except:
            content_scores = [0.5] * len(filtered_places)
    else:
        content_scores = [0.5] * len(filtered_places)
    
    # Simple weather scoring
    weather_scores = []
    for _, place in filtered_places.iterrows():
        score = 0.5  # Base score
        if weather['is_rainy'] and place['Outdoor/Indoor'] == 'Indoor':
            score = 0.8
        elif weather['is_sunny'] and place['Outdoor/Indoor'] == 'Outdoor':
            score = 0.7
        weather_scores.append(score)
    
    # Simple collaborative scoring
    collab_scores = []
    if recommender.db_manager:
        user_ratings_df = recommender.db_manager.get_user_ratings(user_id)
        if not user_ratings_df.empty:
            user_avg_rating = user_ratings_df['Place_Ratings'].mean()
            for _, place in filtered_places.iterrows():
                score = 0.5  # Base score
                if place['Rating'] >= user_avg_rating:
                    score = 0.7
                collab_scores.append(score)
        else:
            collab_scores = [0.5] * len(filtered_places)
    else:
        collab_scores = [0.5] * len(filtered_places)
    
    # Calculate simplified hybrid score
    filtered_places = filtered_places.copy()
    filtered_places['content_score'] = content_scores
    filtered_places['weather_score'] = weather_scores
    filtered_places['collab_score'] = collab_scores
    
    # Normalize distance score
    max_distance = filtered_places['distance_km'].max()
    distance_scores = (1 - filtered_places['distance_km'] / max_distance) if max_distance > 0 else [1] * len(filtered_places)
    
    # Simple weighted hybrid score
    filtered_places['hybrid_score'] = (
        0.3 * filtered_places['content_score'] +
        0.2 * filtered_places['weather_score'] +
        0.2 * filtered_places['collab_score'] +
        0.2 * (filtered_places['Rating'] / 5.0) +
        0.1 * distance_scores
    )
    
    # Sort by hybrid score and return top k
    final_recommendations = filtered_places.sort_values('hybrid_score', ascending=False).head(top_k)
    
    return final_recommendations

def show_analytics_page(places_df, ratings_df, users_df):
    """Halaman analytics dengan visualisasi data"""
    st.markdown('<h2 class="sub-header">📈 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Total Destinasi", len(places_df))
    
    with col2:
        st.metric("👥 Total User", len(users_df))
    
    with col3:
        st.metric("⭐ Total Rating", len(ratings_df))
    
    with col4:
        avg_rating = places_df['Rating'].mean()
        st.metric("📊 Avg Rating", f"{avg_rating:.2f}")
    
    st.markdown("---")
    
    # Charts
    create_analytics_charts(places_df, ratings_df)


if __name__ == "__main__":
    main()