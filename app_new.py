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

def show_dashboard_page(places_df, ratings_df, users_df, db_manager, user):
    """Show main dashboard with data visualization"""
    st.markdown('<h1 class="main-header">📊 Dashboard Wisata Indonesia</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_places = len(places_df)
        st.metric("🏛️ Total Destinasi", f"{total_places:,}")
    
    with col2:
        avg_rating = places_df['Rating'].mean()
        st.metric("⭐ Rating Rata-rata", f"{avg_rating:.1f}/5")
    
    with col3:
        db_users_count = 0
        try:
            db_ratings = db_manager.get_all_user_ratings()
            db_users_count = db_ratings['User_Id'].nunique() if not db_ratings.empty else 0
        except:
            pass
        total_users = len(users_df) + db_users_count
        st.metric("👥 Total Pengguna", f"{total_users:,}")
    
    with col4:
        db_ratings_count = 0
        try:
            db_ratings = db_manager.get_all_user_ratings()
            db_ratings_count = len(db_ratings) if not db_ratings.empty else 0
        except:
            pass
        total_ratings = len(ratings_df) + db_ratings_count
        st.metric("📝 Total Rating", f"{total_ratings:,}")
    
    st.markdown("---")
    
    # Data Visualization
    create_analytics_charts(places_df, ratings_df)
    
    # Top destinations
    st.markdown("### 🏆 Top 10 Destinasi Wisata Terpopuler")
    top_places = places_df.nlargest(10, 'Rating_Count')[['Place_Name', 'City', 'Category', 'Rating', 'Price', 'Rating_Count']]
    st.dataframe(top_places, use_container_width=True)
    
    # User activity summary
    st.markdown("### 👤 Aktivitas Anda")
    user_ratings_df = db_manager.get_user_ratings(user['id'])
    
    if not user_ratings_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎯 Destinasi yang Anda Rating", len(user_ratings_df))
            st.metric("⭐ Rating Rata-rata Anda", f"{user_ratings_df['Place_Ratings'].mean():.1f}/5")
        
        with col2:
            # User's favorite categories
            rated_places = places_df[places_df['Place_Id'].isin(user_ratings_df['Place_Id'])]
            if not rated_places.empty:
                fav_categories = rated_places['Category'].value_counts().head(3)
                st.markdown("**🏷️ Kategori Favorit:**")
                for i, (category, count) in enumerate(fav_categories.items(), 1):
                    st.write(f"{i}. {category} ({count} tempat)")
    else:
        st.info("💡 Belum ada aktivitas rating. Silakan coba fitur rekomendasi!")

def show_content_weather_recommendation_page(places_df, ratings_df, users_df, db_manager, user):
    """Show content-based + weather recommendation page"""
    st.markdown('<h1 class="main-header">🌤️ Rekomendasi Content-Based & Weather</h1>', unsafe_allow_html=True)
    st.markdown("**Rekomendasi berdasarkan kategori favorit dan kondisi cuaca saat ini**")
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Pengaturan Rekomendasi")
    
    # User preferences
    available_categories = places_df['Category'].unique().tolist()
    user_preferences = db_manager.get_user_preferences(user['id'])
    
    selected_categories = st.sidebar.multiselect(
        "🏷️ Pilih Kategori Wisata:",
        available_categories,
        default=user_preferences[:3] if user_preferences else available_categories[:3],
        help="Pilih kategori destinasi yang Anda minati"
    )
    
    num_recommendations = st.sidebar.slider(
        "📝 Jumlah Rekomendasi:",
        min_value=5,
        max_value=20,
        value=10,
        help="Tentukan berapa banyak rekomendasi yang ingin ditampilkan"
    )
    
    # Weather consideration
    consider_weather = st.sidebar.checkbox(
        "🌦️ Pertimbangkan Cuaca",
        value=True,
        help="Filter rekomendasi berdasarkan kondisi cuaca saat ini"
    )
    
    # Generate recommendations button
    if st.sidebar.button("🎯 Generate Recommendations", type="primary"):
        with st.spinner('🔍 Generating content-based recommendations...'):
            # Initialize weather service
            weather_service = WeatherService()
            
            # Get user location
            user_location = get_user_location_from_db(user, db_manager)
            weather = weather_service.get_weather(user_location['lat'], user_location['lng'])
            
            # Content-based filtering
            content_filter = ContentBasedFilter()
            content_filter.fit(places_df)
            
            # Get recommendations
            recommendations = content_filter.recommend_by_category(
                selected_categories, 
                weather if consider_weather else {'is_rainy': False, 'temperature': 25}, 
                top_k=num_recommendations
            )
            
            # Store in session state
            st.session_state.content_recommendations = recommendations
            st.session_state.content_weather = weather
            st.session_state.content_user_location = user_location
            st.session_state.selected_categories = selected_categories
    
    # Display recommendations
    if 'content_recommendations' in st.session_state:
        recommendations = st.session_state.content_recommendations
        weather = st.session_state.content_weather
        user_location = st.session_state.content_user_location
        selected_categories = st.session_state.selected_categories
        
        # Weather info
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="weather-card">
                <h3>{weather['icon']} Cuaca di {user['location']}</h3>
                <p><strong>{weather['description']}</strong></p>
                <div style="display: flex; justify-content: space-around;">
                    <div>🌡️ {weather['temperature']}°C</div>
                    <div>🌧️ {weather['precipitation']} mm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendation summary
        st.markdown(f"""
        ### 📋 Ringkasan Rekomendasi
        - **Kategori yang dipilih:** {', '.join(selected_categories)}
        - **Jumlah rekomendasi:** {len(recommendations)}
        - **Berdasarkan cuaca:** {'Ya' if consider_weather else 'Tidak'}
        """)
        
        if weather['is_rainy'] and consider_weather:
            st.info("☔ Cuaca hujan terdeteksi. Prioritas diberikan pada destinasi indoor.")
        elif weather['temperature'] > 32 and consider_weather:
            st.info("☀️ Cuaca panas terdeteksi. Prioritas diberikan pada destinasi sejuk.")
        
        st.markdown("---")
        
        # Display recommendations
        st.markdown("### 🎯 Rekomendasi Destinasi Wisata")
        
        if not recommendations.empty:
            for idx, (_, place) in enumerate(recommendations.iterrows(), 1):
                create_content_recommendation_card(
                    place, idx, weather, 
                    show_rating_option=True, 
                    db_manager=db_manager, 
                    user_id=user['id']
                )
        else:
            st.warning("❌ Tidak ada rekomendasi yang ditemukan dengan kriteria yang dipilih.")
    
    else:
        # Initial page content
        st.markdown("### 🚀 Cara Menggunakan")
        st.markdown("""
        1. **Pilih Kategori** - Tentukan jenis destinasi wisata yang Anda minati
        2. **Atur Jumlah** - Sesuaikan berapa banyak rekomendasi yang diinginkan  
        3. **Pertimbangkan Cuaca** - Aktifkan untuk filter berdasarkan kondisi cuaca
        4. **Generate** - Klik tombol untuk mendapatkan rekomendasi
        """)

def get_user_location_from_db(user, db_manager):
    """Get user location from database"""
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
    
    location_str = user['location']
    for city in location_mapping:
        if city in location_str:
            return location_mapping[city]
    
    return location_mapping['Yogyakarta']

def create_content_recommendation_card(place, idx, weather, show_rating_option=False, db_manager=None, user_id=None):
    """Create a recommendation card for content-based recommendations"""
    with st.container():
        # Card header
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 🏛️ {idx}. {place['Place_Name']}")
            st.markdown(f"**📍 {place['City']} | 🏷️ {place['Category']}**")
        
        with col2:
            st.metric("⭐ Rating", f"{place['Rating']}/5")
        
        # Description and details
        description = place['Description'][:300] + "..." if len(str(place['Description'])) > 300 else str(place['Description'])
        st.markdown(f"📝 **Deskripsi:** {description}")
        
        # Additional info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"💰 **Harga:** Rp {place['Price']:,}")
        
        with col2:
            st.markdown(f"🏠 **Lokasi:** {place['Outdoor/Indoor']}")
        
        with col3:
            st.markdown(f"📊 **Jumlah Review:** {place['Rating_Count']:,}")
        
        # Weather suitability
        if weather['is_rainy'] and place['Outdoor/Indoor'] == 'Indoor':
            st.success("☔ Cocok untuk cuaca hujan - Destinasi Indoor")
        elif weather['is_sunny'] and place['Outdoor/Indoor'] == 'Outdoor':
            st.success("☀️ Cocok untuk cuaca cerah - Destinasi Outdoor")
        elif weather['temperature'] > 32 and place['Category'] in ['Bahari', 'Cagar Alam']:
            st.success("🌊 Cocok untuk cuaca panas - Destinasi Sejuk")
        
        # Rating option for authenticated users
        if show_rating_option and db_manager and user_id:
            st.markdown("---")
            st.markdown("#### ⭐ Beri Rating untuk Destinasi Ini")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                user_rating = st.slider(
                    f"Rating untuk {place['Place_Name']}", 
                    1, 5, 3, 
                    key=f"content_rating_{place['Place_Id']}_{idx}",
                    help="Berikan penilaian Anda (1=Sangat Buruk, 5=Sangat Baik)"
                )
            
            with col2:
                if st.button(f"💾 Submit Rating", key=f"content_submit_{place['Place_Id']}_{idx}"):
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
        
        st.markdown("---")

def create_analytics_charts(places_df, ratings_df):
    """Create responsive analytics charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        st.markdown("#### 📊 Distribusi Kategori Wisata")
        category_counts = places_df['Category'].value_counts()
        fig_cat = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Kategori Wisata"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Rating distribution
        st.markdown("#### ⭐ Distribusi Rating Destinasi")
        fig_rating = px.histogram(
            places_df,
            x='Rating',
            nbins=20,
            title="Rating Destinasi Wisata"
        )
        st.plotly_chart(fig_rating, use_container_width=True)
    
    # City wise analysis
    st.markdown("#### 🏙️ Analisis per Kota")
    city_stats = places_df.groupby('City').agg({
        'Rating': 'mean',
        'Price': 'mean',
        'Place_Id': 'count'
    }).reset_index()
    city_stats.columns = ['City', 'Avg_Rating', 'Avg_Price', 'Count']
    
    # Show top 15 cities
    city_stats_top = city_stats.nlargest(15, 'Count')
    fig_city = px.bar(
        city_stats_top,
        x='City',
        y='Count',
        title="Top 15 Kota dengan Destinasi Terbanyak",
        height=400
    )
    fig_city.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_city, use_container_width=True)

def main():
    """Main Streamlit application dengan multi-page navigation"""
    
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
    
    # Load data
    users_df, places_df, ratings_df, success = load_datasets()
    
    if not success:
        st.error("❌ Gagal memuat dataset. Pastikan file dataset tersedia di folder 'dataset/'")
        st.info("📁 File yang dibutuhkan: user.csv, destinasi-wisata-YKSM.csv, tour_rating.csv")
        return
    
    # Sidebar navigation
    st.sidebar.markdown("# 🏛️ Tourism Recommendation")
    st.sidebar.markdown(f"**👋 Welcome, {user['full_name']}!**")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "📖 Pilih Halaman:",
        [
            "📊 Dashboard",
            "🌤️ Content-Based + Weather"
        ],
        help="Pilih halaman yang ingin Anda kunjungi"
    )
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="secondary"):
        st.session_state.authenticated = False
        st.session_state.user = None
        # Clear all session state
        for key in list(st.session_state.keys()):
            if key not in ['authenticated', 'user', 'db_manager']:
                del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **ℹ️ Info Akun:**
    - **Email:** {user['email']}
    - **Lokasi:** {user['location']}
    - **Umur:** {user['age']} tahun
    """)
    
    # Show selected page
    if page == "📊 Dashboard":
        show_dashboard_page(places_df, ratings_df, users_df, db_manager, user)
    elif page == "🌤️ Content-Based + Weather":
        show_content_weather_recommendation_page(places_df, ratings_df, users_df, db_manager, user)

if __name__ == "__main__":
    main()
