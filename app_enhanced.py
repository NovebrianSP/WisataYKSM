import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    page_title="🏛️ Enhanced Tourism Recommendation System",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    .main {
        padding: 1rem;
    }
    
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
    
    .dashboard-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-4px);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 6px 20px rgba(116, 185, 255, 0.2);
    }
    
    .weather-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .filter-card {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path="enhanced_tourism_app.db"):
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
        """Get all user ratings"""
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
    """Enhanced weather service"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.city_coords = {
            'Semarang': {'lat': -6.9667, 'lng': 110.4167},
            'Yogyakarta': {'lat': -7.7956, 'lng': 110.3695}
        }
    
    def get_weather_auto(self, city):
        """Get weather automatically using API"""
        coords = self.city_coords.get(city, self.city_coords['Yogyakarta'])
        
        try:
            params = {
                'latitude': coords['lat'],
                'longitude': coords['lng'],
                'current_weather': 'true',
                'daily': 'precipitation_sum,temperature_2m_max,temperature_2m_min',
                'timezone': 'Asia/Jakarta'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current_weather = data.get('current_weather', {})
            
            weather_code = current_weather.get('weathercode', 0)
            temperature = current_weather.get('temperature', 25)
            
            return {
                'temperature': temperature,
                'is_sunny': weather_code in [0, 1, 2],
                'is_rainy': weather_code in [61, 63, 65, 80, 81, 82],
                'weather_code': weather_code,
                'description': self._get_weather_description(weather_code),
                'icon': self._get_weather_icon(weather_code),
                'source': 'API'
            }
        except Exception as e:
            return {
                'temperature': 25,
                'is_sunny': True,
                'is_rainy': False,
                'weather_code': 0,
                'description': 'Cerah',
                'icon': '☀️',
                'source': 'Default'
            }
    
    def get_weather_manual(self, temperature, condition):
        """Get weather from manual input"""
        is_sunny = condition in ['Cerah', 'Berawan']
        is_rainy = condition in ['Hujan Ringan', 'Hujan Lebat']
        
        icon_map = {
            'Cerah': '☀️',
            'Berawan': '⛅',
            'Mendung': '☁️',
            'Hujan Ringan': '🌧️',
            'Hujan Lebat': '⛈️'
        }
        
        return {
            'temperature': temperature,
            'is_sunny': is_sunny,
            'is_rainy': is_rainy,
            'weather_code': 0,
            'description': condition,
            'icon': icon_map.get(condition, '🌤️'),
            'source': 'Manual'
        }
    
    def _get_weather_description(self, code):
        """Convert weather code to description"""
        weather_codes = {
            0: "Cerah", 1: "Sebagian Berawan", 2: "Berawan", 3: "Mendung",
            45: "Berkabut", 48: "Berkabut Tebal", 51: "Gerimis Ringan",
            53: "Gerimis Sedang", 55: "Gerimis Lebat", 61: "Hujan Ringan",
            63: "Hujan Sedang", 65: "Hujan Lebat", 80: "Hujan Deras",
            81: "Hujan Sangat Deras", 82: "Hujan Ekstrem"
        }
        return weather_codes.get(code, "Tidak Diketahui")
    
    def _get_weather_icon(self, code):
        """Get emoji icon for weather"""
        if code in [0]: return "☀️"
        elif code in [1, 2]: return "⛅"
        elif code in [3]: return "☁️"
        elif code in [45, 48]: return "🌫️"
        elif code in [51, 53, 55, 61, 63, 65]: return "🌧️"
        elif code in [80, 81, 82]: return "⛈️"
        else: return "🌤️"

class EnhancedRecommendationSystem:
    """Enhanced recommendation system with multiple algorithms"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.weather_service = WeatherService()
        self.content_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        
        self.places_df = None
        self.ratings_df = None
        self.users_df = None
        self.content_matrix = None
        self.user_similarity_matrix = None
    
    def load_data(self, users_df, places_df, ratings_df):
        """Load and prepare data"""
        self.users_df = users_df
        self.places_df = places_df
        self.ratings_df = ratings_df
        
        # Prepare content-based features
        self._prepare_content_features()
        
        # Prepare collaborative filtering
        self._prepare_collaborative_features()
        
        return True
    
    def _prepare_content_features(self):
        """Prepare TF-IDF matrix for content-based filtering"""
        # Combine text features
        text_features = (
            self.places_df['Place_Name'].fillna('') + ' ' +
            self.places_df['Description'].fillna('') + ' ' +
            self.places_df['Category'].fillna('') + ' ' +
            self.places_df['City'].fillna('')
        )
        
        # Create TF-IDF matrix
        self.content_matrix = self.content_vectorizer.fit_transform(text_features)
    
    def _prepare_collaborative_features(self):
        """Prepare user similarity matrix for collaborative filtering"""
        # Combine CSV and database ratings
        all_ratings = self.ratings_df.copy()
        
        if self.db_manager:
            db_ratings = self.db_manager.get_all_user_ratings()
            if not db_ratings.empty:
                # Offset database user IDs
                max_csv_user_id = self.ratings_df['User_Id'].max() if not self.ratings_df.empty else 0
                db_ratings['User_Id'] = db_ratings['User_Id'] + max_csv_user_id
                all_ratings = pd.concat([all_ratings, db_ratings], ignore_index=True)
        
        # Create user-item matrix
        user_item_matrix = all_ratings.pivot_table(
            index='User_Id', 
            columns='Place_Id', 
            values='Place_Ratings',
            fill_value=0
        )
        
        # Calculate user similarity
        if not user_item_matrix.empty:
            self.user_similarity_matrix = cosine_similarity(user_item_matrix.values)
            self.user_ids = user_item_matrix.index.tolist()
            self.place_ids = user_item_matrix.columns.tolist()
            self.user_item_values = user_item_matrix.values
        else:
            self.user_similarity_matrix = None
    
    def get_content_based_recommendations(self, description_query, top_k=10):
        """Content-based recommendations using TF-IDF similarity"""
        if description_query.strip() == "":
            return self.places_df.nlargest(top_k, 'Rating')
        
        # Transform query
        query_vector = self.content_vectorizer.transform([description_query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.content_matrix).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[::-1][:top_k]
        
        recommendations = self.places_df.iloc[top_indices].copy()
        recommendations['content_score'] = similarities[top_indices]
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, top_k=10):
        """Collaborative filtering recommendations"""
        if self.user_similarity_matrix is None:
            return self.places_df.nlargest(top_k, 'Rating')
        
        try:
            # Find user index
            if user_id not in self.user_ids:
                return self.places_df.nlargest(top_k, 'Rating')
            
            user_idx = self.user_ids.index(user_id)
            
            # Get similar users
            user_similarities = self.user_similarity_matrix[user_idx]
            similar_users = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users
            
            # Get recommendations from similar users
            user_ratings = self.user_item_values[user_idx]
            unrated_items = np.where(user_ratings == 0)[0]
            
            scores = {}
            for item_idx in unrated_items:
                score = 0
                weight_sum = 0
                
                for similar_user_idx in similar_users:
                    if self.user_item_values[similar_user_idx][item_idx] > 0:
                        similarity = user_similarities[similar_user_idx]
                        rating = self.user_item_values[similar_user_idx][item_idx]
                        score += similarity * rating
                        weight_sum += similarity
                
                if weight_sum > 0:
                    scores[self.place_ids[item_idx]] = score / weight_sum
            
            # Sort and get top recommendations
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_place_ids = [place_id for place_id, score in sorted_scores[:top_k]]
            
            recommendations = self.places_df[self.places_df['Place_Id'].isin(top_place_ids)].copy()
            
            # Add collaborative scores
            score_dict = dict(sorted_scores)
            recommendations['collaborative_score'] = recommendations['Place_Id'].map(score_dict)
            
            return recommendations.sort_values('collaborative_score', ascending=False)
            
        except Exception as e:
            return self.places_df.nlargest(top_k, 'Rating')
    
    def get_hybrid_recommendations(self, user_id, description_query, city_filter, 
                                 price_range, weather_info, top_k=10):
        """Hybrid recommendations combining content-based and collaborative filtering"""
        
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(description_query, top_k*2)
        
        # Get collaborative recommendations  
        collaborative_recs = self.get_collaborative_recommendations(user_id, top_k*2)
        
        # Apply filters
        filtered_places = self.places_df.copy()
        
        # City filter
        if city_filter != 'Semua':
            filtered_places = filtered_places[filtered_places['City'].str.contains(city_filter, case=False, na=False)]
        
        # Price range filter
        min_price, max_price = price_range
        filtered_places = filtered_places[
            (filtered_places['Price'] >= min_price) & 
            (filtered_places['Price'] <= max_price)
        ]
        
        # Weather-based filtering
        if weather_info['is_rainy']:
            indoor_places = filtered_places[filtered_places['Outdoor/Indoor'] == 'Indoor']
            if not indoor_places.empty:
                filtered_places = indoor_places
        elif weather_info['temperature'] > 32:
            cool_places = filtered_places[
                (filtered_places['Category'].isin(['Bahari', 'Cagar Alam'])) |
                (filtered_places['Outdoor/Indoor'] == 'Indoor')
            ]
            if not cool_places.empty:
                filtered_places = cool_places
        
        # Combine recommendations
        final_scores = {}
        
        # Content-based scores
        for _, place in content_recs.iterrows():
            place_id = place['Place_Id']
            if place_id in filtered_places['Place_Id'].values:
                score = place.get('content_score', 0) * 0.4
                final_scores[place_id] = final_scores.get(place_id, 0) + score
        
        # Collaborative scores
        for _, place in collaborative_recs.iterrows():
            place_id = place['Place_Id']
            if place_id in filtered_places['Place_Id'].values:
                score = place.get('collaborative_score', 0) * 0.4
                final_scores[place_id] = final_scores.get(place_id, 0) + score
        
        # Popularity scores
        for _, place in filtered_places.iterrows():
            place_id = place['Place_Id']
            popularity_score = (place['Rating'] / 5.0) * (np.log1p(place['Rating_Count']) / 10) * 0.2
            final_scores[place_id] = final_scores.get(place_id, 0) + popularity_score
        
        # Sort by final scores
        sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_place_ids = [place_id for place_id, score in sorted_scores[:top_k]]
        
        # Get final recommendations
        recommendations = filtered_places[filtered_places['Place_Id'].isin(top_place_ids)].copy()
        
        if not recommendations.empty:
            score_dict = dict(sorted_scores)
            recommendations['hybrid_score'] = recommendations['Place_Id'].map(score_dict)
            recommendations = recommendations.sort_values('hybrid_score', ascending=False)
        
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

def show_auth_page():
    """Show authentication page"""
    st.markdown('<h1 class="main-header">🏛️ Enhanced Tourism Recommendation System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Initialize database
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager()
        
        # Authentication tabs
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### 🔑 Login ke Akun Anda")
            
            with st.form("login_form"):
                username = st.text_input("👤 Username")
                password = st.text_input("🔒 Password", type="password")
                
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
                reg_username = st.text_input("👤 Username")
                reg_email = st.text_input("📧 Email")
                reg_password = st.text_input("🔒 Password", type="password")
                reg_full_name = st.text_input("👨‍👩‍👧‍👦 Nama Lengkap")
                reg_location = st.selectbox("📍 Lokasi", [
                    "Semarang, Jawa Tengah",
                    "Yogyakarta, DIY"
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
    """Show enhanced dashboard page"""
    st.markdown('<h1 class="main-header">📊 Dashboard Destinasi Wisata</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_places = len(places_df)
        st.metric("🏛️ Total Destinasi", f"{total_places:,}")
    
    with col2:
        avg_rating = places_df['Rating'].mean()
        st.metric("⭐ Rating Rata-rata", f"{avg_rating:.1f}/5")
    
    with col3:
        price_range = f"Rp {places_df['Price'].min():,.0f} - Rp {places_df['Price'].max():,.0f}"
        st.metric("💰 Range Harga", price_range)
    
    with col4:
        total_categories = places_df['Category'].nunique()
        st.metric("🏷️ Kategori", f"{total_categories}")
    
    st.markdown("---")
    
    # Filters for data exploration
    st.markdown("### 🔍 Jelajahi Data Destinasi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city_filter = st.selectbox(
            "🏙️ Filter Kota:",
            ['Semua'] + sorted(places_df['City'].unique().tolist())
        )
    
    with col2:
        category_filter = st.selectbox(
            "🏷️ Filter Kategori:",
            ['Semua'] + sorted(places_df['Category'].unique().tolist())
        )
    
    with col3:
        indoor_outdoor_filter = st.selectbox(
            "🏠 Lokasi:",
            ['Semua', 'Indoor', 'Outdoor']
        )
    
    # Apply filters
    filtered_df = places_df.copy()
    
    if city_filter != 'Semua':
        filtered_df = filtered_df[filtered_df['City'] == city_filter]
    
    if category_filter != 'Semua':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    
    if indoor_outdoor_filter != 'Semua':
        filtered_df = filtered_df[filtered_df['Outdoor/Indoor'] == indoor_outdoor_filter]
    
    # Display filtered data
    st.markdown(f"### 📋 Data Destinasi Wisata ({len(filtered_df)} dari {len(places_df)} destinasi)")
    
    # Customizable columns
    display_columns = st.multiselect(
        "Pilih kolom yang ingin ditampilkan:",
        places_df.columns.tolist(),
        default=['Place_Name', 'City', 'Category', 'Rating', 'Price', 'Outdoor/Indoor']
    )
    
    if display_columns:
        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            height=400
        )
    
    # Analytics charts
    create_enhanced_analytics_charts(filtered_df, places_df)
    
    # Detailed place information
    st.markdown("### 🔍 Detail Destinasi")
    
    selected_place_name = st.selectbox(
        "Pilih destinasi untuk melihat detail:",
        filtered_df['Place_Name'].tolist()
    )
    
    if selected_place_name:
        selected_place = filtered_df[filtered_df['Place_Name'] == selected_place_name].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>🏛️ {selected_place['Place_Name']}</h3>
                <p><strong>📍 {selected_place['City']} | 🏷️ {selected_place['Category']}</strong></p>
                <p><strong>📝 Deskripsi:</strong> {selected_place['Description']}</p>
                <p><strong>🏠 Lokasi:</strong> {selected_place['Outdoor/Indoor']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("⭐ Rating", f"{selected_place['Rating']}/5")
            st.metric("💰 Harga", f"Rp {selected_place['Price']:,}")
            st.metric("📊 Jumlah Review", f"{selected_place['Rating_Count']:,}")
            
            # Rating form
            if st.button("⭐ Beri Rating", type="secondary"):
                with st.form("rating_form"):
                    user_rating = st.slider("Rating Anda:", 1, 5, 3)
                    if st.form_submit_button("Submit Rating"):
                        if db_manager.add_user_rating(user['id'], selected_place['Place_Id'], user_rating):
                            st.success("✅ Rating berhasil disimpan!")
                            st.rerun()

def show_recommendation_page(places_df, ratings_df, users_df, db_manager, user):
    """Show enhanced recommendation page"""
    st.markdown('<h1 class="main-header">🎯 Sistem Rekomendasi Enhanced</h1>', unsafe_allow_html=True)
    
    # Initialize recommendation system
    if 'rec_system' not in st.session_state:
        with st.spinner('🚀 Initializing recommendation system...'):
            rec_system = EnhancedRecommendationSystem(db_manager)
            rec_system.load_data(users_df, places_df, ratings_df)
            st.session_state.rec_system = rec_system
            st.success("✅ System initialized!")
    
    rec_system = st.session_state.rec_system
    
    # Input filters
    st.markdown("### 🎛️ Pengaturan Rekomendasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="filter-card">
            <h4>🔍 Filter Pencarian</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Description input
        description_query = st.text_area(
            "📝 Deskripsi destinasi yang diinginkan:",
            placeholder="Contoh: pantai yang indah dengan pasir putih, tempat wisata sejarah, museum budaya, dll.",
            height=100
        )
        
        # City selection
        city_filter = st.selectbox(
            "🏙️ Pilih Kota:",
            ['Semua', 'Semarang', 'Yogyakarta']
        )
        
        # Price range
        min_price = places_df['Price'].min()
        max_price = places_df['Price'].max()
        
        price_range = st.slider(
            "💰 Range Harga (Rupiah):",
            min_value=int(min_price),
            max_value=int(max_price),
            value=(int(min_price), int(max_price)),
            step=10000,
            format="Rp %d"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "📝 Jumlah Rekomendasi:",
            min_value=5,
            max_value=20,
            value=10
        )
    
    with col2:
        st.markdown("""
        <div class="weather-card">
            <h4>🌤️ Pengaturan Cuaca</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Weather input method
        weather_method = st.radio(
            "Pilih metode input cuaca:",
            ['🌐 Otomatis (API)', '✋ Manual']
        )
        
        if weather_method == '🌐 Otomatis (API)':
            # Extract city from user location
            user_city = 'Yogyakarta'
            if 'Semarang' in user['location']:
                user_city = 'Semarang'
            elif 'Yogyakarta' in user['location']:
                user_city = 'Yogyakarta'
            
            if st.button("🔄 Ambil Data Cuaca", type="secondary"):
                weather_info = rec_system.weather_service.get_weather_auto(user_city)
                st.session_state.weather_info = weather_info
            
            if 'weather_info' in st.session_state:
                weather = st.session_state.weather_info
                st.markdown(f"""
                **🌡️ Suhu:** {weather['temperature']}°C  
                **🌤️ Kondisi:** {weather['description']} {weather['icon']}  
                **📡 Sumber:** {weather['source']}
                """)
        
        else:
            # Manual weather input
            manual_temp = st.number_input(
                "🌡️ Suhu (°C):",
                min_value=15,
                max_value=40,
                value=27
            )
            
            manual_condition = st.selectbox(
                "🌤️ Kondisi Cuaca:",
                ['Cerah', 'Berawan', 'Mendung', 'Hujan Ringan', 'Hujan Lebat']
            )
            
            weather_info = rec_system.weather_service.get_weather_manual(manual_temp, manual_condition)
            st.session_state.weather_info = weather_info
            
            st.markdown(f"""
            **🌡️ Suhu:** {weather_info['temperature']}°C  
            **🌤️ Kondisi:** {weather_info['description']} {weather_info['icon']}  
            **📡 Sumber:** {weather_info['source']}
            """)
    
    # Generate recommendations
    if st.button("🎯 Generate Recommendations", type="primary", use_container_width=True):
        if 'weather_info' not in st.session_state:
            st.error("❌ Harap atur informasi cuaca terlebih dahulu!")
            return
        
        with st.spinner('🔍 Generating personalized recommendations...'):
            try:
                # Prepare user ID for collaborative filtering
                collab_user_id = user['id']
                if db_manager:
                    # Offset database user ID to avoid conflicts with CSV data
                    max_csv_user_id = ratings_df['User_Id'].max() if not ratings_df.empty else 0
                    collab_user_id = user['id'] + max_csv_user_id
                
                recommendations = rec_system.get_hybrid_recommendations(
                    user_id=collab_user_id,
                    description_query=description_query,
                    city_filter=city_filter,
                    price_range=price_range,
                    weather_info=st.session_state.weather_info,
                    top_k=num_recommendations
                )
                
                st.session_state.recommendations = recommendations
                st.session_state.query_info = {
                    'description': description_query,
                    'city': city_filter,
                    'price_range': price_range,
                    'weather': st.session_state.weather_info
                }
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")
    
    # Display recommendations
    if 'recommendations' in st.session_state:
        recommendations = st.session_state.recommendations
        query_info = st.session_state.query_info
        
        st.markdown("---")
        st.markdown("### 🎯 Rekomendasi untuk Anda")
        
        # Query summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🔍 Query:** {query_info['description'] if query_info['description'] else 'Semua destinasi'}  
            **🏙️ Kota:** {query_info['city']}  
            **💰 Budget:** Rp {query_info['price_range'][0]:,} - Rp {query_info['price_range'][1]:,}
            """)
        
        with col2:
            weather = query_info['weather']
            st.markdown(f"""
            **🌡️ Suhu:** {weather['temperature']}°C  
            **🌤️ Cuaca:** {weather['description']} {weather['icon']}  
            **📊 Hasil:** {len(recommendations)} destinasi ditemukan
            """)
        
        # Weather impact info
        if weather['is_rainy']:
            st.info("☔ Cuaca hujan terdeteksi. Prioritas diberikan pada destinasi indoor.")
        elif weather['temperature'] > 32:
            st.info("🌡️ Cuaca panas terdeteksi. Prioritas diberikan pada destinasi sejuk.")
        
        # Display recommendations
        if not recommendations.empty:
            for idx, (_, place) in enumerate(recommendations.iterrows(), 1):
                create_enhanced_recommendation_card(
                    place, idx, weather, 
                    show_rating_option=True,
                    db_manager=db_manager,
                    user_id=user['id']
                )
        else:
            st.warning("❌ Tidak ada rekomendasi yang sesuai dengan kriteria Anda. Coba ubah filter pencarian.")

def create_enhanced_recommendation_card(place, idx, weather, show_rating_option=False, db_manager=None, user_id=None):
    """Create enhanced recommendation card"""
    with st.container():
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>🏛️ {idx}. {place['Place_Name']}</h3>
            <p><strong>📍 {place['City']} | 🏷️ {place['Category']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Description
            description = str(place['Description'])
            if len(description) > 200:
                description = description[:200] + "..."
            st.markdown(f"📝 **Deskripsi:** {description}")
            
            # Additional info
            st.markdown(f"🏠 **Lokasi:** {place['Outdoor/Indoor']}")
            
            # Hybrid score if available
            if 'hybrid_score' in place and pd.notna(place['hybrid_score']):
                score_percent = (place['hybrid_score'] * 100)
                st.markdown(f"🎯 **Recommendation Score:** {score_percent:.1f}%")
        
        with col2:
            st.metric("⭐ Rating", f"{place['Rating']}/5")
            st.metric("💰 Harga", f"Rp {place['Price']:,}")
        
        with col3:
            st.metric("📊 Reviews", f"{place['Rating_Count']:,}")
            
            # Weather suitability
            if weather['is_rainy'] and place['Outdoor/Indoor'] == 'Indoor':
                st.success("☔ Cocok untuk cuaca hujan")
            elif weather['is_sunny'] and place['Outdoor/Indoor'] == 'Outdoor':
                st.success("☀️ Cocok untuk cuaca cerah")
            elif weather['temperature'] > 32 and place['Category'] in ['Bahari', 'Cagar Alam']:
                st.success("🌊 Destinasi sejuk")
        
        # Rating option
        if show_rating_option and db_manager and user_id:
            with st.expander("⭐ Beri Rating untuk Destinasi Ini"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    user_rating = st.slider(
                        "Rating Anda:", 1, 5, 3,
                        key=f"enhanced_rating_{place['Place_Id']}_{idx}",
                        help="1=Sangat Buruk, 5=Sangat Baik"
                    )
                
                with col2:
                    if st.button("💾 Submit", key=f"enhanced_submit_{place['Place_Id']}_{idx}"):
                        if db_manager.add_user_rating(user_id, place['Place_Id'], user_rating):
                            st.success("✅ Rating tersimpan!")
                            
                            # Update preferences
                            if user_rating >= 4:
                                db_manager.update_user_preferences(user_id, place['Category'], 1.0)
                            elif user_rating <= 2:
                                db_manager.update_user_preferences(user_id, place['Category'], 0.3)
                            
                            st.rerun()
                        else:
                            st.error("❌ Gagal menyimpan rating")
        
        st.markdown("---")

def create_enhanced_analytics_charts(filtered_df, full_df):
    """Create enhanced analytics charts"""
    st.markdown("### 📊 Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        st.markdown("#### 🏷️ Distribusi Kategori")
        category_counts = filtered_df['Category'].value_counts()
        fig_cat = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Kategori Destinasi Wisata"
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Rating vs Price scatter
        st.markdown("#### 💰 Rating vs Harga")
        fig_scatter = px.scatter(
            filtered_df,
            x='Price',
            y='Rating',
            size='Rating_Count',
            color='Category',
            hover_name='Place_Name',
            title="Analisis Rating vs Harga"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # City analysis
    st.markdown("#### 🏙️ Analisis per Kota")
    city_stats = filtered_df.groupby('City').agg({
        'Rating': ['mean', 'count'],
        'Price': 'mean',
        'Rating_Count': 'sum'
    }).round(2)
    
    city_stats.columns = ['Avg_Rating', 'Count', 'Avg_Price', 'Total_Reviews']
    city_stats = city_stats.reset_index()
    
    fig_city = px.bar(
        city_stats,
        x='City',
        y='Count',
        color='Avg_Rating',
        title="Jumlah Destinasi per Kota",
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_city, use_container_width=True)

def main():
    """Main application with enhanced navigation"""
    
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
    st.sidebar.markdown("# 🏛️ Enhanced Tourism System")
    st.sidebar.markdown(f"**👋 {user['full_name']}**")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "📖 Pilih Halaman:",
        [
            "📊 Dashboard Destinasi",
            "🎯 Sistem Rekomendasi"
        ],
        help="Navigasi antar halaman"
    )
    
    # User info
    st.sidebar.markdown("### ℹ️ Info Akun")
    st.sidebar.markdown(f"""
    - **Email:** {user['email']}
    - **Lokasi:** {user['location']}
    - **Umur:** {user['age']} tahun
    """)
    
    # User activity
    user_ratings = db_manager.get_user_ratings(user['id'])
    st.sidebar.markdown(f"- **Rating Diberikan:** {len(user_ratings)}")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="secondary"):
        st.session_state.authenticated = False
        st.session_state.user = None
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ['authenticated', 'user', 'db_manager']:
                del st.session_state[key]
        st.rerun()
    
    # Show selected page
    if page == "📊 Dashboard Destinasi":
        show_dashboard_page(places_df, ratings_df, users_df, db_manager, user)
    elif page == "🎯 Sistem Rekomendasi":
        show_recommendation_page(places_df, ratings_df, users_df, db_manager, user)

if __name__ == "__main__":
    main()
