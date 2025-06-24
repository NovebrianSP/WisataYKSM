import sqlite3
import hashlib
import pandas as pd

DB_PATH = "users.db"

def create_user_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    # Tambahkan tabel preferensi
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            kota TEXT,
            harga TEXT,
            cuaca TEXT,
            lokasi_terkini TEXT,
            kategori TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Tabel untuk favorit destinasi
def create_favorite_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_favorite (
            username TEXT NOT NULL,
            place_id TEXT NOT NULL,
            PRIMARY KEY (username, place_id)
        )
    """)
    conn.commit()
    conn.close()

def add_user(email, username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)", (email, username, password))
    conn.commit()
    conn.close()
    
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def user_exists(username, email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
    user = c.fetchone()
    conn.close()
    return user is not None

# Fungsi untuk menyimpan preferensi pengguna
def save_user_preference(username, kota, harga, cuaca, kategori, lokasi_terkini=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_preferences (username, kota, harga, cuaca, kategori, lokasi_terkini)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, kota, harga, cuaca, kategori, lokasi_terkini))
    conn.commit()
    conn.close()

def get_last_user_preference(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT kota, harga, cuaca, kategori, lokasi_terkini FROM user_preferences
        WHERE username=?
        ORDER BY timestamp DESC LIMIT 1
    """, (username,))
    pref = c.fetchone()
    conn.close()
    return pref

def get_all_user_preferences():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM user_preferences", conn)
    conn.close()
    return df

def get_similar_users(username, top_n=3):
    df = get_all_user_preferences()
    if df.empty or username not in df['username'].values:
        return []
    # Encode preferensi secara sederhana
    pref_cols = ['kota', 'harga', 'cuaca', 'kategori', 'lokasi_terkini']
    user_pref = df[df['username'] == username][pref_cols].mode().iloc[0]
    df = df[df['username'] != username]
    if df.empty:
        return []
    # Hitung kemiripan preferensi (jumlah field yang sama)
    def similarity(row):
        return sum(row[col] == user_pref[col] for col in pref_cols)
    df['similarity'] = df.apply(similarity, axis=1)
    similar_users = df[df['similarity'] == df['similarity'].max()]['username'].unique().tolist()
    return similar_users[:top_n]

# ----------- FAVORITE SYSTEM -----------
def add_favorite(username, place_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO user_favorite (username, place_id) VALUES (?, ?)", (username, place_id))
    conn.commit()
    conn.close()

def remove_favorite(username, place_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM user_favorite WHERE username=? AND place_id=?", (username, place_id))
    conn.commit()
    conn.close()

def is_favorite(username, place_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM user_favorite WHERE username=? AND place_id=?", (username, place_id))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_user_favorites(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT place_id FROM user_favorite WHERE username=?", (username,))
    result = [row[0] for row in c.fetchall()]
    conn.close()