import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import hashlib

class RecommendationEvaluator:
    def __init__(self, data_path, db_path):
        self.data_path = data_path
        self.db_path = db_path
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.load_data()
        
    def load_data(self):
        """Load dataset and prepare TF-IDF model"""
        self.df = pd.read_csv(self.data_path)
        
        # Pastikan kolom numerik
        self.df["Rating"] = pd.to_numeric(self.df["Rating"], errors="coerce")
        self.df["Price"] = pd.to_numeric(self.df["Price"], errors="coerce")
        
        # Create combined text for TF-IDF
        if "Description" in self.df.columns:
            self.df["combined_text"] = (self.df["Place_Name"] + " " + 
                                      self.df["Category"].fillna("") + " " + 
                                      self.df["Description"].fillna(""))
        else:
            self.df["combined_text"] = (self.df["Place_Name"] + " " + 
                                      self.df["Category"].fillna(""))
        
        # Build TF-IDF model
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["combined_text"])
        
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_test_users(self, n_users=50):
        """Create synthetic test users with preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        
        # Create preferences table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                lokasi TEXT,
                harga TEXT,
                cuaca TEXT,
                kategori TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create test data
        test_users = []
        categories = self.df['Category'].dropna().unique().tolist()
        cities = self.df['City'].dropna().unique().tolist()
        harga_options = ["Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"]
        cuaca_options = ["Cerah", "Berawan", "Hujan Ringan", "Mendung"]
        
        for i in range(n_users):
            username = f"test_user_{i}"
            email = f"test_{i}@example.com"
            password = self.hash_password("password123")
            
            # Insert user
            try:
                cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)",
                             (email, username, password))
            except sqlite3.IntegrityError:
                continue  # User already exists
            
            # Create random preferences for each user
            n_prefs = np.random.randint(3, 8)  # 3-7 preferences per user
            for _ in range(n_prefs):
                lokasi = np.random.choice(cities)
                harga = np.random.choice(harga_options)
                cuaca = np.random.choice(cuaca_options)
                kategori = np.random.choice(categories)
                
                cursor.execute('''
                    INSERT INTO user_preferences (username, lokasi, harga, cuaca, kategori)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, lokasi, harga, cuaca, kategori))
            
            test_users.append({
                'username': username,
                'preferences': {
                    'lokasi': np.random.choice(cities),
                    'harga': np.random.choice(harga_options),
                    'cuaca': np.random.choice(cuaca_options),
                    'kategori': np.random.choice(categories)
                }
            })
        
        conn.commit()
        conn.close()
        return test_users
    
    def get_similar_users(self, username):
        """Get similar users based on preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current user preferences
        cursor.execute('''
            SELECT lokasi, harga, cuaca, kategori FROM user_preferences 
            WHERE username = ?
        ''', (username,))
        user_prefs = cursor.fetchall()
        
        if not user_prefs:
            conn.close()
            return []
        
        # Get all other users' preferences
        cursor.execute('''
            SELECT DISTINCT username FROM user_preferences 
            WHERE username != ?
        ''', (username,))
        other_users = [row[0] for row in cursor.fetchall()]
        
        similar_users = []
        user_pref_set = set(user_prefs)
        
        for other_user in other_users:
            cursor.execute('''
                SELECT lokasi, harga, cuaca, kategori FROM user_preferences 
                WHERE username = ?
            ''', (other_user,))
            other_prefs = cursor.fetchall()
            other_pref_set = set(other_prefs)
            
            # Calculate Jaccard similarity
            intersection = len(user_pref_set.intersection(other_pref_set))
            union = len(user_pref_set.union(other_pref_set))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.3:  # Threshold for similarity
                similar_users.append(other_user)
        
        conn.close()
        return similar_users
    
    def get_content_based_recommendations(self, user_text, lokasi=None, cuaca=None, harga=None, top_n=50):
        """Get content-based recommendations"""
        if not user_text.strip():
            temp_df = self.df.copy()
            temp_df['similarity_score'] = 0.5
        else:
            user_vector = self.tfidf.transform([user_text])
            sim_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
            temp_df = self.df.copy()
            temp_df['similarity_score'] = sim_scores
        
        # Apply filters
        if lokasi:
            temp_df = temp_df[temp_df["City"] == lokasi]
        
        if harga:
            if harga == "Murah (<20000)":
                temp_df = temp_df[temp_df["Price"] < 20000]
            elif harga == "Sedang (20000-50000)":
                temp_df = temp_df[(temp_df["Price"] >= 20000) & (temp_df["Price"] <= 50000)]
            elif harga == "Mahal (>50000)":
                temp_df = temp_df[temp_df["Price"] > 50000]
        
        if cuaca:
            temp_df = temp_df[temp_df["Outdoor/Indoor"].notna()]
            if cuaca in ["Cerah", "Cerah Berawan", "Berawan"]:
                temp_df = temp_df[temp_df["Outdoor/Indoor"].isin(["Outdoor", "Indoor"])]
            else:
                temp_df = temp_df[temp_df["Outdoor/Indoor"] == "Indoor"]
        
        return temp_df.sort_values('similarity_score', ascending=False).head(top_n)
    
    def get_hybrid_recommendations(self, username, lokasi, cuaca, user_text=None, harga=None):
        """Get hybrid recommendations combining collaborative and content-based"""
        similar_users = self.get_similar_users(username)
        
        if similar_users:
            # Get collaborative filtering preferences
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in similar_users])
            cursor.execute(f'''
                SELECT lokasi, harga, cuaca, kategori FROM user_preferences 
                WHERE username IN ({placeholders})
            ''', similar_users)
            
            prefs = cursor.fetchall()
            conn.close()
            
            if prefs:
                # Get most common preferences
                prefs_df = pd.DataFrame(prefs, columns=['lokasi', 'harga', 'cuaca', 'kategori'])
                filter_cuaca = prefs_df['cuaca'].mode()[0] if not prefs_df['cuaca'].mode().empty else cuaca
                filter_text = prefs_df['kategori'].mode()[0] if not prefs_df['kategori'].mode().empty else user_text
                filter_harga = prefs_df['harga'].mode()[0] if not prefs_df['harga'].mode().empty else harga
                
                # Get collaborative results
                collaborative = self.get_content_based_recommendations(
                    filter_text if filter_text else "", lokasi, filter_cuaca, filter_harga
                )
                collaborative_ids = collaborative['Place_Name'].tolist()
            else:
                collaborative_ids = []
        else:
            collaborative_ids = []
        
        # Get content-based results
        content_based = self.get_content_based_recommendations(
            user_text if user_text else "", lokasi, cuaca, harga
        )
        content_ids = content_based['Place_Name'].tolist()
        
        # Combine results (prioritize overlap)
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
        
        hybrid_df = self.df[self.df['Place_Name'].isin(hybrid_ids)]
        if 'similarity_score' in content_based.columns:
            hybrid_df = hybrid_df.merge(
                content_based[['Place_Name', 'similarity_score']], 
                on='Place_Name', how='left'
            )
            hybrid_df = hybrid_df.sort_values('similarity_score', ascending=False)
        
        return hybrid_df
    
    def create_ground_truth(self, test_users, n_relevant=10):
        """Create ground truth for evaluation based on user preferences"""
        ground_truth = {}
        
        for user in test_users:
            username = user['username']
            prefs = user['preferences']
            
            # Find destinations that match user preferences
            relevant_destinations = self.df.copy()
            
            # Filter by location preference
            if prefs['lokasi']:
                relevant_destinations = relevant_destinations[
                    relevant_destinations['City'] == prefs['lokasi']
                ]
            
            # Filter by price preference
            if prefs['harga'] == "Murah (<20000)":
                relevant_destinations = relevant_destinations[relevant_destinations['Price'] < 20000]
            elif prefs['harga'] == "Sedang (20000-50000)":
                relevant_destinations = relevant_destinations[
                    (relevant_destinations['Price'] >= 20000) & 
                    (relevant_destinations['Price'] <= 50000)
                ]
            elif prefs['harga'] == "Mahal (>50000)":
                relevant_destinations = relevant_destinations[relevant_destinations['Price'] > 50000]
            
            # Filter by category preference
            if prefs['kategori']:
                relevant_destinations = relevant_destinations[
                    relevant_destinations['Category'] == prefs['kategori']
                ]
            
            # Select top destinations by rating
            relevant_destinations = relevant_destinations.sort_values('Rating', ascending=False)
            ground_truth[username] = relevant_destinations['Place_Name'].head(n_relevant).tolist()
        
        return ground_truth
    
    def evaluate_recommendations(self, recommendations, ground_truth, k=10):
        """Evaluate recommendations using precision, recall, and F1-score"""
        precisions = []
        recalls = []
        f1_scores = []
        
        for username, recommended in recommendations.items():
            if username not in ground_truth:
                continue
                
            relevant = set(ground_truth[username])
            recommended_set = set(recommended[:k])  # Top-k recommendations
            
            if len(recommended_set) == 0:
                precision = 0
                recall = 0
                f1 = 0
            else:
                true_positives = len(relevant.intersection(recommended_set))
                precision = true_positives / len(recommended_set)
                recall = true_positives / len(relevant) if len(relevant) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'precision_std': np.std(precisions),
            'recall_std': np.std(recalls),
            'f1_std': np.std(f1_scores)
        }
    
    def run_evaluation(self, n_users=50, k=10):
        """Run complete evaluation"""
        print("Creating test users...")
        test_users = self.create_test_users(n_users)
        
        print("Creating ground truth...")
        ground_truth = self.create_ground_truth(test_users)
        
        print("Generating content-based recommendations...")
        content_recommendations = {}
        for user in test_users:
            username = user['username']
            prefs = user['preferences']
            
            recommendations = self.get_content_based_recommendations(
                prefs['kategori'], 
                prefs['lokasi'], 
                prefs['cuaca'], 
                prefs['harga']
            )
            content_recommendations[username] = recommendations['Place_Name'].tolist()
        
        print("Generating hybrid recommendations...")
        hybrid_recommendations = {}
        for user in test_users:
            username = user['username']
            prefs = user['preferences']
            
            recommendations = self.get_hybrid_recommendations(
                username,
                prefs['lokasi'],
                prefs['cuaca'],
                prefs['kategori'],
                prefs['harga']
            )
            hybrid_recommendations[username] = recommendations['Place_Name'].tolist()
        
        print("Evaluating content-based model...")
        content_metrics = self.evaluate_recommendations(content_recommendations, ground_truth, k)
        
        print("Evaluating hybrid model...")
        hybrid_metrics = self.evaluate_recommendations(hybrid_recommendations, ground_truth, k)
        
        return {
            'content_based': content_metrics,
            'hybrid': hybrid_metrics,
            'test_users_count': len(test_users),
            'k': k
        }

def main():
    # Setup paths
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "data/destinasi-wisata-YKSM.csv")
    DB_PATH = os.path.join(BASE_DIR, "data/test_users.db")
    
    # Create evaluator
    evaluator = RecommendationEvaluator(DATA_PATH, DB_PATH)
    
    # Run evaluation
    print("Starting recommendation system evaluation...")
    results = evaluator.run_evaluation(n_users=100, k=10)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nTest Setup:")
    print(f"- Number of test users: {results['test_users_count']}")
    print(f"- Top-K recommendations evaluated: {results['k']}")
    
    print(f"\nContent-Based Model:")
    cb_metrics = results['content_based']
    print(f"- Precision: {cb_metrics['precision']:.4f} (±{cb_metrics['precision_std']:.4f})")
    print(f"- Recall: {cb_metrics['recall']:.4f} (±{cb_metrics['recall_std']:.4f})")
    print(f"- F1-Score: {cb_metrics['f1_score']:.4f} (±{cb_metrics['f1_std']:.4f})")
    
    print(f"\nHybrid Model:")
    hybrid_metrics = results['hybrid']
    print(f"- Precision: {hybrid_metrics['precision']:.4f} (±{hybrid_metrics['precision_std']:.4f})")
    print(f"- Recall: {hybrid_metrics['recall']:.4f} (±{hybrid_metrics['recall_std']:.4f})")
    print(f"- F1-Score: {hybrid_metrics['f1_score']:.4f} (±{hybrid_metrics['f1_std']:.4f})")
    
    print(f"\nImprovement (Hybrid vs Content-Based):")
    precision_improvement = ((hybrid_metrics['precision'] - cb_metrics['precision']) / cb_metrics['precision'] * 100) if cb_metrics['precision'] > 0 else 0
    recall_improvement = ((hybrid_metrics['recall'] - cb_metrics['recall']) / cb_metrics['recall'] * 100) if cb_metrics['recall'] > 0 else 0
    f1_improvement = ((hybrid_metrics['f1_score'] - cb_metrics['f1_score']) / cb_metrics['f1_score'] * 100) if cb_metrics['f1_score'] > 0 else 0
    
    print(f"- Precision: {precision_improvement:+.2f}%")
    print(f"- Recall: {recall_improvement:+.2f}%")
    print(f"- F1-Score: {f1_improvement:+.2f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()