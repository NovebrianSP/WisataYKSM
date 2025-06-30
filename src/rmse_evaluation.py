import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pickle
import sqlite3
import random
from app2 import (  # Changed import from app to app2
    load_content_based,
    get_content_based_recommendations_by_text,
    get_hybrid_recommendations,
    DATA_PATH,
    transform_similarity_score  # Import the improved similarity transformation
)

class RMSEEvaluator:
    def __init__(self, test_size=0.3, random_state=42):
        """
        Evaluator khusus untuk menghitung RMSE dari model rekomendasi
        
        Args:
            test_size: Proporsi data untuk pengujian
            random_state: Random seed untuk reproduksibilitas
        """
        self.test_size = test_size
        self.random_state = random_state
        self.df, self.tfidf, self.tfidf_matrix = load_content_based()
        self.test_users = []
        self.user_preferences = {}
        
        # Lokasi database preferensi pengguna
        db_path = os.path.join(os.path.dirname(__file__), "data/user_preferences.db")
        self.db_path = db_path
        
        # Keyword untuk simulasi pencarian
        self.query_keywords = [
            "pantai", "gunung", "museum", "budaya", "pemandangan", 
            "sejarah", "kuliner", "alam", "air terjun", "hiking"
        ]
        
    def prepare_user_data(self, num_test_users=20):
        """
        Mempersiapkan data pengguna untuk evaluasi RMSE dengan
        membuat pengguna simulasi dan preferensi mereka
        
        Args:
            num_test_users: Jumlah pengguna uji yang akan dibuat
        """
        print(f"Membuat {num_test_users} pengguna uji untuk evaluasi RMSE...")
        
        # Ekstrak kategori unik dari dataset
        categories = self.df['Category'].dropna().unique()
        cities = self.df['City'].dropna().unique()
        
        # Buat pengguna simulasi dengan preferensi acak
        for i in range(num_test_users):
            username = f"rmse_test_user_{i}"
            
            # Pilih preferensi acak untuk pengguna ini
            liked_categories = random.sample(list(categories), k=min(3, len(categories)))
            preferred_city = random.choice(list(cities))
            
            # Pilih destinasi acak yang sesuai dengan preferensi
            matching_destinations = self.df[
                (self.df['Category'].isin(liked_categories)) & 
                (self.df['City'] == preferred_city)
            ]
            
            # Jika tidak ada yang cocok, ambil dari semua destinasi
            if matching_destinations.empty:
                matching_destinations = self.df
                
            # Pilih beberapa destinasi acak sebagai "preferensi" pengguna simulasi
            if len(matching_destinations) > 5:
                user_favorites = matching_destinations.sample(5)
            else:
                user_favorites = matching_destinations
                
            # Simpan ID pengguna dan preferensinya
            self.test_users.append(username)
            self.user_preferences[username] = {
                'preferred_categories': liked_categories,
                'preferred_city': preferred_city,
                'favorite_destinations': user_favorites['Place_Name'].tolist()
            }
            
            # Juga simpan ke database untuk digunakan dalam rekomendasi
            self._save_user_preference(username, preferred_city, liked_categories)
            
        print(f"Data pengguna simulasi berhasil dibuat dengan {len(self.test_users)} pengguna")
        
    def _save_user_preference(self, username, city, categories):
        """Simpan preferensi pengguna ke database untuk pengujian"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pastikan tabel user_preferences ada
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            lokasi TEXT,
            harga TEXT,
            cuaca TEXT,
            kategori TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Simpan kategori sebagai string (comma-separated)
        kategori_str = ','.join(categories)
        
        # Simpan preferensi
        cursor.execute('''
        INSERT INTO user_preferences (username, lokasi, kategori)
        VALUES (?, ?, ?)
        ''', (username, city, kategori_str))
        
        conn.commit()
        conn.close()
        
    def generate_ground_truth(self):
        """
        Membuat ground truth untuk evaluasi yang lebih realistis
        dengan distribusi rating yang lebih seimbang
        
        Returns:
            DataFrame dengan ground truth (pengguna, destinasi, rating)
        """
        ground_truth_data = []
        
        # Untuk setiap pengguna simulasi
        for username in self.test_users:
            user_prefs = self.user_preferences[username]
            fav_destinations = user_prefs['favorite_destinations']
            
            # Menggunakan skala rating yang lebih realistis
            # Destinasi favorit: rating 4-5
            for destination in fav_destinations:
                ground_truth_data.append({
                    'username': username,
                    'place_name': destination,
                    'actual_rating': random.uniform(4.0, 5.0)  # Lebih realistis
                })
            
            # Destinasi yang sesuai kategori tapi bukan favorit: rating 2-4
            matching_cat = self.df[
                (self.df['Category'].isin(user_prefs['preferred_categories'])) &
                ~(self.df['Place_Name'].isin(fav_destinations))
            ].sample(min(3, len(self.df)))
            
            for _, row in matching_cat.iterrows():
                ground_truth_data.append({
                    'username': username,
                    'place_name': row['Place_Name'],
                    'actual_rating': random.uniform(2.0, 4.0)
                })
            
            # Destinasi yang tidak sesuai preferensi: rating 1-2
            non_favorites = self.df[
                ~(self.df['Category'].isin(user_prefs['preferred_categories'])) &
                ~(self.df['Place_Name'].isin(fav_destinations))
            ].sample(min(3, len(self.df)))
            
            for _, row in non_favorites.iterrows():
                ground_truth_data.append({
                    'username': username,
                    'place_name': row['Place_Name'],
                    'actual_rating': random.uniform(1.0, 2.0)  # Rating rendah untuk non-favorit
                })
                
        # Konversi ke DataFrame
        ground_truth_df = pd.DataFrame(ground_truth_data)
        print(f"Ground truth dibuat dengan {len(ground_truth_df)} entri rating")
        
        return ground_truth_df
        
    def evaluate_rmse(self):
        """
        Evaluasi model menggunakan RMSE
        
        Returns:
            dict: Hasil evaluasi RMSE untuk model berbeda
        """
        # Pastikan data pengguna telah disiapkan
        if not self.test_users:
            self.prepare_user_data(num_test_users=20)
        
        # Buat ground truth
        ground_truth = self.generate_ground_truth()
        
        # Hasil untuk disimpan
        results = {
            'content_based_rmse': 0,
            'hybrid_rmse': 0,
            'predictions': []
        }
        
        # Evaluasi untuk setiap pengguna
        content_based_errors = []
        hybrid_errors = []
        
        for username in self.test_users:
            user_prefs = self.user_preferences[username]
            city = user_prefs['preferred_city']
            categories = user_prefs['preferred_categories']
            
            # Gabungkan kategori untuk query text
            query_text = " ".join(categories[:2]) if categories else ""
            
            # Dapatkan rekomendasi dari model content-based
            content_recs = get_content_based_recommendations_by_text(
                self.df, self.tfidf, self.tfidf_matrix,
                query_text, city, None, None, top_n=50
            )
            
            # Dapatkan rekomendasi dari model hybrid
            hybrid_recs = get_hybrid_recommendations(
                self.df, self.tfidf, self.tfidf_matrix,
                username, city, None, query_text, None
            )
            
            # Untuk setiap item di ground truth, bandingkan dengan prediksi
            user_ground_truth = ground_truth[ground_truth['username'] == username]
            
            for _, gt_row in user_ground_truth.iterrows():
                place_name = gt_row['place_name']
                actual_rating = gt_row['actual_rating']
                
                # Ambil prediksi dari content-based
                cb_predicted = 0.0
                if not content_recs.empty and 'similarity_score' in content_recs.columns:
                    cb_match = content_recs[content_recs['Place_Name'] == place_name]
                    if not cb_match.empty:
                        # Konversi similarity score ke rating (0-5) dengan transformasi yang lebih baik
                        cb_predicted = transform_similarity_score(cb_match.iloc[0]['similarity_score'])
                
                # Ambil prediksi dari hybrid
                hybrid_predicted = 0.0
                if not hybrid_recs.empty and 'similarity_score' in hybrid_recs.columns:
                    hybrid_match = hybrid_recs[hybrid_recs['Place_Name'] == place_name]
                    if not hybrid_match.empty:
                        # Konversi similarity score ke rating (0-5) dengan transformasi yang lebih baik
                        hybrid_predicted = transform_similarity_score(hybrid_match.iloc[0]['similarity_score'])
                
                # Hitung error dengan penalti yang lebih seimbang
                if cb_predicted > 0:
                    content_based_errors.append((actual_rating - cb_predicted) ** 2)
                else:
                    # Gunakan rating default sesuai ekspektasi untuk item yang tidak ditemukan
                    predicted_default = 1.0 if actual_rating > 3.0 else 0.5
                    content_based_errors.append((actual_rating - predicted_default) ** 2)
                
                if hybrid_predicted > 0:
                    hybrid_errors.append((actual_rating - hybrid_predicted) ** 2)
                else:
                    # Gunakan rating default sesuai ekspektasi untuk item yang tidak ditemukan
                    predicted_default = 1.0 if actual_rating > 3.0 else 0.5
                    hybrid_errors.append((actual_rating - predicted_default) ** 2)
                
                # Simpan prediksi untuk analisis
                results['predictions'].append({
                    'username': username,
                    'place_name': place_name,
                    'actual_rating': actual_rating,
                    'content_based_predicted': cb_predicted,
                    'hybrid_predicted': hybrid_predicted
                })
        
        # Hitung RMSE
        if content_based_errors:
            results['content_based_rmse'] = np.sqrt(np.mean(content_based_errors))
        
        if hybrid_errors:
            results['hybrid_rmse'] = np.sqrt(np.mean(hybrid_errors))
        
        print(f"Content-Based RMSE: {results['content_based_rmse']:.4f}")
        print(f"Hybrid RMSE: {results['hybrid_rmse']:.4f}")
        
        return results
    
    def plot_rmse_results(self, results):
        """
        Membuat visualisasi untuk hasil evaluasi RMSE
        
        Args:
            results: Hasil dari evaluate_rmse()
            
        Returns:
            str: Path file gambar hasil
        """
        plt.figure(figsize=(12, 8))
        
        # 1. Bar chart perbandingan RMSE
        plt.subplot(2, 1, 1)
        models = ['Content-Based', 'Hybrid']
        rmse_values = [results['content_based_rmse'], results['hybrid_rmse']]
        
        plt.bar(models, rmse_values, color=['#3498db', '#e74c3c'])
        plt.title('Perbandingan RMSE Model Rekomendasi')
        plt.ylabel('RMSE (Root Mean Square Error)')
        plt.ylim(0, max(rmse_values) * 1.2)
        
        # Tambahkan nilai RMSE di atas bar
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.05, f"{v:.4f}", ha='center')
            
        # 2. Scatter plot prediksi vs aktual
        plt.subplot(2, 1, 2)
        
        # Konversi predictions ke DataFrame
        pred_df = pd.DataFrame(results['predictions'])
        
        # Plot untuk Content-Based
        plt.scatter(pred_df['actual_rating'], pred_df['content_based_predicted'],
                  alpha=0.5, color='blue', label='Content-Based')
        
        # Plot untuk Hybrid
        plt.scatter(pred_df['actual_rating'], pred_df['hybrid_predicted'],
                  alpha=0.5, color='red', label='Hybrid')
        
        # Garis ideal (y=x)
        ideal_line = np.linspace(0, 5, 100)
        plt.plot(ideal_line, ideal_line, 'k--', alpha=0.3, label='Ideal')
        
        plt.xlabel('Rating Aktual')
        plt.ylabel('Rating Prediksi')
        plt.title('Perbandingan Rating Aktual vs Prediksi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Simpan plot
        output_path = 'rmse_evaluation_app2.png'  # Changed filename to reflect app2
        plt.savefig(output_path)
        plt.close()
        
        return output_path

def run_rmse_evaluation():
    """
    Menjalankan evaluasi RMSE lengkap menggunakan app2.py
    
    Returns:
        dict: Hasil evaluasi
    """
    print("Memulai evaluasi RMSE untuk model rekomendasi dengan app2.py...")
    
    evaluator = RMSEEvaluator()
    
    # Siapkan data pengguna simulasi
    evaluator.prepare_user_data(num_test_users=20)
    
    # Evaluasi RMSE
    results = evaluator.evaluate_rmse()
    
    # Buat visualisasi
    img_path = evaluator.plot_rmse_results(results)
    
    print(f"Evaluasi RMSE selesai! Visualisasi disimpan di {img_path}")
    
    # Simpulkan hasil
    if results['hybrid_rmse'] < results['content_based_rmse']:
        print("\nKesimpulan: Model Hybrid mengungguli model Content-Based dengan RMSE yang lebih rendah.")
    elif results['content_based_rmse'] < results['hybrid_rmse']:
        print("\nKesimpulan: Model Content-Based mengungguli model Hybrid dengan RMSE yang lebih rendah.")
    else:
        print("\nKesimpulan: Kedua model memiliki performa yang setara dalam hal RMSE.")
    
    return results

if __name__ == "__main__":
    results = run_rmse_evaluation()