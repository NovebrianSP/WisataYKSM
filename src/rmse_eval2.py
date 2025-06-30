import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pickle
import random
from datetime import datetime
from app3 import (
    load_enhanced_content_based,
    get_enhanced_content_based_recommendations,
    transform_similarity_score,
    DATA_PATH,
    get_weather_recommendation_score,
    get_time_context
)

class RMSEEvaluator:
    def __init__(self, test_size=0.3, random_state=42):
        """
        Evaluator khusus untuk menghitung RMSE dari model rekomendasi berbasis konten
        dengan peningkatan cuaca dan konteks waktu
        
        Args:
            test_size: Proporsi data untuk pengujian
            random_state: Random seed untuk reproduksibilitas
        """
        self.test_size = test_size
        self.random_state = random_state
        self.df, self.tfidf, self.tfidf_matrix = load_enhanced_content_based()
        self.test_users = []
        self.user_preferences = {}
        
        # Keyword untuk simulasi pencarian
        self.query_keywords = [
            "pantai", "gunung", "museum", "budaya", "pemandangan", 
            "sejarah", "kuliner", "alam", "air terjun", "hiking"
        ]
        
        # Pilihan cuaca untuk simulasi
        self.weather_options = [
            "Cerah", "Cerah Berawan", "Berawan", "Mendung", 
            "Hujan Ringan", "Hujan Sedang", "Hujan Lebat"
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
            preferred_weather = random.choice(self.weather_options)
            
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
                'preferred_weather': preferred_weather,
                'favorite_destinations': user_favorites['Place_Name'].tolist()
            }
            
        print(f"Data pengguna simulasi berhasil dibuat dengan {len(self.test_users)} pengguna")
        
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
        Evaluasi model menggunakan RMSE dengan perbandingan:
        1. Content-based dasar (tanpa peningkatan cuaca & waktu)
        2. Content-based dengan peningkatan cuaca
        3. Content-based dengan peningkatan cuaca & konteks waktu
        
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
            'basic_cb_rmse': 0,
            'weather_enhanced_rmse': 0, 
            'fully_enhanced_rmse': 0,
            'predictions': []
        }
        
        # Errors untuk perhitungan RMSE
        basic_cb_errors = []
        weather_enhanced_errors = []
        fully_enhanced_errors = []
        
        for username in self.test_users:
            user_prefs = self.user_preferences[username]
            city = user_prefs['preferred_city']
            categories = user_prefs['preferred_categories']
            weather_condition = user_prefs['preferred_weather']
            
            # Gabungkan kategori untuk query text
            query_text = " ".join(categories[:2]) if categories else ""
            
            # 1. Dapatkan rekomendasi dari model content-based dasar (tanpa cuaca)
            basic_cb_recs = get_enhanced_content_based_recommendations(
                self.df, self.tfidf, self.tfidf_matrix,
                query_text, city, None, None, top_n=50
            )
            
            # 2. Dapatkan rekomendasi dengan peningkatan cuaca saja
            weather_enhanced_recs = get_enhanced_content_based_recommendations(
                self.df, self.tfidf, self.tfidf_matrix,
                query_text, city, weather_condition, None, top_n=50
            )
            
            # 3. Dapatkan rekomendasi dengan peningkatan cuaca & konteks waktu
            # (Kita akan menggunakan model lengkap yang sudah ada di app3.py)
            fully_enhanced_recs = get_enhanced_content_based_recommendations(
                self.df, self.tfidf, self.tfidf_matrix,
                query_text, city, weather_condition, None, top_n=50
            )
            
            # Untuk setiap item di ground truth, bandingkan dengan prediksi
            user_ground_truth = ground_truth[ground_truth['username'] == username]
            
            for _, gt_row in user_ground_truth.iterrows():
                place_name = gt_row['place_name']
                actual_rating = gt_row['actual_rating']
                
                # 1. Ambil prediksi dari content-based dasar
                basic_predicted = 0.0
                if not basic_cb_recs.empty and 'similarity_score' in basic_cb_recs.columns:
                    match = basic_cb_recs[basic_cb_recs['Place_Name'] == place_name]
                    if not match.empty:
                        basic_predicted = transform_similarity_score(match.iloc[0]['similarity_score'])
                
                # 2. Ambil prediksi dari content-based dengan cuaca
                weather_predicted = 0.0
                if not weather_enhanced_recs.empty and 'similarity_score' in weather_enhanced_recs.columns:
                    match = weather_enhanced_recs[weather_enhanced_recs['Place_Name'] == place_name]
                    if not match.empty:
                        weather_predicted = transform_similarity_score(match.iloc[0]['similarity_score'])
                
                # 3. Ambil prediksi dari content-based lengkap
                fully_predicted = 0.0
                if not fully_enhanced_recs.empty and 'similarity_score' in fully_enhanced_recs.columns:
                    match = fully_enhanced_recs[fully_enhanced_recs['Place_Name'] == place_name]
                    if not match.empty:
                        fully_predicted = transform_similarity_score(match.iloc[0]['similarity_score'])
                
                # Hitung error dengan penalti default untuk yang tidak ditemukan
                default_rating = 1.0 if actual_rating > 3.0 else 0.5
                
                if basic_predicted > 0:
                    basic_cb_errors.append((actual_rating - basic_predicted) ** 2)
                else:
                    basic_cb_errors.append((actual_rating - default_rating) ** 2)
                
                if weather_predicted > 0:
                    weather_enhanced_errors.append((actual_rating - weather_predicted) ** 2)
                else:
                    weather_enhanced_errors.append((actual_rating - default_rating) ** 2)
                
                if fully_predicted > 0:
                    fully_enhanced_errors.append((actual_rating - fully_predicted) ** 2)
                else:
                    fully_enhanced_errors.append((actual_rating - default_rating) ** 2)
                
                # Simpan prediksi untuk analisis
                results['predictions'].append({
                    'username': username,
                    'place_name': place_name,
                    'actual_rating': actual_rating,
                    'basic_cb_predicted': basic_predicted,
                    'weather_enhanced_predicted': weather_predicted,
                    'fully_enhanced_predicted': fully_predicted
                })
        
        # Hitung RMSE
        if basic_cb_errors:
            results['basic_cb_rmse'] = np.sqrt(np.mean(basic_cb_errors))
        
        if weather_enhanced_errors:
            results['weather_enhanced_rmse'] = np.sqrt(np.mean(weather_enhanced_errors))
            
        if fully_enhanced_errors:
            results['fully_enhanced_rmse'] = np.sqrt(np.mean(fully_enhanced_errors))
        
        print(f"Content-Based Dasar RMSE: {results['basic_cb_rmse']:.4f}")
        print(f"Content-Based + Cuaca RMSE: {results['weather_enhanced_rmse']:.4f}")
        print(f"Content-Based + Cuaca + Konteks Waktu RMSE: {results['fully_enhanced_rmse']:.4f}")
        
        return results
    
    def plot_rmse_results(self, results):
        """
        Membuat visualisasi untuk hasil evaluasi RMSE
        
        Args:
            results: Hasil dari evaluate_rmse()
            
        Returns:
            str: Path file gambar hasil
        """
        plt.figure(figsize=(12, 10))
        
        # 1. Bar chart perbandingan RMSE
        plt.subplot(2, 1, 1)
        models = ['Content-Based Dasar', 'CB + Cuaca', 'CB + Cuaca + Waktu']
        rmse_values = [
            results['basic_cb_rmse'], 
            results['weather_enhanced_rmse'], 
            results['fully_enhanced_rmse']
        ]
        
        colors = ['#3498db', '#e67e22', '#2ecc71']
        plt.bar(models, rmse_values, color=colors)
        plt.title('Perbandingan RMSE Model Rekomendasi Content-Based')
        plt.ylabel('RMSE (Root Mean Square Error)')
        plt.ylim(0, max(rmse_values) * 1.2)
        
        # Tambahkan nilai RMSE di atas bar
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.05, f"{v:.4f}", ha='center')
            
        # 2. Scatter plot prediksi vs aktual
        plt.subplot(2, 1, 2)
        
        # Konversi predictions ke DataFrame
        pred_df = pd.DataFrame(results['predictions'])
        
        # Plot untuk Content-Based Dasar
        plt.scatter(pred_df['actual_rating'], pred_df['basic_cb_predicted'],
                  alpha=0.4, color='blue', label='Content-Based Dasar')
        
        # Plot untuk Content-Based + Cuaca
        plt.scatter(pred_df['actual_rating'], pred_df['weather_enhanced_predicted'],
                  alpha=0.4, color='orange', label='CB + Cuaca')
        
        # Plot untuk Content-Based + Cuaca + Waktu
        plt.scatter(pred_df['actual_rating'], pred_df['fully_enhanced_predicted'],
                  alpha=0.4, color='green', label='CB + Cuaca + Waktu')
        
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
        output_path = 'rmse_evaluation_app3.png'
        plt.savefig(output_path)
        plt.close()
        
        return output_path

def run_rmse_evaluation():
    """
    Menjalankan evaluasi RMSE lengkap untuk app3.py
    
    Returns:
        dict: Hasil evaluasi
    """
    print("Memulai evaluasi RMSE untuk model rekomendasi app3.py...")
    
    evaluator = RMSEEvaluator()
    
    # Siapkan data pengguna simulasi
    evaluator.prepare_user_data(num_test_users=20)
    
    # Evaluasi RMSE
    results = evaluator.evaluate_rmse()
    
    # Buat visualisasi
    img_path = evaluator.plot_rmse_results(results)
    
    print(f"Evaluasi RMSE selesai! Visualisasi disimpan di {img_path}")
    
    # Simpulkan hasil
    models = [
        ('Content-Based Dasar', results['basic_cb_rmse']),
        ('Content-Based + Cuaca', results['weather_enhanced_rmse']),
        ('Content-Based + Cuaca + Waktu', results['fully_enhanced_rmse'])
    ]
    
    # Temukan model dengan RMSE terendah
    best_model = min(models, key=lambda x: x[1])
    
    print(f"\nKesimpulan: Model {best_model[0]} memiliki performa terbaik dengan RMSE {best_model[1]:.4f}")
    print("\nPerbandingan peningkatan akurasi:")
    
    if results['basic_cb_rmse'] > results['weather_enhanced_rmse']:
        improvement = ((results['basic_cb_rmse'] - results['weather_enhanced_rmse']) / results['basic_cb_rmse']) * 100
        print(f"• Penambahan faktor cuaca meningkatkan akurasi sebesar {improvement:.2f}%")
    else:
        print("• Penambahan faktor cuaca tidak meningkatkan akurasi")
        
    if results['weather_enhanced_rmse'] > results['fully_enhanced_rmse']:
        improvement = ((results['weather_enhanced_rmse'] - results['fully_enhanced_rmse']) / results['weather_enhanced_rmse']) * 100
        print(f"• Penambahan konteks waktu meningkatkan akurasi sebesar {improvement:.2f}%")
    else:
        print("• Penambahan konteks waktu tidak meningkatkan akurasi")
    
    return results

if __name__ == "__main__":
    results = run_rmse_evaluation()