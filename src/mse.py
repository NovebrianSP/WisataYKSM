import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import hashlib
import random

class MSERecommendationEvaluator:
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
    
    def create_user_rating_matrix(self, n_users=100, n_ratings_per_user=15):
        """Create synthetic user-rating matrix for MSE evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DROP TABLE IF EXISTS user_ratings')
        cursor.execute('DROP TABLE IF EXISTS users')
        cursor.execute('DROP TABLE IF EXISTS user_preferences')
        
        # Create ratings table
        cursor.execute('''
            CREATE TABLE user_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                place_name TEXT NOT NULL,
                rating REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create users table
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        
        # Create user preferences table
        cursor.execute('''
            CREATE TABLE user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                lokasi TEXT,
                harga TEXT,
                cuaca TEXT,
                kategori TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Generate synthetic users and ratings
        categories = self.df['Category'].dropna().unique().tolist()
        cities = self.df['City'].dropna().unique().tolist()
        harga_options = ["Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"]
        cuaca_options = ["Cerah", "Berawan", "Hujan Ringan", "Mendung"]
        
        user_rating_data = []
        
        for i in range(n_users):
            username = f"mse_user_{i}"
            email = f"mse_{i}@example.com"
            password = self.hash_password("password123")
            
            # Insert user
            cursor.execute("INSERT INTO users (email, username, password) VALUES (?, ?, ?)",
                         (email, username, password))
            
            # Create user preferences
            user_category_pref = random.choice(categories)
            user_city_pref = random.choice(cities)
            user_harga_pref = random.choice(harga_options)
            user_cuaca_pref = random.choice(cuaca_options)
            
            cursor.execute('''
                INSERT INTO user_preferences (username, lokasi, harga, cuaca, kategori)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, user_city_pref, user_harga_pref, user_cuaca_pref, user_category_pref))
            
            # Generate ratings based on user preferences
            suitable_places = self.df[
                (self.df['Category'] == user_category_pref) | 
                (self.df['City'] == user_city_pref)
            ]
            
            # If not enough suitable places, add random places
            if len(suitable_places) < n_ratings_per_user:
                remaining_places = self.df[~self.df['Place_Name'].isin(suitable_places['Place_Name'])]
                additional_places = remaining_places.sample(
                    min(n_ratings_per_user - len(suitable_places), len(remaining_places))
                )
                suitable_places = pd.concat([suitable_places, additional_places])
            
            # Sample places to rate
            places_to_rate = suitable_places.sample(min(n_ratings_per_user, len(suitable_places)))
            
            for _, place in places_to_rate.iterrows():
                # Generate rating based on preference match
                base_rating = place['Rating']
                
                # Adjust rating based on preferences
                if place['Category'] == user_category_pref:
                    preference_boost = random.uniform(0.5, 1.5)
                elif place['City'] == user_city_pref:
                    preference_boost = random.uniform(0.2, 0.8)
                else:
                    preference_boost = random.uniform(-0.5, 0.5)
                
                # Add some randomness
                noise = random.uniform(-0.3, 0.3)
                predicted_rating = max(1.0, min(5.0, base_rating + preference_boost + noise))
                
                # Insert rating
                cursor.execute('''
                    INSERT INTO user_ratings (username, place_name, rating)
                    VALUES (?, ?, ?)
                ''', (username, place['Place_Name'], predicted_rating))
                
                user_rating_data.append({
                    'username': username,
                    'place_name': place['Place_Name'],
                    'actual_rating': predicted_rating,
                    'category_pref': user_category_pref,
                    'city_pref': user_city_pref,
                    'harga_pref': user_harga_pref
                })
        
        conn.commit()
        conn.close()
        
        return pd.DataFrame(user_rating_data)
    
    def get_content_based_predicted_rating(self, user_prefs, place_name):
        """Predict rating using content-based approach"""
        place_data = self.df[self.df['Place_Name'] == place_name]
        if place_data.empty:
            return 3.0  # Default rating
        
        place = place_data.iloc[0]
        base_rating = place['Rating']
        
        # Adjust based on category preference
        category_match = 1.0 if place['Category'] == user_prefs.get('kategori') else 0.5
        city_match = 1.0 if place['City'] == user_prefs.get('lokasi') else 0.7
        
        # Price preference adjustment
        price_match = 1.0
        if user_prefs.get('harga'):
            if user_prefs['harga'] == "Murah (<20000)" and place['Price'] < 20000:
                price_match = 1.2
            elif user_prefs['harga'] == "Sedang (20000-50000)" and 20000 <= place['Price'] <= 50000:
                price_match = 1.1
            elif user_prefs['harga'] == "Mahal (>50000)" and place['Price'] > 50000:
                price_match = 1.1
            else:
                price_match = 0.8
        
        # Calculate weighted rating
        preference_weight = (category_match * 0.4 + city_match * 0.3 + price_match * 0.3)
        predicted_rating = base_rating * preference_weight
        
        return max(1.0, min(5.0, predicted_rating))
    
    def get_collaborative_predicted_rating(self, username, place_name):
        """Predict rating using collaborative filtering approach"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get similar users based on shared preferences
            cursor.execute('''
                SELECT DISTINCT u1.username
                FROM user_preferences u1
                JOIN user_preferences u2 ON (u1.kategori = u2.kategori OR u1.lokasi = u2.lokasi)
                WHERE u2.username = ? AND u1.username != ?
                LIMIT 10
            ''', (username, username))
            
            similar_users_result = cursor.fetchall()
            similar_users = [row[0] for row in similar_users_result]
            
            if not similar_users:
                conn.close()
                return 3.0  # Default rating if no similar users
            
            # Get ratings from similar users for this place
            placeholders = ','.join(['?' for _ in similar_users])
            query = f'''
                SELECT rating FROM user_ratings 
                WHERE username IN ({placeholders}) AND place_name = ?
            '''
            cursor.execute(query, similar_users + [place_name])
            
            similar_ratings_result = cursor.fetchall()
            similar_ratings = [row[0] for row in similar_ratings_result]
            
            if similar_ratings:
                predicted_rating = np.mean(similar_ratings)
            else:
                # Fallback: get average rating from similar users for any place
                cursor.execute(f'''
                    SELECT AVG(rating) FROM user_ratings 
                    WHERE username IN ({placeholders})
                ''', similar_users)
                
                avg_result = cursor.fetchone()
                predicted_rating = avg_result[0] if avg_result[0] is not None else 3.0
            
            conn.close()
            return max(1.0, min(5.0, predicted_rating))
            
        except Exception as e:
            print(f"Error in collaborative filtering for {username}, {place_name}: {e}")
            return 3.0
    
    def get_hybrid_predicted_rating(self, username, place_name, user_prefs):
        """Predict rating using hybrid approach (weighted combination)"""
        try:
            # Get predictions from both approaches
            content_pred = self.get_content_based_predicted_rating(user_prefs, place_name)
            collab_pred = self.get_collaborative_predicted_rating(username, place_name)
            
            # Weighted combination (70% content-based, 30% collaborative)
            hybrid_pred = 0.7 * content_pred + 0.3 * collab_pred
            
            return max(1.0, min(5.0, hybrid_pred))
        except Exception as e:
            print(f"Error in hybrid prediction for {username}, {place_name}: {e}")
            return 3.0
    
    def evaluate_mse(self, test_data, approach='content_based'):
        """Evaluate MSE for different approaches"""
        actual_ratings = []
        predicted_ratings = []
        
        print(f"Evaluating {len(test_data)} test samples...")
        
        for idx, row in test_data.iterrows():
            username = row['username']
            place_name = row['place_name']
            actual_rating = row['actual_rating']
            
            user_prefs = {
                'kategori': row['category_pref'],
                'lokasi': row['city_pref'],
                'harga': row.get('harga_pref', None)
            }
            
            try:
                if approach == 'content_based':
                    predicted_rating = self.get_content_based_predicted_rating(user_prefs, place_name)
                elif approach == 'collaborative':
                    predicted_rating = self.get_collaborative_predicted_rating(username, place_name)
                elif approach == 'hybrid':
                    predicted_rating = self.get_hybrid_predicted_rating(username, place_name, user_prefs)
                else:
                    predicted_rating = 3.0  # Default
                
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
                
            except Exception as e:
                print(f"Error processing {username}, {place_name}: {e}")
                actual_ratings.append(actual_rating)
                predicted_ratings.append(3.0)  # Default prediction on error
        
        # Calculate metrics
        mse = mean_squared_error(actual_ratings, predicted_ratings)
        mae = mean_absolute_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'actual_ratings': actual_ratings,
            'predicted_ratings': predicted_ratings
        }
    
    def run_mse_evaluation(self, n_users=100, test_size=0.2):
        """Run complete MSE evaluation"""
        print("Creating synthetic user-rating data...")
        user_rating_data = self.create_user_rating_matrix(n_users)
        
        print(f"Created {len(user_rating_data)} user-rating pairs")
        
        # Split data into train and test
        try:
            train_data, test_data = train_test_split(
                user_rating_data, 
                test_size=test_size, 
                random_state=42,
                stratify=user_rating_data['username']
            )
        except ValueError:
            # If stratification fails, do simple split
            train_data, test_data = train_test_split(
                user_rating_data, 
                test_size=test_size, 
                random_state=42
            )
        
        print(f"Train set: {len(train_data)}, Test set: {len(test_data)}")
        
        # Evaluate different approaches
        approaches = ['content_based', 'collaborative', 'hybrid']
        results = {}
        
        for approach in approaches:
            print(f"\nEvaluating {approach} approach...")
            metrics = self.evaluate_mse(test_data, approach)
            results[approach] = metrics
            
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
        
        return results, test_data
    
    def analyze_prediction_distribution(self, results):
        """Analyze the distribution of predictions vs actual ratings"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, (approach, metrics) in enumerate(results.items()):
                ax = axes[i]
                ax.scatter(metrics['actual_ratings'], metrics['predicted_ratings'], alpha=0.6)
                ax.plot([1, 5], [1, 5], 'r--', lw=2)  # Perfect prediction line
                ax.set_xlabel('Actual Ratings')
                ax.set_ylabel('Predicted Ratings')
                ax.set_title(f'{approach.replace("_", " ").title()}\nMSE: {metrics["mse"]:.4f}')
                ax.set_xlim(1, 5)
                ax.set_ylim(1, 5)
            
            plt.tight_layout()
            plt.savefig('mse_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return fig
        except ImportError:
            print("Matplotlib not available for visualization")
            return None

def main():
    # Setup paths
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "data/destinasi-wisata-YKSM.csv")
    DB_PATH = os.path.join(BASE_DIR, "data/mse_test_users.db")
    
    # Create evaluator
    evaluator = MSERecommendationEvaluator(DATA_PATH, DB_PATH)
    
    # Run MSE evaluation
    print("Starting MSE evaluation for recommendation system...")
    results, test_data = evaluator.run_mse_evaluation(n_users=100, test_size=0.2)
    
    # Print detailed results
    print("\n" + "="*70)
    print("MSE EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTest Setup:")
    print(f"- Number of test users: {test_data['username'].nunique()}")
    print(f"- Number of test ratings: {len(test_data)}")
    print(f"- Rating range: {test_data['actual_rating'].min():.2f} - {test_data['actual_rating'].max():.2f}")
    print(f"- Average actual rating: {test_data['actual_rating'].mean():.2f}")
    
    # Compare approaches
    approaches = ['content_based', 'collaborative', 'hybrid']
    
    for approach in approaches:
        metrics = results[approach]
        print(f"\n{approach.replace('_', ' ').title()} Approach:")
        print(f"- MSE (Mean Squared Error): {metrics['mse']:.4f}")
        print(f"- RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        print(f"- MAE (Mean Absolute Error): {metrics['mae']:.4f}")
        print(f"- Average predicted rating: {np.mean(metrics['predicted_ratings']):.2f}")
    
    # Find best approach
    best_approach = min(results.keys(), key=lambda x: results[x]['mse'])
    worst_approach = max(results.keys(), key=lambda x: results[x]['mse'])
    
    print(f"\n🏆 Best Approach: {best_approach.replace('_', ' ').title()}")
    print(f"   MSE: {results[best_approach]['mse']:.4f}")
    
    print(f"\n❌ Worst Approach: {worst_approach.replace('_', ' ').title()}")
    print(f"   MSE: {results[worst_approach]['mse']:.4f}")
    
    # Calculate improvement
    improvement = ((results[worst_approach]['mse'] - results[best_approach]['mse']) / 
                   results[worst_approach]['mse'] * 100)
    print(f"\n📈 Improvement: {improvement:.2f}% better MSE")
    
    # Interpretation
    print(f"\n📊 Interpretation:")
    best_rmse = results[best_approach]['rmse']
    if best_rmse < 0.5:
        interpretation = "Excellent prediction accuracy"
    elif best_rmse < 1.0:
        interpretation = "Good prediction accuracy"
    elif best_rmse < 1.5:
        interpretation = "Moderate prediction accuracy"
    else:
        interpretation = "Poor prediction accuracy - needs improvement"
    
    print(f"- Best RMSE of {best_rmse:.4f} indicates: {interpretation}")
    print(f"- On average, predictions are off by ±{best_rmse:.2f} rating points")
    
    print("\n" + "="*70)
    
    # Optionally create visualization
    try:
        evaluator.analyze_prediction_distribution(results)
        print("Prediction distribution plots saved as 'mse_analysis.png'")
    except ImportError:
        print("Matplotlib not available - skipping visualization")

if __name__ == "__main__":
    main()