import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Constants
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data/destinasi-wisata-YKSM.csv")
TFIDF_PATH = os.path.join(BASE_DIR, "data/content_based_tfidf.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "data/content_based_matrix.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data_and_models():
    """Load dataset and TF-IDF models"""
    print("Loading dataset and models...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure numeric columns
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    
    # Try to load existing TF-IDF models
    try:
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
        print("TF-IDF models loaded successfully.")
    except (FileNotFoundError, EOFError, pickle.PickleError):
        print("Building new TF-IDF model...")
        # Create combined text field for TF-IDF
        df["combined_text"] = (
            df["Place_Name"] + " " + 
            df["Category"].fillna("") + " " + 
            df["Description"].fillna("")
        )
        
        # Create and fit TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        tfidf_matrix = tfidf.fit_transform(df["combined_text"])
        print("TF-IDF model built.")
        
        # Save the models
        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
    
    return df, tfidf, tfidf_matrix

def compute_ground_truth(df, category_mapping=None):
    """
    Compute ground truth by mapping destinations to relevant categories.
    
    Args:
        df: DataFrame with destinations
        category_mapping: Dictionary mapping categories to groups (optional)
    
    Returns:
        Dictionary mapping destination IDs to relevant category groups
    """
    ground_truth = {}
    
    if category_mapping is None:
        # Default mapping based on category similarity
        category_mapping = {
            'Alam': ['Alam', 'Pantai', 'Gunung', 'Taman', 'Danau'],
            'Budaya': ['Museum', 'Sejarah', 'Budaya', 'Candi'],
            'Kuliner': ['Kuliner', 'Restoran', 'Kafe', 'Pasar'],
            'Rekreasi': ['Rekreasi', 'Hiburan', 'Taman', 'Kolam'],
            'Spiritual': ['Masjid', 'Gereja', 'Vihara', 'Pura', 'Klenteng']
        }
    
    for idx, row in df.iterrows():
        dest_id = row['Place_Name']
        category = row['Category']
        relevant_groups = []
        
        # Find which group(s) this destination belongs to
        for group, categories in category_mapping.items():
            if any(cat in str(category) for cat in categories):
                relevant_groups.append(group)
        
        if not relevant_groups:
            relevant_groups = ['Lainnya']
            
        ground_truth[dest_id] = relevant_groups
    
    return ground_truth

def generate_query_by_category(category_group, df):
    """Generate a search query based on a category group"""
    
    category_query_terms = {
        'Alam': 'wisata alam pemandangan indah pegunungan pantai air terjun',
        'Budaya': 'museum sejarah budaya heritage tradisional',
        'Kuliner': 'kuliner makanan khas restoran enak',
        'Rekreasi': 'wahana hiburan rekreasi keluarga seru',
        'Spiritual': 'tempat ibadah spiritual religius'
    }
    
    return category_query_terms.get(category_group, "wisata " + category_group.lower())

def get_recommendations(df, tfidf, tfidf_matrix, query, top_n=20):
    """Get content-based recommendations using the current system"""
    
    # Transform query with TF-IDF vectorizer
    query_vector = tfidf.transform([query])
    
    # Calculate cosine similarity
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Create result dataframe with scores
    temp_df = df.copy()
    temp_df['similarity_score'] = sim_scores
    
    # Sort by similarity score and get top N
    return temp_df.sort_values('similarity_score', ascending=False).head(top_n)

def calculate_recall_at_k(recommendations, ground_truth, target_category, k_values):
    """
    Calculate Recall@k for different k values
    
    Args:
        recommendations: DataFrame with recommended destinations
        ground_truth: Dictionary mapping destinations to relevant categories
        target_category: Category we're testing for
        k_values: List of k values to calculate recall for
    
    Returns:
        Dictionary with k values as keys and recall values as values
    """
    recalls = {}
    
    # Get all destinations that belong to target category
    relevant_destinations = [dest for dest, cats in ground_truth.items() 
                             if target_category in cats]
    total_relevant = len(relevant_destinations)
    
    if total_relevant == 0:
        print(f"Warning: No relevant destinations found for category {target_category}")
        return {k: 0 for k in k_values}
    
    # Calculate recall for each k
    for k in k_values:
        # Get top k recommendations
        top_k_recs = recommendations.head(k)['Place_Name'].tolist()
        
        # Count how many of these recommendations are relevant
        retrieved_relevant = sum(1 for dest in top_k_recs 
                                if dest in relevant_destinations)
        
        # Calculate recall
        recall = retrieved_relevant / total_relevant
        recalls[k] = recall
    
    return recalls

def evaluate_recall_by_category(df, tfidf, tfidf_matrix):
    """Evaluate recall across different categories"""
    
    # Define category groups to test
    categories = ['Alam', 'Budaya', 'Kuliner', 'Rekreasi', 'Spiritual']
    
    # Compute ground truth
    ground_truth = compute_ground_truth(df)
    
    # K values to evaluate
    k_values = [5, 10, 15, 20]
    
    # Store results
    results = {cat: {} for cat in categories}
    
    for category in categories:
        print(f"Evaluating recall for category: {category}")
        
        # Generate query for this category
        query = generate_query_by_category(category, df)
        print(f"  Query: '{query}'")
        
        # Get recommendations
        recommendations = get_recommendations(df, tfidf, tfidf_matrix, query, top_n=max(k_values))
        
        # Calculate recall at different k values
        recalls = calculate_recall_at_k(recommendations, ground_truth, category, k_values)
        results[category] = recalls
        
        # Print results for this category
        for k, recall in recalls.items():
            print(f"  Recall@{k}: {recall:.4f}")
        print()
    
    return results, k_values

def plot_recall_results(results, k_values):
    """Create plots for recall results"""
    
    plt.figure(figsize=(12, 8))
    
    for category, recalls in results.items():
        recall_values = [recalls[k] for k in k_values]
        plt.plot(k_values, recall_values, marker='o', label=category)
    
    plt.title('Recall@k by Category')
    plt.xlabel('k (Number of recommendations)')
    plt.ylabel('Recall')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"recall_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    # Create bar chart for Recall@10
    plt.figure(figsize=(10, 6))
    categories = list(results.keys())
    recall_at_10 = [results[cat][10] for cat in categories]
    
    bars = plt.bar(categories, recall_at_10, color='skyblue')
    plt.title('Recall@10 by Category')
    plt.xlabel('Category')
    plt.ylabel('Recall')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Save bar chart
    bar_path = os.path.join(RESULTS_DIR, f"recall_at_10_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Bar chart saved to {bar_path}")

def save_results_to_csv(results, k_values):
    """Save recall results to CSV file"""
    
    # Convert results to DataFrame
    data = []
    for category, recalls in results.items():
        row = {'Category': category}
        for k in k_values:
            row[f'Recall@{k}'] = recalls[k]
        data.append(row)
    
    results_df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, f"recall_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Calculate average recall across categories
    avg_row = {'Category': 'Average'}
    for k in k_values:
        avg_row[f'Recall@{k}'] = results_df[f'Recall@{k}'].mean()
    
    avg_df = pd.DataFrame([avg_row])
    results_df = pd.concat([results_df, avg_df], ignore_index=True)
    
    # Print summary
    print("\nSummary of Results:")
    print(results_df)
    
    return results_df

def main():
    """Main evaluation function"""
    start_time = time.time()
    
    # Load data and models
    df, tfidf, tfidf_matrix = load_data_and_models()
    
    # Evaluate recall by category
    results, k_values = evaluate_recall_by_category(df, tfidf, tfidf_matrix)
    
    # Plot results
    plot_recall_results(results, k_values)
    
    # Save results to CSV
    summary_df = save_results_to_csv(results, k_values)
    
    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")
    
    # Calculate overall average recall
    for k in k_values:
        avg_recall = summary_df[f'Recall@{k}'].iloc[-1]  # Last row contains the average
        print(f"Average Recall@{k} across all categories: {avg_recall:.4f}")

if __name__ == "__main__":
    main()