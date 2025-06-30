import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data and models similar to app.py
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data/destinasi-wisata-YKSM.csv")
TFIDF_PATH = os.path.join(BASE_DIR, "data/content_based_tfidf.pkl")
MATRIX_PATH = os.path.join(BASE_DIR, "data/content_based_matrix.pkl")

# Load data
df = pd.read_csv(DATA_PATH)

# Create combined text field regardless of whether we load or create a new model
df["combined_text"] = df["Place_Name"] + " " + df["Category"].fillna("") + " " + df["Description"].fillna("")

# Load TF-IDF model
try:
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    print("TF-IDF model loaded successfully.")
except (FileNotFoundError, EOFError):
    print("TF-IDF model not found. Creating new one.")
    # Create and fit the TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["combined_text"])
    
    # Save the models
    try:
        os.makedirs(os.path.dirname(TFIDF_PATH), exist_ok=True)
        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(tfidf, f)
        with open(MATRIX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
        print("TF-IDF model saved successfully.")
    except Exception as e:
        print(f"Error saving TF-IDF model: {e}")

# Improved transformation function for better RMSE
def transform_similarity_to_rating(similarity_score, base_rating=3.0):
    """
    Improved transformation with adaptive scaling
    """
    if similarity_score < 0.01:
        return max(1.0, base_rating - 1.5)
    elif similarity_score < 0.03:
        return max(1.5, base_rating - 1.0)
    elif similarity_score < 0.06:
        return max(2.0, base_rating - 0.5)
    elif similarity_score < 0.10:
        return base_rating
    elif similarity_score < 0.15:
        return min(4.0, base_rating + 0.5)
    elif similarity_score < 0.25:
        return min(4.5, base_rating + 1.0)
    else:
        return 5.0

# Split data into training and testing sets
# Convert rating to numeric and filter out rows with missing ratings
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
rated_df = df.dropna(subset=["Rating", "combined_text"])

# Use stratified sampling to ensure we have ratings across the full range
train_df, test_df = train_test_split(
    rated_df, 
    test_size=0.2,  # Use 20% of data for testing
    random_state=42,
    stratify=pd.qcut(rated_df["Rating"], q=5, duplicates="drop", labels=False)
)

# Train TF-IDF model on training data only with advanced parameters
train_tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),  # Include bigrams for better semantic capture
    max_features=5000,   # Limit features to reduce noise
    min_df=2,            # Ignore terms that appear in less than 2 documents
    max_df=0.85,         # Ignore terms that appear in more than 85% of documents
    sublinear_tf=True    # Apply sublinear scaling to term frequencies
)
train_tfidf_matrix = train_tfidf.fit_transform(train_df["combined_text"])

# Enhanced prediction function with category and price context
def predict_rating(destination_text, train_df, train_tfidf, train_tfidf_matrix):
    """
    Predict rating for a destination based on its text description
    """
    # Transform query text
    query_vector = train_tfidf.transform([destination_text])
    
    # Calculate cosine similarity
    sim_scores = cosine_similarity(query_vector, train_tfidf_matrix).flatten()
    
    # Add similarity scores to dataframe
    temp_df = train_df.copy()
    temp_df['similarity_score'] = sim_scores
    
    # Get top 10 most similar items (increased from 5 for better prediction)
    temp_df = temp_df.sort_values('similarity_score', ascending=False).head(10)
    
    # Apply sigmoid transformation to emphasize higher similarities
    temp_df['boosted_similarity'] = 1 / (1 + np.exp(-12 * (temp_df['similarity_score'] - 0.2)))
    
    # Transform similarity scores to ratings
    temp_df['predicted_rating'] = temp_df['similarity_score'].apply(transform_similarity_to_rating)
    
    # Take the weighted average of ratings with sigmoid-boosted weights
    if len(temp_df) > 0:
        return np.average(temp_df['predicted_rating'], weights=temp_df['boosted_similarity'])
    else:
        return 3.0  # Default prediction if no similar items found

# Calculate predictions for test set
print("Calculating predictions for test set...")
actual_ratings = []
predicted_ratings = []

for idx, row in test_df.iterrows():
    # Get actual rating
    actual_rating = row["Rating"]
    
    # Get text description for prediction
    description = row["combined_text"]
    
    # Predict rating
    predicted_rating = predict_rating(description, train_df, train_tfidf, train_tfidf_matrix)
    
    # Store results
    actual_ratings.append(actual_rating)
    predicted_ratings.append(predicted_rating)
    
    # Print progress every 20 items
    if len(actual_ratings) % 20 == 0:
        print(f"Processed {len(actual_ratings)}/{len(test_df)} test items")

# Calculate MAE
mae = mean_absolute_error(actual_ratings, predicted_ratings)
print(f"\nMean Absolute Error (MAE): {mae:.4f}")

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((np.array(actual_ratings) - np.array(predicted_ratings))**2))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Additional analysis
print("\nPerformance breakdown by rating range:")
test_results = pd.DataFrame({
    'actual': actual_ratings,
    'predicted': predicted_ratings,
    'abs_error': np.abs(np.array(actual_ratings) - np.array(predicted_ratings))
})

# Group by rating ranges and calculate average error
rating_ranges = [1, 2, 3, 4, 5, 6]
for i in range(len(rating_ranges)-1):
    range_df = test_results[(test_results['actual'] >= rating_ranges[i]) & 
                            (test_results['actual'] < rating_ranges[i+1])]
    if len(range_df) > 0:
        print(f"Rating {rating_ranges[i]}-{rating_ranges[i+1]}: MAE = {range_df['abs_error'].mean():.4f} (n={len(range_df)})")

# Plot distribution of errors
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(test_results['abs_error'], bins=20, alpha=0.7)
    plt.title('Distribution of Absolute Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.axvline(mae, color='r', linestyle='dashed', linewidth=2, label=f'MAE = {mae:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(os.path.join(BASE_DIR, "mae_distribution.png"))
    print(f"\nError distribution plot saved as 'mae_distribution.png'")
except ImportError:
    print("\nMatplotlib not installed. Skipping error distribution plot.")

# Save evaluation results
test_results.to_csv(os.path.join(BASE_DIR, "recommendation_evaluation.csv"), index=False)
print("Detailed evaluation results saved to 'recommendation_evaluation.csv'")