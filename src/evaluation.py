import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import streamlit as st
from app import (
    load_content_based,
    get_content_based_recommendations_by_text,
    get_hybrid_recommendations,
    DATA_PATH
)

class RecommendationEvaluator:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the evaluator with configuration parameters.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.df, self.tfidf, self.tfidf_matrix = load_content_based()
        
        # Load additional data if needed for evaluation
        self.user_queries = [
            "pantai", "gunung", "museum", "budaya", "pemandangan", 
            "sejarah", "kuliner", "alam", "air terjun", "hiking"
        ]
        self.locations = ["Yogyakarta", "Semarang", "Bantul", "Sleman"]
        self.cuaca_options = ["Cerah", "Hujan Ringan", "Berawan"]
        self.harga_options = ["Murah (<20000)", "Sedang (20000-50000)", "Mahal (>50000)"]
        
        # Prepare data splits
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data splits for evaluation"""
        # Split data into train and test sets
        self.train_df, self.test_df = train_test_split(
            self.df,
            test_size=self.test_size, 
            random_state=self.random_state
        )
        print(f"Training set: {len(self.train_df)} items")
        print(f"Test set: {len(self.test_df)} items")
    
    def evaluate_content_based(self):
        """
        Evaluate content-based recommendation model.
        
        Returns:
            dict: Evaluation metrics
        """
        results = {
            'relevance_scores': [],
            'coverage': 0,
            'avg_similarity_score': 0,
            'query_performance': {},
            'location_performance': {}
        }
        
        all_recommendations = set()
        total_similarity = 0
        count = 0
        
        # Test with different queries
        for query in self.user_queries:
            query_results = []
            
            # Test with different locations
            for location in self.locations:
                # Get recommendations
                recs = get_content_based_recommendations_by_text(
                    self.df, self.tfidf, self.tfidf_matrix,
                    query, location, None, None, top_n=20
                )
                
                if not recs.empty and 'similarity_score' in recs.columns:
                    # Track coverage
                    all_recommendations.update(recs['Place_Name'].tolist())
                    
                    # Track average similarity score
                    avg_sim = recs['similarity_score'].mean()
                    total_similarity += avg_sim
                    count += 1
                    
                    # Track location performance
                    if location not in results['location_performance']:
                        results['location_performance'][location] = []
                    results['location_performance'][location].append(avg_sim)
                    
                    # Track query performance
                    if query not in results['query_performance']:
                        results['query_performance'][query] = []
                    results['query_performance'][query].append(avg_sim)
                    
                    query_results.append(avg_sim)
            
            if query_results:
                results['relevance_scores'].append(np.mean(query_results))
            
        # Calculate overall metrics
        if count > 0:
            results['avg_similarity_score'] = total_similarity / count
        results['coverage'] = len(all_recommendations) / len(self.df) * 100
        
        # Average by query and location
        for query in results['query_performance']:
            results['query_performance'][query] = np.mean(results['query_performance'][query])
        
        for location in results['location_performance']:
            results['location_performance'][location] = np.mean(results['location_performance'][location])
        
        return results
    
    def evaluate_hybrid(self, username="test_user"):
        """
        Evaluate hybrid recommendation model.
        
        Args:
            username: Username to use for evaluation
        
        Returns:
            dict: Evaluation metrics
        """
        results = {
            'relevance_scores': [],
            'coverage': 0,
            'avg_similarity_score': 0,
            'query_performance': {},
            'location_performance': {}
        }
        
        all_recommendations = set()
        total_similarity = 0
        count = 0
        
        # Test with different queries
        for query in self.user_queries:
            query_results = []
            
            # Test with different locations
            for location in self.locations:
                # Get recommendations
                recs = get_hybrid_recommendations(
                    self.df, self.tfidf, self.tfidf_matrix,
                    username, location, "Cerah", query, None
                )
                
                if not recs.empty and 'similarity_score' in recs.columns:
                    # Track coverage
                    all_recommendations.update(recs['Place_Name'].tolist())
                    
                    # Track average similarity score
                    avg_sim = recs['similarity_score'].mean()
                    total_similarity += avg_sim
                    count += 1
                    
                    # Track location performance
                    if location not in results['location_performance']:
                        results['location_performance'][location] = []
                    results['location_performance'][location].append(avg_sim)
                    
                    # Track query performance
                    if query not in results['query_performance']:
                        results['query_performance'][query] = []
                    results['query_performance'][query].append(avg_sim)
                    
                    query_results.append(avg_sim)
            
            if query_results:
                results['relevance_scores'].append(np.mean(query_results))
            
        # Calculate overall metrics
        if count > 0:
            results['avg_similarity_score'] = total_similarity / count
        results['coverage'] = len(all_recommendations) / len(self.df) * 100
        
        # Average by query and location
        for query in results['query_performance']:
            results['query_performance'][query] = np.mean(results['query_performance'][query])
        
        for location in results['location_performance']:
            results['location_performance'][location] = np.mean(results['location_performance'][location])
        
        return results
    
    def compare_models(self):
        """Compare content-based and hybrid models"""
        print("Evaluating content-based recommendations...")
        content_results = self.evaluate_content_based()
        
        print("Evaluating hybrid recommendations...")
        hybrid_results = self.evaluate_hybrid()
        
        print("\n=== Content-Based Recommendation Results ===")
        print(f"Average Similarity Score: {content_results['avg_similarity_score']:.4f}")
        print(f"Catalog Coverage: {content_results['coverage']:.2f}%")
        
        print("\n=== Hybrid Recommendation Results ===")
        print(f"Average Similarity Score: {hybrid_results['avg_similarity_score']:.4f}")
        print(f"Catalog Coverage: {hybrid_results['coverage']:.2f}%")
        
        # Return both results for further analysis or visualization
        return content_results, hybrid_results
    
    def plot_results(self, content_results, hybrid_results):
        """
        Create visualizations to compare model performance.
        
        Args:
            content_results: Results from content-based evaluation
            hybrid_results: Results from hybrid evaluation
        """
        plt.figure(figsize=(15, 12))
        
        # 1. Overall comparison
        plt.subplot(2, 2, 1)
        metrics = ['avg_similarity_score', 'coverage']
        content_values = [content_results['avg_similarity_score'], content_results['coverage']/100]
        hybrid_values = [hybrid_results['avg_similarity_score'], hybrid_results['coverage']/100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, content_values, width, label='Content-Based')
        plt.bar(x + width/2, hybrid_values, width, label='Hybrid')
        plt.ylabel('Score')
        plt.title('Overall Model Comparison')
        plt.xticks(x, ['Similarity Score', 'Coverage (normalized)'])
        plt.legend()
        
        # 2. Query performance comparison
        plt.subplot(2, 2, 2)
        queries = list(content_results['query_performance'].keys())
        content_query_vals = [content_results['query_performance'][q] for q in queries]
        hybrid_query_vals = [hybrid_results['query_performance'][q] for q in queries]
        
        x = np.arange(len(queries))
        plt.bar(x - width/2, content_query_vals, width, label='Content-Based')
        plt.bar(x + width/2, hybrid_query_vals, width, label='Hybrid')
        plt.ylabel('Average Similarity Score')
        plt.title('Query Performance Comparison')
        plt.xticks(x, queries, rotation=45, ha='right')
        plt.legend()
        
        # 3. Location performance comparison
        plt.subplot(2, 2, 3)
        locations = list(content_results['location_performance'].keys())
        content_loc_vals = [content_results['location_performance'][loc] for loc in locations]
        hybrid_loc_vals = [hybrid_results['location_performance'][loc] for loc in locations]
        
        x = np.arange(len(locations))
        plt.bar(x - width/2, content_loc_vals, width, label='Content-Based')
        plt.bar(x + width/2, hybrid_loc_vals, width, label='Hybrid')
        plt.ylabel('Average Similarity Score')
        plt.title('Location Performance Comparison')
        plt.xticks(x, locations, rotation=45, ha='right')
        plt.legend()
        
        # 4. Relevance scores comparison
        plt.subplot(2, 2, 4)
        plt.boxplot([content_results['relevance_scores'], hybrid_results['relevance_scores']], 
                   labels=['Content-Based', 'Hybrid'])
        plt.ylabel('Relevance Score Distribution')
        plt.title('Recommendation Relevance Comparison')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()
        
        return 'model_comparison.png'

def evaluate_recommendation_diversity(df_cb, tfidf, tfidf_matrix, num_samples=50):
    """
    Evaluate recommendation diversity based on content similarity between recommendations.
    
    Args:
        df_cb: DataFrame with destination data
        tfidf: TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of destinations
        num_samples: Number of query samples to use
    
    Returns:
        dict: Diversity metrics
    """
    # Define sample queries and locations
    sample_queries = [
        "pantai", "gunung", "museum", "budaya", "pemandangan", 
        "sejarah", "kuliner", "alam", "air terjun", "hiking"
    ]
    sample_locations = ["Yogyakarta", "Semarang", ""]
    
    results = {
        'content_based_diversity': [],
        'hybrid_diversity': []
    }
    
    # Create temporary test user data
    test_username = "diversity_test_user"
    
    for query in sample_queries[:min(5, len(sample_queries))]:
        for location in sample_locations[:min(2, len(sample_locations))]:
            # Get content-based recommendations
            content_recs = get_content_based_recommendations_by_text(
                df_cb, tfidf, tfidf_matrix, query, location, None, None, top_n=10
            )
            
            if not content_recs.empty:
                # Calculate intralist diversity (average pairwise dissimilarity)
                if len(content_recs) > 1:
                    # Use Place_Name for identifying unique places
                    content_places = content_recs['Place_Name'].unique()[:10]
                    if len(content_places) > 1:
                        # Get indices of these places in the original matrix
                        indices = [df_cb.index[df_cb['Place_Name'] == place][0] for place in content_places 
                                  if not df_cb.index[df_cb['Place_Name'] == place].empty]
                        
                        if len(indices) > 1:
                            # Extract TF-IDF vectors for these places
                            vectors = tfidf_matrix[indices]
                            
                            # Calculate pairwise similarities
                            similarities = cosine_similarity(vectors)
                            
                            # Calculate diversity (1 - average similarity excluding self-similarity)
                            n = similarities.shape[0]
                            diversity = 1 - (np.sum(similarities) - n) / (n * (n-1))
                            results['content_based_diversity'].append(diversity)
            
            # Get hybrid recommendations
            hybrid_recs = get_hybrid_recommendations(
                df_cb, tfidf, tfidf_matrix, test_username, 
                location, "Cerah", query, None
            )
            
            if not hybrid_recs.empty:
                # Calculate intralist diversity (average pairwise dissimilarity)
                if len(hybrid_recs) > 1:
                    # Use Place_Name for identifying unique places
                    hybrid_places = hybrid_recs['Place_Name'].unique()[:10]
                    if len(hybrid_places) > 1:
                        # Get indices of these places in the original matrix
                        indices = [df_cb.index[df_cb['Place_Name'] == place][0] for place in hybrid_places
                                  if not df_cb.index[df_cb['Place_Name'] == place].empty]
                        
                        if len(indices) > 1:
                            # Extract TF-IDF vectors for these places
                            vectors = tfidf_matrix[indices]
                            
                            # Calculate pairwise similarities
                            similarities = cosine_similarity(vectors)
                            
                            # Calculate diversity (1 - average similarity excluding self-similarity)
                            n = similarities.shape[0]
                            diversity = 1 - (np.sum(similarities) - n) / (n * (n-1))
                            results['hybrid_diversity'].append(diversity)
    
    # Calculate average diversity scores
    results['avg_content_diversity'] = np.mean(results['content_based_diversity']) if results['content_based_diversity'] else 0
    results['avg_hybrid_diversity'] = np.mean(results['hybrid_diversity']) if results['hybrid_diversity'] else 0
    
    return results

def run_evaluation():
    """Run the complete evaluation"""
    print("Starting recommendation model evaluation...")
    
    # Load models and data
    df_cb, tfidf, tfidf_matrix = load_content_based()
    
    print("\n1. Evaluating model performance...")
    evaluator = RecommendationEvaluator()
    content_results, hybrid_results = evaluator.compare_models()
    
    print("\n2. Evaluating recommendation diversity...")
    diversity_results = evaluate_recommendation_diversity(df_cb, tfidf, tfidf_matrix)
    print(f"Content-based diversity: {diversity_results['avg_content_diversity']:.4f}")
    print(f"Hybrid diversity: {diversity_results['avg_hybrid_diversity']:.4f}")
    
    print("\n3. Creating visualizations...")
    img_path = evaluator.plot_results(content_results, hybrid_results)
    
    print(f"\nEvaluation complete! Visualization saved to {img_path}")
    
    # Return combined results
    return {
        'content_based': content_results,
        'hybrid': hybrid_results,
        'diversity': diversity_results
    }

if __name__ == "__main__":
    results = run_evaluation()