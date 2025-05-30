import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, r2_score, classification_report, 
                           confusion_matrix, silhouette_score, adjusted_rand_score)
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class BookReviewsML:
    def __init__(self, goodreads_path, amazon_path):
        """Initialize"""
        self.goodreads_path = goodreads_path
        self.amazon_path = amazon_path
        self.goodreads_df = None
        self.amazon_df = None
        self.combined_df = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load bothg datasets"""
        try:
            self.goodreads_df = pd.read_csv(self.goodreads_path)
            self.amazon_df = pd.read_csv(self.amazon_path)
            
            # Add source identifier
            self.goodreads_df['source'] = 'goodreads'
            self.amazon_df['source'] = 'amazon'
            
            # Combine datasets for more comprehensive analysis
            common_columns = list(set(self.goodreads_df.columns) & set(self.amazon_df.columns))
            
            gr_subset = self.goodreads_df[common_columns].copy()
            am_subset = self.amazon_df[common_columns].copy()
            
            self.combined_df = pd.concat([gr_subset, am_subset], ignore_index=True)
            
            print(f"Goodreads dataset: {self.goodreads_df.shape}")
            print(f"Amazon dataset: {self.amazon_df.shape}")
            print(f"Combined dataset: {self.combined_df.shape}")
            print(f"Common columns: {len(common_columns)}")
            
            return True
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False
    
    def prepare_features(self, df, target_column, feature_columns=None):
        """Prepare features for ML models"""
        if feature_columns is None:
            # Automatically select relevant features
            feature_columns = []
            
            # Numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_column and not col.endswith('_id'):
                    feature_columns.append(col)
            
            # Categorical features that might be useful
            categorical_cols = ['primary_genre', 'length_category', 'popularity_category', 
                              'is_series', 'language_code', 'source']
            for col in categorical_cols:
                if col in df.columns:
                    feature_columns.append(col)
        
        # Create feature matrix
        feature_df = df[feature_columns + [target_column]].copy()
        
        # Handle missing values
        for col in feature_df.columns:
            if feature_df[col].dtype in ['object', 'category']:
                feature_df[col] = feature_df[col].fillna('Unknown')
            else:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Encode categorical variables
        categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns
        categorical_features = [col for col in categorical_features if col != target_column]
        
        encoded_features = []
        
        for col in categorical_features:
            if feature_df[col].nunique() <= 10:  # One-hot encode if few categories
                dummies = pd.get_dummies(feature_df[col], prefix=col)
                encoded_features.append(dummies)
            else:  # Label encode if many categories
                le = LabelEncoder()
                feature_df[f'{col}_encoded'] = le.fit_transform(feature_df[col])
        
        # Combine all features
        if encoded_features:
            encoded_df = pd.concat(encoded_features, axis=1)
            numeric_features = feature_df.select_dtypes(include=[np.number])
            final_features = pd.concat([numeric_features, encoded_df], axis=1)
        else:
            final_features = feature_df.select_dtypes(include=[np.number])
        
        # Remove target from features
        if target_column in final_features.columns:
            y = final_features[target_column]
            X = final_features.drop(columns=[target_column])
        else:
            y = feature_df[target_column]
            X = final_features
        
        return X, y
    
    def rating_prediction_analysis(self):
        """Predict book ratings using various ML algorithms"""
        print("\n" + "*"*60)
        print("RATING PREDICTION")
        print("*"*60)
        
        # Use combined dataset for more data
        df = self.combined_df.copy()
        
        # Filter valid ratings
        df = df[df['average_rating'].notna() & (df['average_rating'] > 0)]
        
        if len(df) < 100:
            print("Insufficient data for rating prediction.")
            return
        
        print(f"Dataset size for rating prediction: {len(df):,} books")
        
        # Prepare features
        X, y = self.prepare_features(df, 'average_rating')
        
        if X.empty:
            print("No suitable features found for prediction.")
            return
        
        print(f"Features used: {X.columns.tolist()}")
        print(f"Feature matrix shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) 
        # Models to test
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        results = {}
        
        print(f"\nTraining and evaluating models...")
        
        for name, model in models.items():
            print(f"\n{name}:")
            
            # Use scaled data for SVR, original for tree-based models
            if name == 'SVR':
                X_train_model, X_test_model = X_train_scaled, X_test_scaled
            else:
                X_train_model, X_test_model = X_train, X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            # Predictions
            y_pred = model.predict(X_test_model)
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'RMSE': rmse,
                'R2': r2,
                'CV_R2_mean': cv_scores.mean(),
                'CV_R2_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in results:
            rf_model = models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features :")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Visualization
        self.plot_regression_results(y_test, results)
        
        return results
    
    def popularity_classification_analysis(self):
        """Classify books into popularity categories"""
        print("\n" + "*"*60)
        print("POPULARITY CLASSIFICATION")
        print("*"*60)
        
        df = self.goodreads_df.copy()
        
        # Create popularity categories if not exist
        if 'popularity_category' not in df.columns:
            if 'ratings_count' in df.columns:
                df = df[df['ratings_count'].notna() & (df['ratings_count'] > 0)]
                bins = [0, 100, 1000, 10000, 100000, float('inf')]
                labels = ['Unknown', 'Niche', 'Modest', 'Popular', 'Bestseller']
                df['popularity_category'] = pd.cut(df['ratings_count'], bins=bins, labels=labels)
            else:
                print("Cannot create popularity categories: missing ratings_count column")
                return
        
        # Filter out 'Unknown' category and ensure sufficient samples
        df = df[df['popularity_category'] != 'Unknown']
        category_counts = df['popularity_category'].value_counts()
        
        # Keep only categories with at least 50 samples
        valid_categories = category_counts[category_counts >= 50].index
        df = df[df['popularity_category'].isin(valid_categories)]
        
        if len(df) < 200:
            print("Insufficient data for popularity classification.")
            return
        
        print(f"Dataset size: {len(df):,} books")
        print(f"Categories: {df['popularity_category'].value_counts().to_dict()}")
        
        # Prepare features
        X, y = self.prepare_features(df, 'popularity_category')

        if X.empty:
            print("No suitable features found for classification.")
            return
        
        print(f"Features used: {X.columns.tolist()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        # Models to test
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVC': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        print(f"\nTraining classification models...")
        
        for name, model in models.items():
            print(f"\n{name}:")
            
            # Use scaled data for LR and SVC, original for RF
            if name in ['Logistic Regression', 'SVC']:
                X_train_model, X_test_model = X_train_scaled, X_test_scaled
            else:
                X_train_model, X_test_model = X_train, X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            # Predictions
            y_pred = model.predict(X_test_model)
            # Cross-validation accuracy
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5)
            
            results[name] = {
                'accuracy': (y_pred == y_test).mean(),
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'predictions': y_pred,
                'model': model
            }
            
            print(f"  Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Detailed results for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_accuracy_mean'])
        best_predictions = results[best_model_name]['predictions']
        
        print(f"\nDetailed Results for Best Model ({best_model_name}):")
        print(classification_report(y_test, best_predictions))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, best_predictions, best_model_name)
        
        return results
    
    def clustering_analysis(self):
        """Perform clustering analysis to identify book groups"""
        print("\n" + "*"*60)
        print("CLUSTERING ANALYSIS")
        print("*"*60)
        
        df = self.goodreads_df.copy()
        
        # Select numeric features for clustering
        numeric_features = ['average_rating', 'ratings_count']
        if 'num_pages' in df.columns:
            numeric_features.append('num_pages')
        if 'review_engagement' in df.columns:
            numeric_features.append('review_engagement')
        
        # Filter to existing columns
        available_features = [col for col in numeric_features if col in df.columns]
        
        if len(available_features) < 2:
            print("Insufficient numeric features for clustering.")
            return
        
        # Prepare data
        cluster_data = df[available_features].dropna()
        
        # Remove outliers (using IQR method)
        for col in cluster_data.columns:
            Q1 = cluster_data[col].quantile(0.25)
            Q3 = cluster_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cluster_data = cluster_data[(cluster_data[col] >= lower_bound) & 
                                      (cluster_data[col] <= upper_bound)]

        if len(cluster_data) < 100:
            print("Insufficient data after outlier removal.")
            return
        
        print(f"Clustering dataset: {cluster_data.shape}")
        print(f"Features: {available_features}")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.4f}")
        
        # Perform final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans_final.fit_predict(scaled_data)
        
        # Add cluster labels to original data
        cluster_data['cluster'] = cluster_labels
        # Analyze clusters
        print(f"\nCluster Analysis:")
        for i in range(optimal_k):
            cluster_books = cluster_data[cluster_data['cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_books):,} books):")
            for feature in available_features:
                mean_val = cluster_books[feature].mean()
                print(f"  {feature}: {mean_val:.3f}")
        
        # PCA for visualization
        if len(available_features) > 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"\nPCA explained variance: {explained_variance:.3f}")
        else:
            pca_data = scaled_data
            explained_variance = 1.0
        
        # Visualization
        self.plot_clustering_results(cluster_data, pca_data, cluster_labels, 
                                   optimal_k, available_features, k_range, 
                                   inertias, silhouette_scores)
        
        return cluster_data, cluster_labels
    
    def genre_prediction_analysis(self):
        """Predict book genres based on other features"""
        print("\n" + "="*60)
        print("GENRE PREDICTION")
        print("="*60)
        
        df = self.goodreads_df.copy()
        
        if 'primary_genre' not in df.columns:
            print("Genre information not available.")
            return
        
        # Filter to most common genres
        genre_counts = df['primary_genre'].value_counts()
        top_genres = genre_counts[genre_counts >= 100].head(8).index.tolist()
        
        df = df[df['primary_genre'].isin(top_genres)]
        
        if len(df) < 200:
            print("Insufficient data for genre prediction.")
            return
        
        print(f"Dataset size: {len(df):,} books")
        print(f"Genres: {df['primary_genre'].value_counts().to_dict()}")
        
        # Prepare features (exclude genre-related columns)
        feature_columns = ['average_rating', 'ratings_count', 'num_pages', 
                          'is_series', 'length_category', 'popularity_category']
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X, y = self.prepare_features(df, 'primary_genre', feature_columns)
        
        if X.empty:
            print("No suitable features found for genre prediction.")
            return
        
        print(f"Features used: {X.columns.tolist()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        # Predictions
        y_pred = rf_model.predict(X_test)
        # Accuracy
        accuracy = (y_pred == y_test).mean()
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
        
        print(f"\nGenre Prediction Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance for Genre Prediction:")
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return rf_model, feature_importance
    
    def plot_regression_results(self, y_test, results):
        """Plot regression model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        models = list(results.keys())
        rmse_scores = [results[model]['RMSE'] for model in models]
        r2_scores = [results[model]['R2'] for model in models]
        
        axes[0,0].bar(models, rmse_scores)
        axes[0,0].set_title('RMSE Comparison')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        axes[0,1].bar(models, r2_scores)
        axes[0,1].set_title('R² Score Comparison')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Actual vs Predicted for best model
        best_model = max(results.keys(), key=lambda k: results[k]['R2'])
        best_predictions = results[best_model]['predictions']
        
        axes[1,0].scatter(y_test, best_predictions, alpha=0.6)
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Rating')
        axes[1,0].set_ylabel('Predicted Rating')
        axes[1,0].set_title(f'Actual vs Predicted ({best_model})')
        
        # Residuals plot
        residuals = y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Predicted Rating')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title(f'Residuals Plot ({best_model})')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_clustering_results(self, cluster_data, pca_data, cluster_labels, 
                              optimal_k, features, k_range, inertias, silhouette_scores):
        """Plot clustering analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Elbow curve
        axes[0,0].plot(k_range, inertias, 'bo-')
        axes[0,0].set_xlabel('Number of Clusters')
        axes[0,0].set_ylabel('Inertia')
        axes[0,0].set_title('Elbow Method for Optimal k')
        axes[0,0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[0,0].legend()
        
        # Silhouette scores
        axes[0,1].plot(k_range, silhouette_scores, 'ro-')
        axes[0,1].set_xlabel('Number of Clusters')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].set_title('Silhouette Score vs Number of Clusters')
        axes[0,1].axvline(x=optimal_k, color='b', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[0,1].legend()
        
        # Cluster visualization (PCA space)
        scatter = axes[1,0].scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis')
        axes[1,0].set_xlabel('First Principal Component')
        axes[1,0].set_ylabel('Second Principal Component')
        axes[1,0].set_title('Clusters in PCA Space')
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Cluster characteristics
        if len(features) >= 2:
            for i in range(optimal_k):
                cluster_books = cluster_data[cluster_data['cluster'] == i]
                axes[1,1].scatter(cluster_books[features[0]], cluster_books[features[1]], 
                                label=f'Cluster {i}', alpha=0.6)
            axes[1,1].set_xlabel(features[0])
            axes[1,1].set_ylabel(features[1])
            axes[1,1].set_title('Clusters in Original Feature Space')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_ml_analysis(self):
        print("MACHINE LEARNING ANALYSIS FOR BOOK REVIEWS")
        print("*" * 60)
        
        if not self.load_and_prepare_data():
            return False
        
        # Run all ML analyses
        print("\n1. RATING PREDICTION ANALYSIS")
        rating_results = self.rating_prediction_analysis()
        
        print("\n2. POPULARITY CLASSIFICATION ANALYSIS")
        popularity_results = self.popularity_classification_analysis()
        
        print("\n3. CLUSTERING ANALYSIS")
        clustering_results = self.clustering_analysis()
        
        print("\n4. GENRE PREDICTION ANALYSIS")
        genre_results = self.genre_prediction_analysis()
        
        print("\n" + "*"*60)
        print("MACHINE LEARNING ANALYSIS COMPLETE!")
        print("*"*60)
        
        # Results summary
        print("\nSUMMARY OF RESULTS:")
        if rating_results:
            best_rating_model = max(rating_results.keys(), key=lambda k: rating_results[k]['R2'])
            print(f"Best Rating Prediction: {best_rating_model} (R² = {rating_results[best_rating_model]['R2']:.4f})")
        
        if popularity_results:
            best_pop_model = max(popularity_results.keys(), key=lambda k: popularity_results[k]['cv_accuracy_mean'])
            print(f"Best Popularity Classification: {best_pop_model} (Accuracy = {popularity_results[best_pop_model]['cv_accuracy_mean']:.4f})")
        
        return {
            'rating_prediction': rating_results,
            'popularity_classification': popularity_results,
            'clustering': clustering_results,
            'genre_prediction': genre_results
        }

# Usage example:
if __name__ == "__main__":
    goodreads_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/goodreads_dataset/edited_books.csv"
    amazon_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/amazon_book_reviews_dataset/edited_amazon_books.csv"
    
    # Create ML analysis instance and run complete analysis
    ml_analyzer = BookReviewsML(goodreads_path, amazon_path)
    results = ml_analyzer.run_complete_ml_analysis()