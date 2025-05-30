import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class BookReviewsEDA:
    def __init__(self, goodreads_path, amazon_path):
        """Initialize with paths to datasets"""
        self.goodreads_path = goodreads_path
        self.amazon_path = amazon_path
        self.goodreads_df = None
        self.amazon_df = None
        self.merged_df = None
        
    def load_datasets(self):
        """Load both datasets"""
        try:
            self.goodreads_df = pd.read_csv(self.goodreads_path)
            print(f"Goodreads dataset: {self.goodreads_df.shape}")
            
            self.amazon_df = pd.read_csv(self.amazon_path)
            print(f"Amazon dataset: {self.amazon_df.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False
    
    def basic_statistics(self):
        """Generate basic statistics for both datasets"""   
        datasets = [("Goodreads", self.goodreads_df), ("Amazon", self.amazon_df)]
        
        for name, df in datasets:
            print(f"\n{name} Dataset:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Basic stats for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\n  Numeric columns summary:")
                print(df[numeric_cols].describe())
            
            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"\n  Missing values:")
                for col, count in missing[missing > 0].items():
                    print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    
    def rating_analysis(self):
        """Analyze rating distributions and patterns"""
        print("\n" + "*"*60)
        print("RATING ANALYSIS")
        print("*"*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Goodreads rating 
        if 'average_rating' in self.goodreads_df.columns:
            axes[0,0].hist(self.goodreads_df['average_rating'].dropna(), bins=50, alpha=0.7, color='skyblue')
            axes[0,0].set_title('Goodreads Rating Distribution')
            axes[0,0].set_xlabel('Average Rating')
            axes[0,0].set_ylabel('Frequency')
            
            gr_mean = self.goodreads_df['average_rating'].mean()
            gr_median = self.goodreads_df['average_rating'].median()
            axes[0,0].axvline(gr_mean, color='red', linestyle='--', label=f'Mean: {gr_mean:.2f}')
            axes[0,0].axvline(gr_median, color='green', linestyle='--', label=f'Median: {gr_median:.2f}')
            axes[0,0].legend()
            
            print(f"Goodreads Ratings:")
            print(f"  Mean: {gr_mean:.3f}")
            print(f"  Median: {gr_median:.3f}")
            print(f"  Std: {self.goodreads_df['average_rating'].std():.3f}")
        
        # Amazon rating 
        if 'average_rating' in self.amazon_df.columns:
            amazon_ratings = self.amazon_df['average_rating'].dropna()
            # If there are remove zero ratings 
            amazon_ratings = amazon_ratings[amazon_ratings > 0]  
            
            if len(amazon_ratings) > 0:
                axes[0,1].hist(amazon_ratings, bins=30, alpha=0.7, color='lightcoral')
                axes[0,1].set_title('Amazon Rating Distribution')
                axes[0,1].set_xlabel('Average Rating')
                axes[0,1].set_ylabel('Frequency')
                
                am_mean = amazon_ratings.mean()
                am_median = amazon_ratings.median()
                axes[0,1].axvline(am_mean, color='red', linestyle='--', label=f'Mean: {am_mean:.2f}')
                axes[0,1].axvline(am_median, color='green', linestyle='--', label=f'Median: {am_median:.2f}')
                axes[0,1].legend()
                
                print(f"\nAmazon Ratings:")
                print(f"  Mean: {am_mean:.3f}")
                print(f"  Median: {am_median:.3f}")
                print(f"  Std: {amazon_ratings.std():.3f}")
        
        # Ratings count comparison
        if 'ratings_count' in self.goodreads_df.columns:
            # Log scale for visualization
            gr_ratings_count = self.goodreads_df['ratings_count'].dropna()
            gr_ratings_count = gr_ratings_count[gr_ratings_count > 0]
            
            axes[1,0].hist(np.log10(gr_ratings_count), bins=50, alpha=0.7, color='lightgreen')
            axes[1,0].set_title('Goodreads Ratings Count (Log Scale)')
            axes[1,0].set_xlabel('Log10(Ratings Count)')
            axes[1,0].set_ylabel('Frequency')
            
            print(f"\nGoodreads Ratings Count:")
            print(f"  Mean: {gr_ratings_count.mean():.0f}")
            print(f"  Median: {gr_ratings_count.median():.0f}")
        
        if 'ratings_count' in self.amazon_df.columns:
            am_ratings_count = self.amazon_df['ratings_count'].dropna()
            am_ratings_count = am_ratings_count[am_ratings_count > 0]
            
            if len(am_ratings_count) > 0:
                axes[1,1].hist(np.log10(am_ratings_count), bins=30, alpha=0.7, color='orange')
                axes[1,1].set_title('Amazon Ratings Count (Log Scale)')
                axes[1,1].set_xlabel('Log10(Ratings Count)')
                axes[1,1].set_ylabel('Frequency')
                
                print(f"\nAmazon Ratings Count:")
                print(f"  Mean: {am_ratings_count.mean():.0f}")
                print(f"  Median: {am_ratings_count.median():.0f}")
        
        plt.tight_layout()
        plt.show()
    
    def genre_analysis(self):
        """Analyze genre distributions and popularity"""
        print("\n" + "*"*60)
        print("GENRE ANALYSIS")
        print("*"*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Goodreads genre 
        if 'primary_genre' in self.goodreads_df.columns:
            gr_genres = self.goodreads_df['primary_genre'].value_counts().head(10)
            print("Top 10 Goodreads Genres:")
            for genre, count in gr_genres.items():
                print(f"  {genre}: {count:,} books")
            
            axes[0,0].barh(range(len(gr_genres)), gr_genres.values)
            axes[0,0].set_yticks(range(len(gr_genres)))
            axes[0,0].set_yticklabels(gr_genres.index)
            axes[0,0].set_title('Top 10 Goodreads Genres')
            axes[0,0].set_xlabel('Number of Books')
        
        # Amazon genre 
        if 'primary_genre' in self.amazon_df.columns:
            am_genres = self.amazon_df['primary_genre'].value_counts().head(10)
            print(f"\nTop 10 Amazon Genres:")
            for genre, count in am_genres.items():
                print(f"  {genre}: {count:,} books")
            
            axes[0,1].barh(range(len(am_genres)), am_genres.values)
            axes[0,1].set_yticks(range(len(am_genres)))
            axes[0,1].set_yticklabels(am_genres.index)
            axes[0,1].set_title('Top 10 Amazon Genres')
            axes[0,1].set_xlabel('Number of Books')
        
        # Average rating by genre in Goodreads
        if 'primary_genre' in self.goodreads_df.columns and 'average_rating' in self.goodreads_df.columns:
            genre_ratings = self.goodreads_df.groupby('primary_genre')['average_rating'].agg(['mean', 'count']).reset_index()
            genre_ratings = genre_ratings[genre_ratings['count'] >= 50].sort_values('mean', ascending=False).head(10)
            
            axes[1,0].bar(range(len(genre_ratings)), genre_ratings['mean'])
            axes[1,0].set_xticks(range(len(genre_ratings)))
            axes[1,0].set_xticklabels(genre_ratings['primary_genre'], rotation=45, ha='right')
            axes[1,0].set_title('Average Rating by Genre (Goodreads)')
            axes[1,0].set_ylabel('Average Rating')
            
            print(f"\nBest Rated Genres (Goodreads, min 50 books):")
            for _, row in genre_ratings.iterrows():
                print(f"  {row['primary_genre']}: {row['mean']:.3f} ({row['count']} books)")
        
        # Popularity by genre in Goodreads
        if 'primary_genre' in self.goodreads_df.columns and 'ratings_count' in self.goodreads_df.columns:
            genre_popularity = self.goodreads_df.groupby('primary_genre')['ratings_count'].agg(['mean', 'count']).reset_index()
            genre_popularity = genre_popularity[genre_popularity['count'] >= 50].sort_values('mean', ascending=False).head(10)
            
            axes[1,1].bar(range(len(genre_popularity)), genre_popularity['mean'])
            axes[1,1].set_xticks(range(len(genre_popularity)))
            axes[1,1].set_xticklabels(genre_popularity['primary_genre'], rotation=45, ha='right')
            axes[1,1].set_title('Average Popularity by Genre (Goodreads)')
            axes[1,1].set_ylabel('Average Ratings Count')
            
            print(f"\nMost Popular Genres (Goodreads, min 50 books):")
            for _, row in genre_popularity.iterrows():
                print(f"  {row['primary_genre']}: {row['mean']:.0f} avg ratings ({row['count']} books)")
        
        plt.tight_layout()
        plt.show()
    
    def series_analysis(self):
        """Analyze series vs standalone books"""
        print("\n" + "*"*60)
        print("SERIES VS STANDALONE ANALYSIS")
        print("*"*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        datasets = [("Goodreads", self.goodreads_df), ("Amazon", self.amazon_df)]
        
        for idx, (name, df) in enumerate(datasets):
            if 'is_series' in df.columns:
                series_counts = df['is_series'].value_counts()
                series_pct = df['is_series'].mean() * 100
                
                print(f"\n{name} Series Statistics:")
                print(f"  Books in series: {series_counts.get(True, 0):,} ({series_pct:.1f}%)")
                print(f"  Standalone books: {series_counts.get(False, 0):,} ({100-series_pct:.1f}%)")
                
                # Pie chart
                labels = ['Standalone', 'Series']
                sizes = [series_counts.get(False, 0), series_counts.get(True, 0)]
                axes[0, idx].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[0, idx].set_title(f'{name}: Series vs Standalone')
                
                # Compare ratings
                if 'average_rating' in df.columns:
                    series_ratings = df[df['is_series'] == True]['average_rating'].dropna()
                    standalone_ratings = df[df['is_series'] == False]['average_rating'].dropna()
                    
                    if len(series_ratings) > 0 and len(standalone_ratings) > 0:
                        axes[1, idx].boxplot([standalone_ratings, series_ratings], 
                                           labels=['Standalone', 'Series'])
                        axes[1, idx].set_title(f'{name}: Rating Distribution')
                        axes[1, idx].set_ylabel('Average Rating')
                        
                        print(f"  Average rating - Series: {series_ratings.mean():.3f}")
                        print(f"  Average rating - Standalone: {standalone_ratings.mean():.3f}")
        
        plt.tight_layout()
        plt.show()
    
    def length_analysis(self):
        """Analyze book length and its impact on ratings"""
        print("\n" + "*"*60)
        print("BOOK LENGTH ANALYSIS")
        print("*"*60)
        
        if 'length_category' in self.goodreads_df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Length distribution
            length_counts = self.goodreads_df['length_category'].value_counts()
            print("Goodreads Book Length Distribution:")
            for category, count in length_counts.items():
                print(f"  {category}: {count:,} books")
            
            axes[0,0].bar(length_counts.index, length_counts.values)
            axes[0,0].set_title('Book Length Distribution (Goodreads)')
            axes[0,0].set_xlabel('Length Category')
            axes[0,0].set_ylabel('Number of Books')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Length + Rating
            if 'average_rating' in self.goodreads_df.columns:
                length_rating = self.goodreads_df.groupby('length_category')['average_rating'].agg(['mean', 'count']).reset_index()
                length_rating = length_rating[length_rating['count'] >= 10]  # Min 10 books
                
                axes[0,1].bar(length_rating['length_category'], length_rating['mean'])
                axes[0,1].set_title('Average Rating by Length Category')
                axes[0,1].set_xlabel('Length Category')
                axes[0,1].set_ylabel('Average Rating')
                axes[0,1].tick_params(axis='x', rotation=45)
                
                print(f"\nAverage Rating by Length:")
                for _, row in length_rating.iterrows():
                    print(f"  {row['length_category']}: {row['mean']:.3f} ({row['count']} books)")
            
            # Length + Popularity
            if 'ratings_count' in self.goodreads_df.columns:
                length_popularity = self.goodreads_df.groupby('length_category')['ratings_count'].agg(['mean', 'count']).reset_index()
                length_popularity = length_popularity[length_popularity['count'] >= 10]
                
                axes[1,0].bar(length_popularity['length_category'], length_popularity['mean'])
                axes[1,0].set_title('Average Popularity by Length Category')
                axes[1,0].set_xlabel('Length Category')
                axes[1,0].set_ylabel('Average Ratings Count')
                axes[1,0].tick_params(axis='x', rotation=45)
                
                print(f"\nAverage Popularity by Length:")
                for _, row in length_popularity.iterrows():
                    print(f"  {row['length_category']}: {row['mean']:.0f} ratings ({row['count']} books)")
            
            # If accessible page count distribution
            if 'num_pages' in self.goodreads_df.columns:
                pages = self.goodreads_df['num_pages'].dropna()
                 # Filter out unrealistic values
                pages = pages[(pages > 0) & (pages < 2000)] 
                
                axes[1,1].hist(pages, bins=50, alpha=0.7)
                axes[1,1].set_title('Page Count Distribution')
                axes[1,1].set_xlabel('Number of Pages')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].axvline(pages.mean(), color='red', linestyle='--', 
                                label=f'Mean: {pages.mean():.0f}')
                axes[1,1].legend()
            
            plt.tight_layout()
            plt.show()
    
    def popularity_analysis(self):
        """Analyze popularity"""
        print("\n" + "*"*60)
        print("POPULARITY ANALYSIS")
        print("*"*60)
        
        # Popularity categories
        if 'popularity_category' in self.goodreads_df.columns:
            pop_counts = self.goodreads_df['popularity_category'].value_counts()
            print("Goodreads Popularity Distribution:")
            for category, count in pop_counts.items():
                print(f"  {category}: {count:,} books")
        
        # Ratings and Popularity relation
        if 'average_rating' in self.goodreads_df.columns and 'ratings_count' in self.goodreads_df.columns:
            # Calculate relation
            valid_data = self.goodreads_df[['average_rating', 'ratings_count']].dropna()
            relation = valid_data['average_rating'].corr(np.log10(valid_data['ratings_count']))
            
            print(f"\nRating-Popularity Relation: {relation:.3f}")
        
        # Popular series analysis
        if 'is_series' in self.goodreads_df.columns and 'ratings_count' in self.goodreads_df.columns:
            series_pop = self.goodreads_df[self.goodreads_df['is_series'] == True]['ratings_count']
            standalone_pop = self.goodreads_df[self.goodreads_df['is_series'] == False]['ratings_count']
            
            if len(series_pop) > 0 and len(standalone_pop) > 0:
                print(f"\nMedian popularity - Series: {series_pop.median():.0f}")
                print(f"Median popularity - Standalone: {standalone_pop.median():.0f}")
        
        # Top rated books
        if 'title' in self.goodreads_df.columns and 'average_rating' in self.goodreads_df.columns:
            # Books with at least 1000 ratings are taken
            popular_books = self.goodreads_df[self.goodreads_df['ratings_count'] >= 1000]
            top_rated = popular_books.nlargest(10, 'average_rating')[['title', 'average_rating', 'ratings_count']]
            
            print(f"\nTop 10 Highest Rated Popular Books (≥1000 ratings):")
            for _, book in top_rated.iterrows():
                title = book['title'][:40] + "..." if len(book['title']) > 40 else book['title']
                print(f"  {title}: {book['average_rating']:.3f} ({book['ratings_count']:,} ratings)")
    
    def relation_analysis(self):
        """Analyze relations between different variables"""
        print("\n" + "*"*60)
        print("RELATION ANALYSIS")
        print("*"*60)
        
        # Select numeric columns 
        numeric_cols = ['average_rating', 'ratings_count']
        if 'num_pages' in self.goodreads_df.columns:
            numeric_cols.append('num_pages')
        if 'review_engagement' in self.goodreads_df.columns:
            numeric_cols.append('review_engagement')
        # Filter columns 
        available_cols = [col for col in numeric_cols if col in self.goodreads_df.columns]
        
        if len(available_cols) >= 2:
            relation_data = self.goodreads_df[available_cols].dropna()
            
            # Create relation matrix
            corr_matrix = relation_data.corr()
            
            print("Relation Matrix:")
            print(corr_matrix.round(3))
            
            # Identify strongest ones
            print(f"\nStrongest Relations:")
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.1:  
                        print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    def comparative_analysis(self):
        """Compare Goodreads and Amazon data"""
        print("\n" + "*"*60)
        print("GOODREADS - AMAZON COMPARISON")
        print("*"*60)
        
        # Compare average ratings
        if 'average_rating' in self.goodreads_df.columns and 'average_rating' in self.amazon_df.columns:
            gr_ratings = self.goodreads_df['average_rating'].dropna()
            am_ratings = self.amazon_df['average_rating'].dropna()
            am_ratings = am_ratings[am_ratings > 0]  # Remove zero ratings
            
            if len(am_ratings) > 0:
                print(f"Rating Comparison:")
                print(f"  Goodreads - Mean: {gr_ratings.mean():.3f}, Std: {gr_ratings.std():.3f}")
                print(f"  Amazon - Mean: {am_ratings.mean():.3f}, Std: {am_ratings.std():.3f}")
        
        # Compare genre distributions
        if 'primary_genre' in self.goodreads_df.columns and 'primary_genre' in self.amazon_df.columns:
            gr_genres = self.goodreads_df['primary_genre'].value_counts().head(5)
            am_genres = self.amazon_df['primary_genre'].value_counts().head(5)
            
            print(f"\nTop 5 Genres Comparison:")
            print("Goodreads | Amazon")
            print("*" * 60)
            for i in range(5):
                gr_genre = gr_genres.index[i] if i < len(gr_genres) else "N/A"
                am_genre = am_genres.index[i] if i < len(am_genres) else "N/A"
                gr_count = gr_genres.iloc[i] if i < len(gr_genres) else 0
                am_count = am_genres.iloc[i] if i < len(am_genres) else 0
                print(f"{gr_genre[:15]:<15} | {am_genre[:15]}")
                print(f"({gr_count:,}){'':>8} | ({am_count:,})")
    
    def run_complete_analysis(self):
        print("Starting Comprehensive Exploratory Data Analysis")
        print("=" * 60)
        
        if not self.load_datasets():
            return False
        
        # Run all analysis components
        self.basic_statistics()
        self.rating_analysis()
        self.genre_analysis()
        self.series_analysis()
        self.length_analysis()
        self.popularity_analysis()
        self.relation_analysis()
        self.comparative_analysis()
        
        print("\n" + "*"*60)
        print("EXPLORATORY DATA ANALYSIS COMPLETE!")
        print("*"*60)
        
        return True

# Usage example:
if __name__ == "__main__":
    goodreads_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/goodreads_dataset/edited_books.csv"
    amazon_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/amazon_book_reviews_dataset/edited_amazon_books.csv"
    
    # Create EDA instance and run analysis
    eda = BookReviewsEDA(goodreads_path, amazon_path)
    eda.run_complete_analysis()