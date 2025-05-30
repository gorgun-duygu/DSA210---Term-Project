import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, spearmanr, wilcoxon
import warnings
warnings.filterwarnings('ignore')

class BookReviewsHypothesisTesting:
    def __init__(self, goodreads_path, amazon_path):
        """Initialize the hypothesis testing class with dataset paths"""
        self.goodreads_path = goodreads_path
        self.amazon_path = amazon_path
        self.goodreads_df = None
        self.amazon_df = None
        self.alpha = 0.05  # Significance level
        
    def load_datasets(self):
        """Load both datasets"""
        try:
            self.goodreads_df = pd.read_csv(self.goodreads_path)
            self.amazon_df = pd.read_csv(self.amazon_path)
            print(f"Datasets loaded successfully")
            print(f"Goodreads: {self.goodreads_df.shape}")
            print(f"Amazon: {self.amazon_df.shape}")
            return True
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False
    
    def print_test_results(self, test_name, statistic, p_value, conclusion, effect_size=None):
        print(f"\n{test_name}")
        print("-" * len(test_name))
        print(f"Test Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.6f}")
        if effect_size is not None:
            print(f"Effect Size: {effect_size:.4f}")
        print(f"Significance Level: α = {self.alpha}")
        
        if p_value < self.alpha:
            print("Result: REJECT null hypothesis")
        else:
            print("Result: FAIL TO REJECT null hypothesis")
        
        print(f"Conclusion: {conclusion}")
        print("*" * 60)
    
    def calculate_effect_size(self, group1, group2):
        """Calculate Cohen's d effect size"""
        pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
        if pooled_std > 0:
            return abs(group1.mean() - group2.mean()) / pooled_std
        return 0
    
    def hypothesis1_series_vs_standalone_ratings(self):
        """H1: Series and standalone books have different average ratings"""
        print("\n" + "*"*60)
        print("HYPOTHESIS 1: SERIES VS STANDALONE BOOK RATINGS")
        print("*"*60)
        
        if 'is_series' not in self.goodreads_df.columns or 'average_rating' not in self.goodreads_df.columns:
            print("Required columns not found for this test.")
            return
        
        # Prepare data
        series_ratings = self.goodreads_df[self.goodreads_df['is_series'] == True]['average_rating'].dropna()
        standalone_ratings = self.goodreads_df[self.goodreads_df['is_series'] == False]['average_rating'].dropna()
        
        if len(series_ratings) == 0 or len(standalone_ratings) == 0:
            print("Insufficient data for this test. (0 ratings found)")
            return
        
        print(f"DATA SUMMARY:")
        print(f"Series books: {len(series_ratings):,} samples, mean = {series_ratings.mean():.3f}")
        print(f"Standalone books: {len(standalone_ratings):,} samples, mean = {standalone_ratings.mean():.3f}")
        print(f"Difference: {series_ratings.mean() - standalone_ratings.mean():.3f}")
        
        # Mann-Whitney U test 
        statistic, p_value = mannwhitneyu(series_ratings, standalone_ratings, alternative='two-sided')
        
        # Effect size
        effect_size = self.calculate_effect_size(series_ratings, standalone_ratings)
        
        # Conclusion
        if series_ratings.mean() > standalone_ratings.mean():
            direction = "higher"
        else:
            direction = "lower"
        
        conclusion = f"Series books have {direction} average ratings than standalone books."
        
        self.print_test_results("Mann-Whitney U Test: Series vs Standalone", 
                               statistic, p_value, conclusion, effect_size)
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.boxplot([standalone_ratings, series_ratings], labels=['Standalone', 'Series'])
        plt.title('Rating Distribution: Series vs Standalone')
        plt.ylabel('Average Rating')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(standalone_ratings, bins=30, alpha=0.7, label='Standalone', density=True)
        plt.hist(series_ratings, bins=30, alpha=0.7, label='Series', density=True)
        plt.title('Rating Density Distribution')
        plt.xlabel('Average Rating')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def hypothesis2_length_impact_on_ratings(self):
        """H2: At least one length category has different average ratings"""
        print("\n" + "*"*60)
        print("HYPOTHESIS 2: BOOK LENGTH IMPACT ON RATINGS")
        print("*"*60)
        
        if 'length_category' not in self.goodreads_df.columns or 'average_rating' not in self.goodreads_df.columns:
            print("Required columns not found for this test.")
            return
        
        # Prepare data - remove unknown categories
        data = self.goodreads_df[self.goodreads_df['length_category'] != 'Unknown'].copy()
        
        if len(data) == 0:
            print("No valid length category data found.")
            return
        
        # Group data by length category
        length_groups = []
        length_names = []
        
        for category in data['length_category'].unique():
            if pd.notna(category):
                group_data = data[data['length_category'] == category]['average_rating'].dropna()
                if len(group_data) >= 30:  # Minimum sample size
                    length_groups.append(group_data)
                    length_names.append(category)
        
        print(f"DATA SUMMARY:")
        print(f"Length categories analyzed: {len(length_groups)}")
        for i, name in enumerate(length_names):
            print(f"   {name}: {len(length_groups[i]):,} books, mean = {length_groups[i].mean():.3f}")
        
        # Kruskal-Wallis test (non-parametric ANOVA equivalent)
        statistic, p_value = kruskal(*length_groups)
        
        # Effect size (eta-squared approximation)
        n_total = sum(len(group) for group in length_groups)
        eta_squared = (statistic - len(length_groups) + 1) / (n_total - len(length_groups))
        
        conclusion = "Book length categories have significantly different average ratings." if p_value < self.alpha else "No significant difference in ratings across length categories."
        
        self.print_test_results("Kruskal-Wallis Test: Length vs Ratings", 
                               statistic, p_value, conclusion, eta_squared)
        
        # Post-hoc analysis if significant
        if p_value < self.alpha:
            #Post-Hoc Pairwise Comparison
            from itertools import combinations
            for i, j in combinations(range(len(length_groups)), 2):
                stat, p_val = mannwhitneyu(length_groups[i], length_groups[j], alternative='two-sided')
                significance = "✅" if p_val < 0.05 else "❌"
                print(f"   {length_names[i]} vs {length_names[j]}: p = {p_val:.4f} {significance}")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        box_data = [group.values for group in length_groups]
        plt.boxplot(box_data, labels=length_names)
        plt.title('Rating Distribution by Length Category')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        means = [group.mean() for group in length_groups]
        plt.bar(length_names, means, alpha=0.7)
        plt.title('Mean Rating by Length Category')
        plt.ylabel('Mean Rating')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def hypothesis3_genre_ratings_differences(self):
        """H3: At least one genre has different average ratings"""
        print("\n" + "*"*60)
        print("HYPOTHESIS 3: GENRE DIFFERENCES IN RATINGS")
        print("*"*60)
        
        if 'primary_genre' not in self.goodreads_df.columns or 'average_rating' not in self.goodreads_df.columns:
            print("Required columns not found for this test.")
            return
        
        # Get top genres with sufficient sample sizes
        genre_counts = self.goodreads_df['primary_genre'].value_counts()
        top_genres = genre_counts[genre_counts >= 100].head(10).index.tolist()
        
        # Prepare data
        genre_groups = []
        genre_names = []
        
        for genre in top_genres:
            if genre != 'Unknown':
                genre_data = self.goodreads_df[self.goodreads_df['primary_genre'] == genre]['average_rating'].dropna()
                if len(genre_data) >= 50:
                    genre_groups.append(genre_data)
                    genre_names.append(genre[:20])  # Truncate long names
        
        print(f"DATA SUMMARY:")
        print(f"Genres analyzed: {len(genre_groups)}")
        for i, name in enumerate(genre_names):
            print(f"   {name}: {len(genre_groups[i]):,} books, mean = {genre_groups[i].mean():.3f}")
        
        # Kruskal-Wallis test
        statistic, p_value = kruskal(*genre_groups)
        
        # Effect size
        n_total = sum(len(group) for group in genre_groups)
        eta_squared = (statistic - len(genre_groups) + 1) / (n_total - len(genre_groups))
        
        conclusion = "Genres have significantly different average ratings." if p_value < self.alpha else "No significant difference in ratings across genres."
        
        self.print_test_results("Kruskal-Wallis Test: Genre vs Ratings", 
                               statistic, p_value, conclusion, eta_squared)
        
        # Visualization
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        box_data = [group.values for group in genre_groups]
        plt.boxplot(box_data, labels=genre_names)
        plt.title('Rating Distribution by Genre')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        means = [group.mean() for group in genre_groups]
        stds = [group.std() for group in genre_groups]
        plt.bar(genre_names, means, yerr=stds, capsize=5, alpha=0.7)
        plt.title('Mean Rating by Genre (with Standard Deviation)')
        plt.ylabel('Mean Rating')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def hypothesis4_popularity_rating_relation(self):
        """H4: There is a significant correlation between popularity and average rating"""
        print("\n" + "*"*60)
        print("HYPOTHESIS 4: POPULARITY-RATING CORRELATION")
        print("*"*60)
        
        if 'ratings_count' not in self.goodreads_df.columns or 'average_rating' not in self.goodreads_df.columns:
            print("Required columns not found for this test.")
            return
        
        # Prepare data
        data = self.goodreads_df[['ratings_count', 'average_rating']].dropna()
        data = data[data['ratings_count'] > 0]  # Remove zero ratings

        popularity = data['ratings_count']
        ratings = data['average_rating']
        
        print(f"DATA SUMMARY:")
        print(f"Sample size: {len(data):,}")
        print(f"Popularity range: {popularity.min():,} to {popularity.max():,}")
        print(f"Rating range: {ratings.min():.3f} to {ratings.max():.3f}")
        
        # Spearman correlation (robust, works with any relationship shape)
        correlation, p_value = spearmanr(popularity, ratings)
        
        print(f"\nRELATION ANALYSIS:")
        print(f"Spearman correlation: ρ = {correlation:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        conclusion = f"There is a {strength} {direction} correlation between popularity and ratings." if p_value < self.alpha else "No significant correlation between popularity and ratings."
        
        self.print_test_results("Spearman Rank Correlation", 
                               correlation, p_value, conclusion)
        
        # Visualization
        plt.figure(figsize=(15, 5))
        
        # Sample for better visualization
        sample_size = min(5000, len(data))
        sample_data = data.sample(sample_size, random_state=42)
        
        plt.subplot(1, 3, 1)
        plt.scatter(sample_data['ratings_count'], sample_data['average_rating'], alpha=0.5, s=10)
        plt.xlabel('Ratings Count')
        plt.ylabel('Average Rating')
        plt.title('Popularity vs Rating (Linear Scale)')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.scatter(np.log10(sample_data['ratings_count']), sample_data['average_rating'], alpha=0.5, s=10)
        plt.xlabel('Log10(Ratings Count)')
        plt.ylabel('Average Rating')
        plt.title('Log Popularity vs Rating')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        # Popularity bins
        data['popularity_bin'] = pd.cut(np.log10(data['ratings_count']), bins=5, 
                                       labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        bin_means = data.groupby('popularity_bin')['average_rating'].mean()
        plt.bar(range(len(bin_means)), bin_means.values, alpha=0.7)
        plt.xticks(range(len(bin_means)), bin_means.index)
        plt.xlabel('Popularity Level')
        plt.ylabel('Mean Rating')
        plt.title('Mean Rating by Popularity Level')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def hypothesis5_goodreads_vs_amazon_ratings(self):
        """H5: Goodreads and Amazon ratings are different"""
        print("\n" + "*"*60)
        print("HYPOTHESIS 5: GOODREADS VS AMAZON RATINGS")
        print("*"*60)
        
        if 'title' not in self.goodreads_df.columns or 'title' not in self.amazon_df.columns:
            print("Title columns not found for matching.")
            return
        
        if 'average_rating' not in self.goodreads_df.columns or 'average_rating' not in self.amazon_df.columns:
            print("Rating columns not found.")
            return
        
        # Clean titles for matching
        def clean_title(title):
            if pd.isna(title):
                return ""
            return str(title).lower().strip().replace("'", "").replace('"', '').replace(" ", "")
        
        gr_clean = self.goodreads_df.copy()
        am_clean = self.amazon_df.copy()
        
        gr_clean['clean_title'] = gr_clean['title'].apply(clean_title)
        am_clean['clean_title'] = am_clean['title'].apply(clean_title)
        
        # Remove books with zero ratings from Amazon
        am_clean = am_clean[am_clean['average_rating'] > 0]
        
        # Find matching books
        matched = pd.merge(
            gr_clean[['clean_title', 'average_rating']].rename(columns={'average_rating': 'goodreads_rating'}),
            am_clean[['clean_title', 'average_rating']].rename(columns={'average_rating': 'amazon_rating'}),
            on='clean_title',
            how='inner'
        )
        
        print(f"DATA SUMMARY:")
        print(f"Matched books found: {len(matched):,}")
        
        if len(matched) < 30:
            print("Insufficient matched books for paired analysis. (Less than 30 matched books)")
            print("Performing independent samples comparison instead...")
            
            # Compare overall distributions
            gr_ratings = self.goodreads_df['average_rating'].dropna()
            am_ratings = self.amazon_df['average_rating'].dropna()
            am_ratings = am_ratings[am_ratings > 0]
            
            if len(am_ratings) == 0:
                print("No valid Amazon ratings found.")
                return
            
            print(f"Goodreads ratings: {len(gr_ratings):,} books, mean = {gr_ratings.mean():.3f}")
            print(f"Amazon ratings: {len(am_ratings):,} books, mean = {am_ratings.mean():.3f}")
            
            # Mann-Whitney U test (independent samples)
            statistic, p_value = mannwhitneyu(gr_ratings, am_ratings, alternative='two-sided')
            effect_size = self.calculate_effect_size(gr_ratings, am_ratings)
            test_name = "Mann-Whitney U Test (Independent Samples)"
            
        else:
            print(f"Goodreads mean: {matched['goodreads_rating'].mean():.3f}")
            print(f"Amazon mean: {matched['amazon_rating'].mean():.3f}")
            print(f"Mean difference: {matched['goodreads_rating'].mean() - matched['amazon_rating'].mean():.3f}")
            
            # Wilcoxon signed-rank test for paired data (robust version of paired t-test)
            statistic, p_value = wilcoxon(matched['goodreads_rating'], matched['amazon_rating'])
            
            # Effect size for paired data
            differences = matched['goodreads_rating'] - matched['amazon_rating']
            effect_size = differences.mean() / differences.std() if differences.std() > 0 else 0
            test_name = "Wilcoxon Signed-Rank Test (Matched Books)"
            
            # Visualization for matched books
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.scatter(matched['goodreads_rating'], matched['amazon_rating'], alpha=0.6)
            plt.plot([1, 5], [1, 5], 'r--', label='Perfect Agreement')
            plt.xlabel('Goodreads Rating')
            plt.ylabel('Amazon Rating')
            plt.title('Matched Books: Goodreads vs Amazon Ratings')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.hist(differences, bins=30, alpha=0.7)
            plt.axvline(0, color='red', linestyle='--', label='No Difference')
            plt.axvline(differences.mean(), color='green', linestyle='--', 
                       label=f'Mean Diff: {differences.mean():.3f}')
            plt.xlabel('Rating Difference (Goodreads - Amazon)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Rating Differences')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.boxplot([matched['goodreads_rating'], matched['amazon_rating']], 
                       labels=['Goodreads', 'Amazon'])
            plt.ylabel('Average Rating')
            plt.title('Rating Distribution Comparison')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            corr_coef = matched['goodreads_rating'].corr(matched['amazon_rating'])
            plt.text(0.5, 0.7, f'Correlation: {corr_coef:.3f}', 
                    transform=plt.gca().transAxes, fontsize=14, ha='center')
            plt.text(0.5, 0.5, f'Mean Difference: {differences.mean():.3f}', 
                    transform=plt.gca().transAxes, fontsize=14, ha='center')
            plt.text(0.5, 0.3, f'Sample Size: {len(matched):,}', 
                    transform=plt.gca().transAxes, fontsize=14, ha='center')
            plt.title('Summary Statistics')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        conclusion = "Goodreads and Amazon ratings are significantly different." if p_value < self.alpha else "No significant difference between Goodreads and Amazon ratings."
        
        self.print_test_results(test_name, statistic, p_value, conclusion, effect_size)
    
    def hypothesis6_length_popularity_relationship(self):
        """H6: There is a significant correlation between book length and popularity"""
        print("\n" + "*"*60)
        print("HYPOTHESIS 6: BOOK LENGTH VS POPULARITY CORRELATION")
        print("*"*60)
        
        if 'num_pages' not in self.goodreads_df.columns or 'ratings_count' not in self.goodreads_df.columns:
            print("Required columns not found for this test.")
            return
        
        # Prepare data
        data = self.goodreads_df[['num_pages', 'ratings_count']].dropna()
        data = data[(data['num_pages'] > 50) & (data['num_pages'] < 1500)]  # Filter realistic page counts
        data = data[data['ratings_count'] > 0]
        
        pages = data['num_pages']
        popularity = data['ratings_count']
        
        print(f"DATA SUMMARY:")
        print(f"Sample size: {len(data):,}")
        print(f"Page range: {pages.min():.0f} to {pages.max():.0f}")
        print(f"Popularity range: {popularity.min():,} to {popularity.max():,}")
        
        # Spearman correlation 
        correlation, p_value = spearmanr(pages, popularity)
        
        print(f"\nRELATION ANALYSIS:")
        print(f"Spearman correlation: ρ = {correlation:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        # Interpret relation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if correlation > 0 else "negative"
        
        conclusion = f"Book length has a {strength} {direction} relationship with popularity." if p_value < self.alpha else "No significant relationship between book length and popularity."
        
        self.print_test_results("Spearman Rank Correlation: Length vs Popularity", 
                               correlation, p_value, conclusion)
        
        # Visualization
        plt.figure(figsize=(12, 8))
        
        # Sample for visualization
        sample_size = min(3000, len(data))
        sample_data = data.sample(sample_size, random_state=42)
        
        plt.subplot(2, 2, 1)
        plt.scatter(sample_data['num_pages'], sample_data['ratings_count'], alpha=0.5, s=10)
        plt.xlabel('Number of Pages')
        plt.ylabel('Ratings Count')
        plt.title('Pages vs Popularity (Linear Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter(sample_data['num_pages'], np.log10(sample_data['ratings_count']), alpha=0.5, s=10)
        plt.xlabel('Number of Pages')
        plt.ylabel('Log10(Ratings Count)')
        plt.title('Pages vs Log-Popularity')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # Page bins
        data['page_bins'] = pd.cut(data['num_pages'], bins=5, 
                                  labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        bin_popularity = data.groupby('page_bins')['ratings_count'].median()
        plt.bar(range(len(bin_popularity)), bin_popularity.values, alpha=0.7)
        plt.xticks(range(len(bin_popularity)), bin_popularity.index, rotation=45)
        plt.xlabel('Length Category')
        plt.ylabel('Median Popularity')
        plt.title('Median Popularity by Length Category')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Box plot
        page_groups = [data[data['page_bins'] == cat]['ratings_count'].values for cat in bin_popularity.index]
        plt.boxplot(page_groups, labels=bin_popularity.index)
        plt.xlabel('Length Category')
        plt.ylabel('Ratings Count')
        plt.title('Popularity Distribution by Length')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_all_hypothesis_tests(self):
        """Run all hypothesis tests using simplified approach"""
        print("RUN ALL HYPOTHESIS TESTS")
        print("*" * 60)
        
        if not self.load_datasets():
            return False
        
        # Track results
        results = {}
        
        # Run all hypothesis tests
        print("\n📋 Running all 6 hypothesis tests...")
        
        try:
            self.hypothesis1_series_vs_standalone_ratings()
            results['hypothesis1'] = 'completed'
        except Exception as e:
            print(f"Hypothesis 1 failed: {e}")
            results['hypothesis1'] = 'failed'
        
        try:
            self.hypothesis2_length_impact_on_ratings()
            results['hypothesis2'] = 'completed'
        except Exception as e:
            print(f"Hypothesis 2 failed: {e}")
            results['hypothesis2'] = 'failed'
        
        try:
            self.hypothesis3_genre_ratings_differences()
            results['hypothesis3'] = 'completed'
        except Exception as e:
            print(f"Hypothesis 3 failed: {e}")
            results['hypothesis3'] = 'failed'
        
        try:
            self.hypothesis4_popularity_rating_relation()
            results['hypothesis4'] = 'completed'
        except Exception as e:
            print(f"Hypothesis 4 failed: {e}")
            results['hypothesis4'] = 'failed'
        
        try:
            self.hypothesis5_goodreads_vs_amazon_ratings()
            results['hypothesis5'] = 'completed'
        except Exception as e:
            print(f"Hypothesis 5 failed: {e}")
            results['hypothesis5'] = 'failed'
        
        try:
            self.hypothesis6_length_popularity_relationship()
            results['hypothesis6'] = 'completed'
        except Exception as e:
            print(f"Hypothesis 6 failed: {e}")
            results['hypothesis6'] = 'failed'
        
        # Results summary
        completed = sum(1 for v in results.values() if v == 'completed')
        total = len(results)
        
        print("\n" + "*"*60)
        print("HYPOTHESIS TESTING COMPLETE!")
        print("*"*60)
        print(f"Tests completed: {completed}/{total}")
        
        return True

# Usage example:
if __name__ == "__main__":
    goodreads_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/goodreads_dataset/edited_books.csv"
    amazon_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/amazon_book_reviews_dataset/edited_amazon_books.csv"
    
    # Create hypothesis testing instance and run all tests
    hypothesis_tester = BookReviewsHypothesisTesting(goodreads_path, amazon_path)
    hypothesis_tester.run_all_hypothesis_tests()