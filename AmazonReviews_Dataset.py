import pandas as pd
import numpy as np
import re
import os
from datetime import datetime

books_data_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/amazon_book_reviews_dataset/books_data.csv"
books_rating_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/amazon_book_reviews_dataset/Books_rating.csv"
output_dir = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/amazon_book_reviews_dataset"

# Create a directory for the output if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created/verified: {output_dir}")

# Load the datasets
print("Loading datasets...")
try:
    books_data = pd.read_csv(books_data_path, on_bad_lines='skip')
    print(f"Books data loaded: {books_data.shape}")
    print(f"Books data columns: {books_data.columns.tolist()}")
except Exception as e:
    print(f"Error loading books data: {e}")
    exit()

try:
    books_rating = pd.read_csv(books_rating_path, on_bad_lines='skip')
    print(f"Books rating loaded: {books_rating.shape}")
    print(f"Books rating columns: {books_rating.columns.tolist()}")
except Exception as e:
    print(f"Error loading books rating: {e}")
    exit()

# Check for common columns to merge on
print("\nChecking for merge columns:")
print(f"'asin' in books_data: {'asin' in books_data.columns}")
print(f"'asin' in books_rating: {'asin' in books_rating.columns}")
print(f"'Title' in books_data: {'Title' in books_data.columns}")
print(f"'Title' in books_rating: {'Title' in books_rating.columns}")

# Standardize column names at the beginning
if 'Title' in books_data.columns:
    books_data['title'] = books_data['Title']
if 'Title' in books_rating.columns:
    books_rating['title'] = books_rating['Title']

# Processing books_data.csv
print("\nProcessing books_data.csv...")

# Change missing values to median
num_cols_data = books_data.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols_data:
    books_data[col] = books_data[col].fillna(books_data[col].median())

# Change category columns into unknown
category_cols_data = books_data.select_dtypes(include=['object']).columns
for col in category_cols_data:
    books_data[col] = books_data[col].fillna('Unknown')

# Extract genre information from categories
if 'categories' in books_data.columns:
    print("Extracting genre information...")
    
    def extract_primary_genre(categories_str):
        if pd.isna(categories_str) or categories_str == 'Unknown':
            return 'Unknown'
        
        # Clean up the categories string
        categories_str = str(categories_str).lower()
        categories_str = categories_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
        
        categories = [cat.strip() for cat in categories_str.split(',') if cat.strip()]
        return categories[0] if categories else 'Unknown'
    
    books_data['primary_genre'] = books_data['categories'].apply(extract_primary_genre)
    print("Primary genres extracted")

# Create book length categories if there isn't a price 
if 'price_category' not in books_data.columns and 'Price' in books_rating.columns:
    print("Creating length categories based on price data...")
    # Get average price per book from ratings data if available
    if 'clean_title' in books_rating.columns:
        avg_prices = books_rating.groupby('clean_title')['Price'].mean().reset_index()
        avg_prices.columns = ['clean_title', 'avg_price']
        books_data = books_data.merge(avg_prices, on='clean_title', how='left')
        
        # Create price categories as proxy for length
        price_bins = [0, 5, 10, 15, 25, 50, 1000]
        price_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Premium']
        books_data['length_category'] = pd.cut(books_data['avg_price'], bins=price_bins, labels=price_labels, include_lowest=True)
        books_data['length_category'] = books_data['length_category'].fillna('Unknown')
    else:
        books_data['length_category'] = 'Unknown'

# IMPROVED SERIES INFORMATION EXTRACTION
print("Extracting series information with improved detection...")
books_data['is_series'] = False
books_data['series_name'] = 'Standalone'
books_data['series_position'] = np.nan

def improved_series_detection(title):
    """Enhanced series detection function"""
    if pd.isna(title) or title == 'Unknown':
        return False, 'Standalone', np.nan
    
    title_str = str(title)
    
    # Pattern 1: Original complex pattern
    series_pattern = r'(.+?)\s*[\(\[]([^#\d]+)(?:#|\s+#|\s+Book\s+|,\s+Book\s+|,\s+Vol\.\s+|,\s+Vol\s+|,\s+Volume\s+|:\s+Book\s+|:\s+Volume\s+)(\d+(?:\.\d+)?)[\)\]]'
    match = re.search(series_pattern, title_str)
    if match:
        base_title = match.group(1).strip()
        series_name = match.group(2).strip()
        idx = float(match.group(3))
        return True, series_name, idx
    
    # Pattern 2: Book/Volume followed by number
    book_number_pattern = r'^(.+?)\s+(?:Book|Vol\.?|Volume)\s+(\d+(?:\.\d+)?)(?:\s|$|:)'
    match = re.search(book_number_pattern, title_str, re.IGNORECASE)
    if match:
        series_name = match.group(1).strip()
        idx = float(match.group(2))
        return True, series_name, idx
    
    # Pattern 3: Number at the end
    end_number_pattern = r'^(.+?)\s+(\d{1,2})$'
    match = re.search(end_number_pattern, title_str)
    if match:
        number = int(match.group(2))
        if 1 <= number <= 20:  # Likely series numbers
            series_name = match.group(1).strip()
            return True, series_name, float(number)
    
    # Pattern 4: Roman numerals
    roman_pattern = r'^(.+?)\s+(I{1,3}|IV|V|VI{0,3}|IX|X{1,2}|XI{0,3})$'
    match = re.search(roman_pattern, title_str)
    if match:
        roman_numerals = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12}
        roman_num = match.group(2)
        if roman_num in roman_numerals:
            series_name = match.group(1).strip()
            return True, series_name, float(roman_numerals[roman_num])
    
    # Pattern 5: Hash symbol with number
    hash_pattern = r'^(.+?)\s*#\s*(\d+(?:\.\d+)?)(?:\s|$)'
    match = re.search(hash_pattern, title_str)
    if match:
        series_name = match.group(1).strip()
        idx = float(match.group(2))
        return True, series_name, idx
    
    # Pattern 6: Parentheses with just a number
    paren_number_pattern = r'^(.+?)\s*\((\d{1,2})\)$'
    match = re.search(paren_number_pattern, title_str)
    if match:
        number = int(match.group(2))
        if 1 <= number <= 20:  # Likely series
            series_name = match.group(1).strip()
            return True, series_name, float(number)
    
    # Pattern 7: Explicit series indicators
    explicit_series = ['trilogy', 'duology', 'quartet', 'quintet', 'saga']
    for indicator in explicit_series:
        if indicator in title_str.lower():
            return True, 'Series', np.nan
    
    # Pattern 8: "Series" in parentheses or title
    if ' series' in title_str.lower() or '(series' in title_str.lower():
        return True, 'Series', np.nan
    
    # Not a series
    return False, 'Standalone', np.nan

# Apply the function to detect series
if 'title' in books_data.columns:
    series_info = books_data['title'].apply(improved_series_detection)
    books_data['is_series'] = [info[0] for info in series_info]
    books_data['series_name'] = [info[1] for info in series_info]
    books_data['series_position'] = [info[2] for info in series_info]
    print(f"Improved series detection complete: {books_data['is_series'].sum():,} series detected ({books_data['is_series'].mean()*100:.1f}%)")

# Create match_title column for matching with Goodreads
if 'title' in books_data.columns:
    books_data['match_title'] = books_data['title'].apply(lambda x: 
        str(x).lower().replace(r'[^\w\s]', '').strip() if not pd.isna(x) else "")

# Processing books_rating.csv
print("\nProcessing books_rating.csv...")

# Change missing values to median
num_cols_rating = books_rating.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols_rating:
    books_rating[col] = books_rating[col].fillna(books_rating[col].median())

# Change category columns into unknown
category_cols_rating = books_rating.select_dtypes(include=['object']).columns
for col in category_cols_rating:
    books_rating[col] = books_rating[col].fillna('Unknown')

# Extract review date information 
if 'review/time' in books_rating.columns:
    try:
        books_rating['review_date'] = pd.to_datetime(books_rating['review/time'], errors='coerce')
        books_rating['review_year'] = books_rating['review_date'].dt.year
        print("Review dates processed")
    except:
        print("Could not parse review dates")
elif 'reviewTime' in books_rating.columns:
    try:
        books_rating['review_date'] = pd.to_datetime(books_rating['reviewTime'], errors='coerce')
        books_rating['review_year'] = books_rating['review_date'].dt.year
        print("Review dates processed")
    except:
        print("Could not parse review dates")

# Process ratings - use 'review/score' column
rating_col = 'review/score' if 'review/score' in books_rating.columns else 'overall'

if rating_col in books_rating.columns:
    # Rename for consistency
    books_rating['overall'] = books_rating[rating_col]
    
    # Ensure rating is on 5-point scale
    max_rating = books_rating['overall'].max()
    if max_rating != 5.0:
        books_rating['overall'] = books_rating['overall'] * (5.0 / max_rating)
        print(f"Normalized ratings from {max_rating}-star to 5-star scale")
    
    # Create rating categories 
    rating_bins = [0, 1.5, 2.5, 3.5, 4.5, 5.1]  # 5.1 to include 5.0
    rating_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    books_rating['rating_category'] = pd.cut(books_rating['overall'], bins=rating_bins, labels=rating_labels)
    print("Rating categories created")
else:
    print("No rating column found in books_rating")

# Try to merge the datasets using 'title' column
amazon_merged = books_data.copy()  # Default to books_data if merge fails

# Clean titles for better matching
def clean_title_for_matching(title):
    if pd.isna(title):
        return ""
    return str(title).lower().strip().replace("'", "").replace('"', '')

if 'title' in books_rating.columns and 'title' in books_data.columns:
    print("\nBoth datasets have 'title' column. Attempting merge...")
    
    # Create clean titles for matching
    books_data['clean_title'] = books_data['title'].apply(clean_title_for_matching)
    books_rating['clean_title'] = books_rating['title'].apply(clean_title_for_matching)
    
    # Calculate aggregate metrics from ratings
    if 'overall' in books_rating.columns:
        agg_ratings = books_rating.groupby('clean_title').agg({
            'overall': ['mean', 'count', 'std']
        }).reset_index()
        
        # Flatten MultiIndex columns
        agg_ratings.columns = ['clean_title', 'amazon_avg_rating', 'amazon_ratings_count', 'amazon_rating_std']
        
        # Create popularity categories
        popularity_bins = [0, 10, 50, 200, 1000, float('inf')]
        popularity_labels = ['Unknown', 'Niche', 'Modest', 'Popular', 'Bestseller']
        agg_ratings['amazon_popularity_category'] = pd.cut(
            agg_ratings['amazon_ratings_count'], 
            bins=popularity_bins, 
            labels=popularity_labels
        )
        
        # Merge the datasets
        amazon_merged = books_data.merge(agg_ratings, on='clean_title', how='left')
        print(f"Merge successful! Shape after merge: {amazon_merged.shape}")
        
        # Fill missing values from the merge
        for col in amazon_merged.columns:
            if 'amazon_' in col and amazon_merged[col].isna().any():
                if amazon_merged[col].dtype in [np.float64, np.int64]:
                    amazon_merged[col] = amazon_merged[col].fillna(0)
                else:
                    amazon_merged[col] = amazon_merged[col].fillna('Unknown')
    else:
        print("No 'overall' ratings found in books_rating")
        
elif 'Title' in books_rating.columns and 'Title' in books_data.columns:
    print("\nTrying to merge using original 'Title' columns...")
    # Similar process but with Title columns
    books_data['clean_title'] = books_data['Title'].apply(clean_title_for_matching)
    books_rating['clean_title'] = books_rating['Title'].apply(clean_title_for_matching)
    
    rating_col = 'review/score' if 'review/score' in books_rating.columns else 'overall'
    
    if rating_col in books_rating.columns:
        agg_ratings = books_rating.groupby('clean_title').agg({
            rating_col: ['mean', 'count', 'std']
        }).reset_index()
        
        # Flatten MultiIndex columns
        agg_ratings.columns = ['clean_title', 'amazon_avg_rating', 'amazon_ratings_count', 'amazon_rating_std']
        
        # Create popularity categories
        popularity_bins = [0, 10, 50, 200, 1000, float('inf')]
        popularity_labels = ['Unknown', 'Niche', 'Modest', 'Popular', 'Bestseller']
        agg_ratings['amazon_popularity_category'] = pd.cut(
            agg_ratings['amazon_ratings_count'], 
            bins=popularity_bins, 
            labels=popularity_labels
        )
        
        # Merge the datasets
        amazon_merged = books_data.merge(agg_ratings, on='clean_title', how='left')
        print(f"Merge successful! Shape after merge: {amazon_merged.shape}")
        
        # Fill missing values from the merge
        for col in amazon_merged.columns:
            if 'amazon_' in col and amazon_merged[col].isna().any():
                if amazon_merged[col].dtype in [np.float64, np.int64]:
                    amazon_merged[col] = amazon_merged[col].fillna(0)
                else:
                    amazon_merged[col] = amazon_merged[col].fillna('Unknown')
else:
    print("Cannot merge: No common title column found")
    print("Proceeding with books_data...")

# Column mapping to match with Goodreads names
amazon_columns_mapping = {
    'Title': 'title',
    'title': 'title',
    'authors': 'authors',
    'ratingsCount': 'ratings_count',  
    'amazon_avg_rating': 'average_rating',
    'amazon_ratings_count': 'ratings_count',  
    'price_category': 'length_category',
    'is_series': 'is_series',
    'series_name': 'series_name',
    'series_position': 'series_position',
    'publishedDate': 'publication_date',
    'amazon_popularity_category': 'popularity_category'
}

# Create columns that match Goodreads 
for goodreads_col in ['title', 'authors', 'average_rating', 'ratings_count', 
                      'length_category', 'is_series', 'series_name', 'series_position',
                      'language_code', 'publication_date', 'popularity_category']:
    
    # Find matching column from Amazon data
    matching_col = next((k for k, v in amazon_columns_mapping.items() 
                        if v == goodreads_col and k in amazon_merged.columns), None)
    
    if matching_col and matching_col != goodreads_col:
        amazon_merged[goodreads_col] = amazon_merged[matching_col]
    elif goodreads_col not in amazon_merged.columns:
        # Create empty column if not found
        if goodreads_col in ['average_rating', 'ratings_count', 'series_position']:
            if goodreads_col == 'ratings_count' and 'ratingsCount' in amazon_merged.columns:
                amazon_merged[goodreads_col] = amazon_merged['ratingsCount']
            elif goodreads_col == 'average_rating' and 'amazon_avg_rating' in amazon_merged.columns:
                amazon_merged[goodreads_col] = amazon_merged['amazon_avg_rating']
            else:
                amazon_merged[goodreads_col] = 0
        elif goodreads_col == 'is_series':
            amazon_merged[goodreads_col] = False
        else:
            amazon_merged[goodreads_col] = 'Unknown'

# Ensure we have title column
if 'title' not in amazon_merged.columns and 'Title' in amazon_merged.columns:
    amazon_merged['title'] = amazon_merged['Title']

# Select final columns
goodreads_compatible_cols = [
    'title', 'authors', 'average_rating', 'ratings_count', 
    'length_category', 'is_series', 'series_name', 'series_position',
    'language_code', 'publication_date', 'popularity_category'
]

# Add Amazon columns that exist
amazon_cols = [col for col in amazon_merged.columns if col.startswith('amazon_')]
additional_cols = ['clean_title', 'primary_genre', 'ratingsCount', 'categories', 'publishedDate']
additional_cols = [col for col in additional_cols if col in amazon_merged.columns]

final_cols = goodreads_compatible_cols + amazon_cols + additional_cols

# Filter to only include columns that exist
final_cols = [col for col in final_cols if col in amazon_merged.columns]

# Create final dataset
edited_amazon = amazon_merged[final_cols].copy()

print(f"\nFinal dataset shape: {edited_amazon.shape}")
print(f"Final columns: {edited_amazon.columns.tolist()}")

# Print information
print(f"\nData Quality Check:")
print(f"- Non-zero ratings: {(edited_amazon['average_rating'] > 0).sum()}")
print(f"- Books with rating counts: {(edited_amazon['ratings_count'] > 0).sum()}")
print(f"- Books in series: {edited_amazon['is_series'].sum()}")
if 'primary_genre' in edited_amazon.columns:
    print(f"- Books with genres: {(edited_amazon['primary_genre'] != 'Unknown').sum()}")
if 'amazon_avg_rating' in edited_amazon.columns:
    print(f"- Books with Amazon ratings: {(edited_amazon['amazon_avg_rating'] > 0).sum()}")

# Save the processed dataset
output_path = os.path.join(output_dir, "edited_amazon_books.csv")
try:
    edited_amazon.to_csv(output_path, index=False)
    print(f"Processed Amazon dataset saved to: {output_path}")
except Exception as e:
    print(f"Error saving file: {e}")

# Save a sample of the original reviews data
if len(books_rating) > 0:
    sample_size = min(5000, len(books_rating))
    essential_cols = ['Title', 'review/score', 'review/time', 'rating_category', 'User_id']
    sample_cols = [col for col in essential_cols if col in books_rating.columns]
    
    if sample_cols:
        reviews_sample = books_rating[sample_cols].sample(sample_size, random_state=42)
        reviews_sample_path = os.path.join(output_dir, "amazon_reviews_sample.csv")
        try:
            reviews_sample.to_csv(reviews_sample_path, index=False)
            print(f"Sample of {sample_size} Amazon reviews saved to: {reviews_sample_path}")
        except Exception as e:
            print(f"Error saving reviews sample: {e}")

# Print final statistics
print("\n" + "*"*60)
print("PROCESSING COMPLETE!")
print("="*60)

print(f"Total books processed: {len(edited_amazon):,}")
if 'average_rating' in edited_amazon.columns:
    avg_rating = edited_amazon['average_rating'].mean()
    rating_std = edited_amazon['average_rating'].std()
    print(f"Average rating: {avg_rating:.2f} (std: {rating_std:.2f})")

if 'ratings_count' in edited_amazon.columns:
    avg_ratings = edited_amazon['ratings_count'].mean()
    median_ratings = edited_amazon['ratings_count'].median()
    print(f"Average ratings per book: {avg_ratings:.0f} (median: {median_ratings:.0f})")

if 'is_series' in edited_amazon.columns:
    series_pct = edited_amazon['is_series'].mean() * 100
    series_count = edited_amazon['is_series'].sum()
    print(f"Books in series: {series_count:,} ({series_pct:.1f}%)")

if 'popularity_category' in edited_amazon.columns:
    bestsellers = (edited_amazon['popularity_category'] == 'Bestseller').sum()
    popular = (edited_amazon['popularity_category'] == 'Popular').sum()
    print(f"Bestsellers: {bestsellers:,}, Popular books: {popular:,}")

print("*"*60)