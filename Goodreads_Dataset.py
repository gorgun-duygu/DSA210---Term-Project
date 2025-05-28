import pandas as pd
import numpy as np
import re
import os

# File paths
file_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/dgorgun_DSA210/goodreads_dataset/books.csv"
output_dir = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/dgorgun_DSA210/goodreads_dataset"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created/verified: {output_dir}")

# Load the dataset
print("Loading dataset...")
try:
    books_df = pd.read_csv(file_path, on_bad_lines='skip')
    print(f"Dataset loaded successfully: {books_df.shape}")
    print(f"Columns: {books_df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print(f"\n*** INITIAL DATA INFO ***")
print(f"Total books: {len(books_df):,}")
print(f"Columns: {len(books_df.columns)}")

# Change missing values to median for numeric columns
num_cols = books_df.select_dtypes(include=['float64', 'int64']).columns
print(f"Numeric columns found: {len(num_cols)}")
for col in num_cols:
    missing_count = books_df[col].isna().sum()
    if missing_count > 0:
        books_df[col] = books_df[col].fillna(books_df[col].median())
        print(f"  - {col}: filled {missing_count:,} missing values with median")

# Change category columns to 'Unknown'
category_cols = books_df.select_dtypes(include=['object']).columns
print(f"Categorical columns found: {len(category_cols)}")
for col in category_cols:
    missing_count = books_df[col].isna().sum()
    if missing_count > 0:
        books_df[col] = books_df[col].fillna('Unknown')
        print(f"  - {col}: filled {missing_count:,} missing values with 'Unknown'")

# IMPROVED SERIES INFORMATION EXTRACTION
print("\n*** Extracting series information ***")
books_df['is_series'] = False
books_df['series_name'] = 'Standalone'
books_df['series_position'] = np.nan

def improved_series_detection(title):
    """Enhanced series detection function"""
    if pd.isna(title) or title == 'Unknown':
        return False, 'Standalone', np.nan
    
    title_str = str(title)
    
    # Pattern 1: Original complex pattern (keep what's working)
    series_pattern = r'(.+?)\s*[\(\[]([^#\d]+)(?:#|\s+#|\s+Book\s+|,\s+Book\s+|,\s+Vol\.\s+|,\s+Vol\s+|,\s+Volume\s+|:\s+Book\s+|:\s+Volume\s+)(\d+(?:\.\d+)?)[\)\]]'
    match = re.search(series_pattern, title_str)
    if match:
        base_title = match.group(1).strip()
        series_name = match.group(2).strip()
        idx = float(match.group(3))
        return True, series_name, idx
    
    # Pattern 2: Book/Volume followed by number (not in parentheses)
    book_number_pattern = r'^(.+?)\s+(?:Book|Vol\.?|Volume)\s+(\d+(?:\.\d+)?)(?:\s|$|:)'
    match = re.search(book_number_pattern, title_str, re.IGNORECASE)
    if match:
        series_name = match.group(1).strip()
        idx = float(match.group(2))
        return True, series_name, idx
    
    # Pattern 3: Number at the end (but avoid false positives)
    # Only if the number is 1-20 (common series range) and preceded by space
    end_number_pattern = r'^(.+?)\s+(\d{1,2})$'
    match = re.search(end_number_pattern, title_str)
    if match:
        number = int(match.group(2))
        if 1 <= number <= 20:  # Likely series numbers
            series_name = match.group(1).strip()
            return True, series_name, float(number)
    
    # Pattern 4: Roman numerals (I, II, III, IV, V, etc.)
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

# Apply improved series detection
print("Applying improved series detection...")
series_info = books_df['title'].apply(improved_series_detection)
books_df['is_series'] = [info[0] for info in series_info]
books_df['series_name'] = [info[1] for info in series_info]
books_df['series_position'] = [info[2] for info in series_info]

series_count = books_df['is_series'].sum()
series_percentage = books_df['is_series'].mean() * 100
print(f"Series detection complete: {series_count:,} series detected ({series_percentage:.1f}%)")

# Get genres information
print("\n*** Extracting genres information ***")
if 'genres' in books_df.columns:
    print("Extracting primary genres from genres column...")
    books_df['genres'] = books_df['genres'].astype(str).str.lower()
    books_df['genres'] = books_df['genres'].str.replace(r'[\[\]\'"]', '', regex=True)
    
    books_df['primary_genre'] = books_df['genres'].apply(
        lambda x: x.split(',')[0].strip() if x and x != 'unknown' and x != 'nan' else 'Unknown'
    )
    
    genre_counts = books_df['primary_genre'].value_counts()
    print(f"Primary genres extracted. Top 5 genres:")
    for genre, count in genre_counts.head().items():
        print(f"  - {genre}: {count:,} books")
else:
    print("No 'genres' column found")
    books_df['primary_genre'] = 'Unknown'

# Get book length information
print("\n*** Extracting book length information ***")
if 'num_pages' in books_df.columns:
    print("Creating length categories based on page count...")
    original_count = len(books_df)
    
    # Filter out books with unrealistic page counts
    books_df = books_df[(books_df['num_pages'] > 0) & (books_df['num_pages'] < 5000)]
    filtered_count = len(books_df)
    
    if original_count != filtered_count:
        print(f"  - Filtered out {original_count - filtered_count:,} books with invalid page counts")
    
    length_bins = [0, 100, 200, 350, 500, 1000, 5000]
    length_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Epic']
    books_df['length_category'] = pd.cut(books_df['num_pages'], bins=length_bins, labels=length_labels)
    
    length_counts = books_df['length_category'].value_counts()
    print(f"Length categories created:")
    for category, count in length_counts.items():
        print(f"  - {category}: {count:,} books")
else:
    print("No 'num_pages' column found")
    books_df['length_category'] = 'Unknown'

# Get popularity information
if 'ratings_count' in books_df.columns and 'text_reviews_count' in books_df.columns:
    print("Calculating review engagement and popularity categories...")
    
    # Calculate review engagement (reviews per rating)
    books_df['review_engagement'] = books_df['text_reviews_count'] / books_df['ratings_count'].where(books_df['ratings_count'] > 0, 1)
    
    popularity_bins = [0, 100, 1000, 10000, 100000, float('inf')]
    popularity_labels = ['Unknown', 'Niche', 'Modest', 'Popular', 'Bestseller']
    books_df['popularity_category'] = pd.cut(books_df['ratings_count'], bins=popularity_bins, labels=popularity_labels)
    
    popularity_counts = books_df['popularity_category'].value_counts()
    print(f"Popularity categories created:")
    for category, count in popularity_counts.items():
        print(f"  - {category}: {count:,} books")
    
    avg_engagement = books_df['review_engagement'].mean()
    print(f"Average review engagement: {avg_engagement:.3f} reviews per rating")
else:
    print("Missing 'ratings_count' or 'text_reviews_count' columns")
    books_df['review_engagement'] = 0
    books_df['popularity_category'] = 'Unknown'

# Create final dataset
print("\n*** Creating final dataset ***")

# Define columns to keep
columns_to_keep = [
    'title', 'authors', 'average_rating', 'ratings_count', 
    'primary_genre', 'num_pages', 'length_category', 
    'is_series', 'series_name', 'series_position',
    'language_code', 'publication_date'
]

# Only keep columns that exist in the original dataset
existing_columns = [col for col in columns_to_keep if col in books_df.columns]
print(f"Standard columns available: {len(existing_columns)}/{len(columns_to_keep)}")

# Add newly created columns
new_columns = ['review_engagement', 'popularity_category']
for col in new_columns:
    if col in books_df.columns:
        existing_columns.append(col)

print(f"Final columns selected: {len(existing_columns)}")
print(f"Columns: {existing_columns}")

# Create the clean dataset
edited_books = books_df[existing_columns].copy()

print(f"\nFinal dataset shape: {edited_books.shape}")

# Quality checks
print(f"\n*** DATA QUALITY CHECK ***")
if 'average_rating' in edited_books.columns:
    avg_rating = edited_books['average_rating'].mean()
    print(f"Average rating: {avg_rating:.2f}")

if 'ratings_count' in edited_books.columns:
    avg_ratings = edited_books['ratings_count'].mean()
    print(f"Average ratings count: {avg_ratings:.0f}")

if 'is_series' in edited_books.columns:
    series_pct = edited_books['is_series'].mean() * 100
    print(f"Books in series: {series_pct:.1f}%")

if 'primary_genre' in edited_books.columns:
    known_genres = (edited_books['primary_genre'] != 'Unknown').sum()
    print(f"Books with known genres: {known_genres:,} ({known_genres/len(edited_books)*100:.1f}%)")

# Save the cleaned dataset
new_dataset_path = os.path.join(output_dir, "edited_books.csv")
try:
    edited_books.to_csv(new_dataset_path, index=False)
    print(f"\nProcessed Goodreads dataset saved to: {new_dataset_path}")
except Exception as e:
    print(f"Error saving file: {e}")

# Print final statistics
print("\n" + "*"*60)
print("PROCESSING COMPLETE!")
print("*"*60)

print(f"Total books processed: {len(edited_books):,}")
if 'average_rating' in edited_books.columns:
    avg_rating = edited_books['average_rating'].mean()
    rating_std = edited_books['average_rating'].std()
    print(f"Average rating: {avg_rating:.2f} (std: {rating_std:.2f})")

if 'ratings_count' in edited_books.columns:
    avg_ratings = edited_books['ratings_count'].mean()
    median_ratings = edited_books['ratings_count'].median()
    print(f"Average ratings per book: {avg_ratings:.0f} (median: {median_ratings:.0f})")

if 'is_series' in edited_books.columns:
    series_pct = edited_books['is_series'].mean() * 100
    series_count = edited_books['is_series'].sum()
    print(f"Books in series: {series_count:,} ({series_pct:.1f}%)")

if 'popularity_category' in edited_books.columns:
    bestsellers = (edited_books['popularity_category'] == 'Bestseller').sum()
    popular = (edited_books['popularity_category'] == 'Popular').sum()
    print(f"Bestsellers: {bestsellers:,}, Popular books: {popular:,}")

print("*"*60)