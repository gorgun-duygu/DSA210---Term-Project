import pandas as pd
import numpy as np
import re
import os


file_path = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/dgorgun_DSA210/goodreads_dataset/books.csv"
books_df = pd.read_csv(file_path, on_bad_lines='skip')
# Output directory 
output_dir = "C:/Users/gorgu/OneDrive/Masaüstü/Spring 2025/DSA 210/Project/dgorgun_DSA210/goodreads_dataset"

# For numerical columns: replace missing values with median
num_cols = books_df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    books_df[col] = books_df[col].fillna(books_df[col].median())

# Change category columns into unknown
category_cols = books_df.select_dtypes(include=['object']).columns
for col in category_cols:
    books_df[col] = books_df[col].fillna('Unknown')

# Get series information
print("Extracting series information...")
books_df['is_series'] = False
books_df['series_name'] = 'Standalone'
books_df['series_position'] = np.nan

# Search various patterns for series titles
series_pattern = r'(.+?)\s*[\(\[]([^#\d]+)(?:#|\s+#|\s+Book\s+|,\s+Book\s+|,\s+Vol\.\s+|,\s+Vol\s+|,\s+Volume\s+|:\s+Book\s+|:\s+Volume\s+)(\d+(?:\.\d+)?)[\)\]]'

def get_series_info(title):
    if pd.isna(title) or title == 'Unknown':
        return False, 'Standalone', np.nan
    
    match = re.search(series_pattern, title)
    if match:
        base_title = match.group(1).strip()
        series_name = match.group(2).strip()
        idx = float(match.group(3))
        return True, series_name, idx
    
    # Check for generalized series indicators
    if ' series' in title.lower() or 'trilogy' in title.lower() or 'duology' in title.lower():
        return True, 'Series', np.nan
    
    return False, 'Standalone', np.nan

series_info = books_df['title'].apply(get_series_info) # Get information with the function
books_df['is_series'] = [info[0] for info in series_info]
books_df['series_name'] = [info[1] for info in series_info]
books_df['series_position'] = [info[2] for info in series_info]

# Get genres information
if 'genres' in books_df.columns:
    books_df['genres'] = books_df['genres'].astype(str).str.lower()
    books_df['genres'] = books_df['genres'].str.replace(r'[\[\]\'"]', '', regex=True)
    
    books_df['primary_genre'] = books_df['genres'].apply(
        lambda x: x.split(',')[0].strip() if x and x != 'unknown' and x != 'nan' else 'Unknown'
    )

# Get book length information
if 'num_pages' in books_df.columns:
    books_df = books_df[(books_df['num_pages'] > 0) & (books_df['num_pages'] < 5000)]
    
    length_bins = [0, 100, 200, 350, 500, 1000, 5000]
    length_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Epic']
    books_df['length_category'] = pd.cut(books_df['num_pages'], bins=length_bins, labels=length_labels)
else:
    books_df['length_category'] = 'Unknown'

# Get popularity information
print("Creating popularity metrics...")
if 'ratings_count' in books_df.columns and 'text_reviews_count' in books_df.columns:
    books_df['review_engagement'] = books_df['text_reviews_count'] / books_df['ratings_count'].where(books_df['ratings_count'] > 0, 1)
    
    popularity_bins = [0, 100, 1000, 10000, 100000, float('inf')]
    popularity_labels = ['Unknown', 'Niche', 'Modest', 'Popular', 'Bestseller']
    books_df['popularity_category'] = pd.cut(books_df['ratings_count'], bins=popularity_bins, labels=popularity_labels)

# Create new columns for the edited dataset
columns_to_keep = [
    'title', 'authors', 'average_rating', 'ratings_count', 
    'primary_genre', 'num_pages', 'length_category', 
    'is_series', 'series_name', 'series_position',
    'language_code', 'publication_date'
]

# Only keep columns that exist in the original dataset
columns_to_keep = [col for col in columns_to_keep if col in books_df.columns]

# Add newly created columns
new_columns = ['review_engagement', 'popularity_category']
for col in new_columns:
    if col in books_df.columns:
        columns_to_keep.append(col)

# Create the clean dataset
edited_books = books_df[columns_to_keep].copy()

# 7. Save the cleaned dataset
new_dataset_path = os.path.join(output_dir, "edited_books.csv")
edited_books.to_csv(new_dataset_path, index=False)

print(f"New edited dataset saved and can be seen in: {new_dataset_path}")