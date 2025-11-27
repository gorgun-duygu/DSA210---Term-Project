# EDA & Hypothesis Testing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Get the path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
csv_path = os.path.join(parent_dir, "kaggle_anime_database.csv")

# Ensure folders exist in the main directory
# Folders to store the plots and the summary
outputs_dir = os.path.join(parent_dir, "outputs")
reports_dir = os.path.join(parent_dir, "reports")
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
# Load Data
df = pd.read_csv(csv_path)
print("Data loaded successfully!")

print(df.head())

# Clean Data
# Remove the rows with missing ratings 
df = df.dropna(subset=["rating"])
# Convert the columns to numeric 
df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
df["members"] = pd.to_numeric(df["members"], errors="coerce")

# Take the first genre only for simplfification
df["main_genre"] = df["genre"].astype(str).str.split(",").str[0].str.strip()

# Save the basic information to a text file
with open(os.path.join(reports_dir, "eda_results.txt"), "w") as f:
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Columns: {list(df.columns)}\n")
    f.write(str(df.describe()))

# EDA
# Rating Distribution -> plot
plt.figure(figsize=(7,4))
sns.histplot(df["rating"], bins=30, kde=True)
plt.title("Distribution of Anime Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig(os.path.join(outputs_dir, "rating_distribution.png"))
plt.close()

# Rating by Type -> plot
plt.figure(figsize=(8,5))
sns.boxplot(x="type", y="rating", data=df)
plt.title("Ratings by Anime Type")
plt.savefig(os.path.join(outputs_dir, "ratings_by_type.png"))
plt.close()

# Average Rating by Genre -> plot
avg_by_genre = df.groupby("main_genre")["rating"].mean().sort_values(ascending=False).head(15)
plt.figure(figsize=(10,6))
sns.barplot(x=avg_by_genre.index, y=avg_by_genre.values)
plt.title("Top 15 Genres by Average Rating")
plt.xticks(rotation=45)
plt.ylabel("Average Rating")
plt.savefig(os.path.join(outputs_dir, "top_genres_avg_rating.png"))
plt.close()

# Popularity vs Rating -> plot
plt.figure(figsize=(7,5))
sns.scatterplot(x=np.log1p(df["members"]), y=df["rating"], alpha=0.4)
plt.title("Popularity (log members) vs Rating")
plt.xlabel("Log (Members)")
plt.ylabel("Rating")
plt.savefig(os.path.join(outputs_dir, "popularity_vs_rating.png"))
plt.close()

# Correlations between rating and members
pearson_corr = df["rating"].corr(df["members"], method="pearson")
spearman_corr = df["rating"].corr(df["members"], method="spearman")

# Record EDA findings to a text file
with open(os.path.join(reports_dir, "eda_results.txt"), "a") as f:
    f.write("\n\nEDA Findings:\n")
    f.write(f"Pearson correlation - rating vs members: {pearson_corr:.3f}\n")
    f.write(f"Spearman correlation - rating vs members: {spearman_corr:.3f}\n")

# Hypothesis Testing
# H0: The genre of an anime does not affect its ratings.
# H1: The genre of an anime affects its ratings.
results = []

# H1 -> ANOVA test
genres = df["main_genre"].value_counts().head(6).index
samples = [df.loc[df["main_genre"] == g, "rating"] for g in genres]
f_stat, p_val = stats.f_oneway(*samples)
results.append(f"ANOVA - Genre vs Rating: F={f_stat:.3f}, p={p_val:.4f}")

# H2: Are Movies more popular than TV? 
# H2 -> Welch’s independent t-test
movies = np.log1p(df.loc[df["type"] == "Movie", "members"].dropna())
tv = np.log1p(df.loc[df["type"] == "TV", "members"].dropna())
t_stat, p_val = stats.ttest_ind(movies, tv, equal_var=False)
results.append(f"t-test - Movie vs TV Popularity: t={t_stat:.3f}, p={p_val:.4f}")

# H3: Does number of episodes affect rating? 
# H3 -> Correlation test
episodes = df["episodes"].dropna()
ratings = df.loc[episodes.index, "rating"]
corr, p_val = stats.pearsonr(episodes, ratings)
results.append(f"Correlation - Episodes vs Rating: r={corr:.3f}, p={p_val:.4f}")

# Write the test results
with open(os.path.join(reports_dir, "eda_results.txt"), "a") as f:
    f.write("\n\nHypothesis Tests:\n")
    for r in results:
        f.write(r + "\n")

print("Analysis completed!")
print("Analysis results are saved in the reports folder and the created plots are saved in the outputs folder.")