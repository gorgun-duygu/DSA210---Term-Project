# Anime Popularity

# 1. Motivation
As a computer science student at Sabancı University, I often watch anime online and think what feature of the anime makes it very popular/ highly rated. Through this project, my aim is to explore the trends in anime ratings and determine the key features of the animes which have high impact on the audience and overall popularity.

------

# 2. Data Collection & Sources
- Primary dataset: "Anime Recommendations Database" from Kaggle
  Primary dataset contains information about:
  - Anime ID to match with myanimelist.net
  - Title of the anime
  - Genre
  - Type (movie, TV, OVA, etc.)
  - Number of Episodes (if its a 'Movie' then 1)
  - Rating
    
- In order to enrich another set of data is planned to be collected from MyAnimeList.net which contains additional information such as 'Release year' and 'Popularity rank'.

- Data collecting plan would be consisting of:
  - Dowloading dataset from Kaggle as primary dataset.
  - Analyzing and filtering/ cleaning the data obtained from online.
  - Utilizing MyAnimeList API to enrich the primary dataset.
  - Merge both datasets with unique variables and create a finalized csv file.
  
-----

# 3. Research Questions
- Does the genre of an anime significantly affect its ratings? 
- How does the release year of an anime affects its ratings?
- Are animes with the 'Movie' type more popular than those with the 'TV' type?
- Does the number of episodes affect an anime’s overall rating? (if the type is not 'Movie')

------

# 4. Exploratory Data Analysis & Hypothesis Tests
- The dataset collected from Kaggle was analyzed through exploratory data analysis techniques. The goal was to explore patterns in the data and answer the research questions.
## 4.1 Data Cleaning 
- Before condcuting the anlaysis, the dataset was cleaned:
  - Missing values in the dataset were identified and removed.
  - The episodes column was converted from object to numeric.
  - The first genre of the animes were taken for simplification and placed under a newly created column named 'main_genre'.
## 4.2 Exploratory Data Analysis Results 
 #### 4.2.1 Distribution of Anime Ratings
 - According to the plot results, there is a normal distrubiton centered around 6.5 - 7.0.
 - Animes rated very low and very high are less in number (rare).
 #### 4.2.2 Ratings by Anime Type
 - According to the boxplot results, animes with the type "TV" have the highest median ratings.
 #### 4.2.3 Average Rating by Main Genre 
 - According to the plot results with the top 15 genres (Josei, Mystery, Drama, Action, Police, Game, Shounen, Adventure, Military, Harem, Romance, School, Thriller, Comdey, Martial Arts), "Josei" genre animes   have the highest average rating, but closely followed by the other genres.
 #### 4.2.4 Popularity vs Rating
 - According to the scatter plot results, there is a positive relationship between the number of members (popularity) and the rating of an anime. However, the relationship is not perfectly linear.
 - Computed correlations are: "Pearson = 0.388" and "Spearmann = 0.666".
## 4.3 Hypothesis Test Results
 #### 4.3.1 H1 : Does the genre of an anime affect its ratings?
 - According to ANOVA test, F=90.807, p=0.0000. It shows that there is a significant difference in ratings between genres.
 #### 4.3.2 H2 : Are "Movie" type more popular than the type "TV"?
 - According to t-test, t=-27.729, p=0.0000. Value t being negative shows that animes with type "TV" are more popular.
 #### 4.3.3 H3 : Does the number of episodes affect an anime's rating?
 - According to correlation test, r=0.089, p=0.0000. It shows that even though the number of episodes have an affect on the ratings, its not high. In other words, there is a weak positive correlation between the number of episodes and ratings.
