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
