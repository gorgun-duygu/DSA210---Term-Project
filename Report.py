#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# DSA 210 Introduction to Data Science - Final Project Report

# 1. Motivation
Books have a big part in our daily lives. The aim of this project is to look through book reviews and analyize the trends of various book genres. 
Through analyzing book reviews a pattern can be found within different genres and their popularity.
Dataset for the project will be taken from Goodreads, and to enrich another set of data, for example from Amazon Book Reviews, will be used.

# 2. Research Questions
#1. Do series books receive different ratings than standalone books?
#2. How does book length impact ratings and popularity?
#3. Are there significant differences in ratings in different genres?
#4. What factors best predict a book's success/popularity?
#5. How do Goodreads and Amazon differ from each other in according to the users?

# 3. Data Collection and Processing/Creating a filtered and edited versions
# Data Sources
# (Both datasets were too large to place on github so they were edited through local folder)
# Primary Dataset: Goodreads Books
#- Source from -> Kaggle dataset containing comprehensive book information
#- Key Features -> Title, authors, ratings, genres, publication details, page counts

# Secondary Dataset: Amazon Book Reviews  
#- Source from -> Kaggle dataset of Amazon book reviews
#- Key Features -> Book metadata, user ratings, review timestamps

# Processing/Creating a filtered and edited versions
# Processing steps included:
#- Missing Value Treatment -> Median imputation for numeric, "Unknown" for categorical
#- Filtering -> Series detection, genre extraction, length categorization
#- Data Removal -> Filtered out unrealistic page counts and ratings

# 4. Exploratory Data Analysis
# Data was analysized through visualizations and obtained statistics before conducting hypothesis tests.

# 5. Hypothesis Testing
# 6 Hypothesis tests were conducted to answer research questions.
# Hypothesis 1 -> Series and standalone books have different average ratings
# Hypothesis 2 -> At least one length category has different average ratings
# Hypothesis 3 -> At least one genre has different average ratings
# Hypothesis 4 -> There is a significant correlation between popularity and average rating
# Hypothesis 5 -> Goodreads and Amazon ratings are different
# Hypothesis 6 -> There is a significant correlation between book length and popularity

# 6. Machine Learning 
# Objective in the machine learning consists of:
#- Rating Prediction 
#- Popularity Classification 
#- Clustering Analysis 
#- Genre Prediction 

