Student Feedback Analysis Project

Overview
This project aims to analyze and categorize student feedback using machine learning techniques and ensure the comments are free of harmful language. The main components of the project include preprocessing with NLTK, LSTM model training for feedback categorization, and integration of profanity filtering using 'better_profanity' and HuggingFace BERT. The UI is developed using Flask, featuring a star rating system for clarity in feedback evaluation.

Features
Data Input: Student feedback data is sourced from 'student_feedback.csv'.
Preprocessing: NLTK is used to preprocess and tokenize feedback comments.
Machine Learning Model: LSTM model is employed to categorize feedback into five levels: excellent, good, neutral, bad, and very bad.
Profanity Filtering: 'better_profanity' library identifies and replaces harmful words with euphemisms using HuggingFace BERT.
User Interface: Flask framework is used to create a user-friendly interface, enhancing readability with a star rating method for feedback evaluation.
