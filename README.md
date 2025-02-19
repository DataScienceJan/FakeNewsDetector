# Fake News Detection Project

A simple **Fake News Detector** built with **Python**, **scikit-learn**, and **Streamlit**. This project uses **Natural Language Processing (NLP)** techniques to classify news articles as "Real" or "Fake."

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Training](#model-training)
8. [Screenshots](#screenshots)
9. [License](#license)

---

## Project Overview

The goal of this project is to showcase an end-to-end NLP pipeline for detecting fake news. We:
- Scrape or parse an article’s main text from a given URL using **newspaper3k**.
- Clean and preprocess the text (tokenization, removing punctuation, lowercasing, etc.).
- Convert text into numerical features using either **CountVectorizer** or **TfidfVectorizer**.
- Train a **Logistic Regression** classifier (though other models can be used).
- Deploy a user-friendly **Streamlit** app that predicts if an article is “Real” or “Fake” in real time.

---

## Features

- **URL Parsing**: Enter a URL, and the app automatically fetches and processes the article text.
- **Text Preprocessing**: Cleans and normalizes text for improved model performance.
- **Classification**: Logistic Regression classifier outputs a prediction and confidence score.
- **Interactive UI**: Powered by Streamlit, providing real-time feedback on predictions.

---

## Technologies Used

- **Python 3.9+** (or 3.10+)
- **Streamlit** for building the web application.
- **scikit-learn** for machine learning (Logistic Regression, vectorization, etc.).
- **newspaper3k** for article scraping.
- **nltk** for text preprocessing and tokenization.
- **pandas** and **numpy** for data manipulation and analysis.

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/<YourUsername>/fake_news_detector.git
   cd fake_news_detector

   Install dependencies:

bash
Kopier
pip install -r requirements.txt
If you haven’t installed NLTK data locally, you may need:

python
Kopier
import nltk
nltk.download('punkt')
(You can also download other corpora as needed.)

(Optional) Create and activate a virtual environment:

bash
Kopier
python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate   # On Windows
Usage
Run the Streamlit app:
bash
Kopier
streamlit run app.py
Open your browser to the URL provided in the terminal (usually http://localhost:8501).
Enter a news article URL and click "Fetch & Check" to see the prediction.
Dataset
We used Fake.csv and True.csv from Kaggle’s Fake News dataset (or a similar dataset).
Make sure the dataset is included in your project folder (if allowed by license) or provide a link to where users can download it.
Model Training
Data Preparation:
Combine and label real/fake data from CSVs.
Clean and preprocess (remove punctuation, stopwords, etc.).
Vectorization:
Use CountVectorizer or TfidfVectorizer from scikit-learn.
Train/Test Split:
train_test_split for an 80/20 or 70/30 split.
Model:
Logistic Regression for simplicity.
You can experiment with RandomForest or DistilBERT (transformers) for better performance.
Evaluation:
Accuracy, Precision, Recall, F1-score.
Confusion matrix to visualize true vs. false predictions.
Screenshots
1. App Interface
A screenshot of the Streamlit app predicting news authenticity.

2. Example Output
The app classifies the article as "REAL" or "FAKE" with a confidence score.

License
This project is licensed under the MIT License - feel free to modify and distribute as you see fit.
(If your dataset has specific usage terms, add them here too.)

Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Contact
