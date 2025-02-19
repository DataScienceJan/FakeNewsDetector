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

