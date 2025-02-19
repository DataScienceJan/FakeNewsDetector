import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm  # For progress bar
import pickle

from common import preprocess_text  # Import the shared preprocessing function

#  Load Data
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

#  Label each set: fake=0, real=1
fake_df["label"] = 0
real_df["label"] = 1

#  Combine and shuffle data
data = pd.concat([fake_df, real_df], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

#  Check for "text" column and preprocess text
if "text" not in data.columns:
    raise KeyError("Column 'text' not found in dataset!")
data["text"] = data["text"].astype(str).fillna("").apply(lambda x: x[:1000])
tqdm.pandas()
data["clean_text"] = data["text"].progress_apply(preprocess_text)

#  Prepare features and labels
X = data["clean_text"]
y = data["label"]

#  Use TfidfVectorizer with consistent settings
vectorizer = TfidfVectorizer(max_features=5000)
X_vectors = vectorizer.fit_transform(X)

#  Split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

#  Train Logistic Regression model with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

#  Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(" Accuracy:", accuracy)

#  Optionally, save the model and vectorizer for use in the app
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
