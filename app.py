import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle

# For advanced article parsing
import requests
from bs4 import BeautifulSoup
from newspaper import Article

# If needed, uncomment these lines to ensure NLTK data is available in your environment:
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

###################################
# 1. Define your helper functions #
###################################

def fetch_text_newspaper(url):
    """
    Attempts to parse the main text of an article using newspaper3k.
    This helps avoid extra content like menus and footers.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text  # Only the main body content

def fetch_text_fallback(url):
    """
    Fallback approach using requests + BeautifulSoup if newspaper3k fails.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raises an error for non-200 status
    soup = BeautifulSoup(response.text, "html.parser")
    # Basic text extraction (may include headers, footers, etc.)
    return soup.get_text(separator=" ", strip=True)

def fetch_text_from_url(url):
    """
    Tries to fetch article text using newspaper3k first.
    Falls back to requests + BeautifulSoup if there's an error.
    """
    try:
        return fetch_text_newspaper(url)
    except Exception:
        try:
            return fetch_text_fallback(url)
        except Exception as e:
            st.error(f"Failed to fetch or parse URL: {str(e)}")
            return ""

def preprocess_text(text):
    """
    Converts text to lowercase, tokenizes, removes stopwords, 
    and keeps only alphanumeric tokens (e.g., 'covid19', '2025').
    """
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    # Keep alphanumeric tokens and remove stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

##############################################
# 2. Load your pre-trained model & vectorizer
##############################################

with open("vectorizer.pkl", "rb") as vf:
    vectorizer = pickle.load(vf)

with open("model.pkl", "rb") as mf:
    model = pickle.load(mf)

#####################################
# 3. Build the Streamlit application
#####################################

st.title("Fake News Detection with URL Parsing")

st.sidebar.header("About")
st.sidebar.info(
    "This demo uses a Logistic Regression model trained on English news data. "
    "We use newspaper3k to extract main article text and then apply the same "
    "preprocessing pipeline used during training."
)

# Text input for URLs
user_url = st.text_input("Enter a URL to a news article:")

if st.button("Fetch & Check"):
    with st.spinner("Fetching and analyzing article..."):
        # 1. Fetch the raw text from the URL
        raw_text = fetch_text_from_url(user_url)
        
        # 2. Preprocess the fetched text
        processed_text = preprocess_text(raw_text)
        
        if not processed_text.strip():
            st.warning("No valid text found at this URL (or text is very short).")
        else:
            # 3. Vectorize and predict
            input_vector = vectorizer.transform([processed_text])
            prediction_proba = model.predict_proba(input_vector)[0]
            prob_fake, prob_real = prediction_proba

            # 4. Apply a custom threshold (optional)
            threshold = 0.60
            if prob_real >= threshold:
                st.success(f"This news seems REAL! (Confidence: {prob_real:.2f})")
            else:
                st.error(f"This news seems FAKE! (Confidence: {prob_fake:.2f})")
