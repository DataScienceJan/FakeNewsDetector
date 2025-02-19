# common.py
import nltk
from nltk import data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Only download NLTK data if not already installed
try:
    data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess the input text by converting to lowercase, tokenizing,
    removing stopwords, and keeping only alphabetic tokens.
    """
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    # Only keep alphabetic tokens that are not stopwords
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)
