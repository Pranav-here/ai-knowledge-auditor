# core/loader.py

import fitz # PyMuPDf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype='pdf') as doc:
        text = "\n".join(page.get_text() for page in doc)
    return text if text.strip() else None  # return if not empty string


def get_keywords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]  # return non stopword alphanum tokens

def chunk_text(text, window = 1500, overlap = 400):
    chunks = []
    for start in range(0, len(text), overlap):
        end = start + window
        chunks.append(text[start:end])
    return chunks