# core/loader.py

import fitz # PyMuPDf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math

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

def chunk_text(text: str, *, chunk_size: int = 500, overlap: int = 300):
    """
    Splits text into fixed-size chunks of `chunk_size` characters,
    overlapping by `overlap` characters to preserve context.
    """
    if not text:
        return []

    chunks = []
    start = 0
    doc_len = len(text)
    while start < doc_len:
        end = min(start + chunk_size, doc_len)
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks
