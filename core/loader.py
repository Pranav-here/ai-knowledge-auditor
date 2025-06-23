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

def chunk_text(
    text: str,
    *,                       # force keyword args
    min_window: int = 120,   # tightest possible chunk
    max_window: int = 400,   # loosest chunk before context gets fuzzy
    target_chunks: int = 250 # how many chunks we’d like for an “average-sized” doc
):
    """
    Auto-tunes chunk size & overlap based on document length.
    • Short docs → smaller window, small overlap
    • Long docs  → larger window, bigger overlap (to keep #chunks reasonable)
    """
    doc_len = len(text)
    if doc_len == 0:
        return []

    # 1) Pick a window so (#chunks ≈ target_chunks) but clamp between min & max.
    window = int(doc_len / target_chunks)
    window = min(max(window, min_window), max_window)

    # 2) Overlap = 20-30 % of window (heuristic)
    overlap = max(int(window * 0.3), 50)

    chunks = []
    for start in range(0, doc_len, window - overlap):
        end = start + window
        chunks.append(text[start:end])

    return chunks