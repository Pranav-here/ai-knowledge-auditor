# core/embedder.py

from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", device = "cpu").to("cpu")

def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")