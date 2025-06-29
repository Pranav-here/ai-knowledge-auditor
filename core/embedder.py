# # core/embedder.py

# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# def load_embedder():
#     return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device = "cpu").to("cpu")

# def load_summarizer():
#     return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_embedder():
    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu"
    ).to("cpu")

def load_summarizer():
    try:
        return pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            local_files_only=True
        )
    except Exception as e:
        # If it can’t find the model in cache, just disable summarization
        print("⚠️ Summarizer unavailable:", e)
        return None
