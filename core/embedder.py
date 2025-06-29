# # core/embedder.py

# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# def load_embedder():
#     return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device = "cpu").to("cpu")

# def load_summarizer():
#     return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# def load_embedder():
#     return SentenceTransformer(
#         "sentence-transformers/all-mpnet-base-v2",
#         device="cpu"
#     ).to("cpu")

# def load_summarizer():
#     try:
#         return pipeline(
#             "summarization",
#             model="sshleifer/distilbart-cnn-12-6",
#             local_files_only=True
#         )
#     except Exception as e:
#         # If it can’t find the model in cache, just disable summarization
#         print("⚠️ Summarizer unavailable:", e)
#         return None
from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def load_embedder():
    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device="cpu"
    ).to("cpu")

def load_summarizer():
    # Return a function that takes text & returns a summary
    def summarize(text, max_length=120, min_length=30):
        # We'll ignore max/min length and just pick 2 sentences
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count=2)
        # Join sentences back into one string
        return [" ".join(str(sentence) for sentence in summary)]
    return summarize
