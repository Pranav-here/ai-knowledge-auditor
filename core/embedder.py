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
# core/embedder.py

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
    """
    Returns a function: summarize(text, max_length, min_length) -> [summary_str]
    Uses LexRank to pick the top 5 sentences.
    """
    def summarize(text, max_length=120, min_length=30):
        # parse the raw chunk
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        # extract the 5 most “central” sentences
        summary_sents = summarizer(parser.document, sentences_count=5)
        # join them into one string
        joined = " ".join(str(sentence) for sentence in summary_sents)
        return [joined]
    return summarize
