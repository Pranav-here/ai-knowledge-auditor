import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Load the embedding model once
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure NLTK punkt tokenizer is available
nltk.download("punkt", quiet=True)


def find_best_chunk(question: str, answer: str, index, metadata: dict, top_k: int = 3,) -> tuple[str, float]:
    """
        Given a question and the model's answer, search the FAISS index to
        find the most relevant text chunk and return (chunk_text, trust_score_percent).
    """
    # 1. Embed question and answer, then average
    q_vec = _embed_model.encode([question], convert_to_numpy=True)[0]
    a_vec = _embed_model.encode([answer],   convert_to_numpy=True)[0]
    combo = ((q_vec + a_vec) / 2).astype("float32")

    # 2. FAISS search
    scores, ids = index.search(np.array([combo]), top_k)
    best_id    = int(ids[0][0])
    best_score = float(scores[0][0]) * 100  # percent

    # 3. Retrieve chunk text
    chunk_text = metadata[best_id]["text"]
    return chunk_text, round(best_score, 2)


def compute_trust(question: str, answer: str, chunk: str,) -> float:
    """
        Compute a raw cosine‐similarity trust score (0–100%) between the
        averaged QA embedding and the chunk embedding.
    """
    q_vec = _embed_model.encode([question], convert_to_numpy=True)[0]
    a_vec = _embed_model.encode([answer],   convert_to_numpy=True)[0]
    c_vec = _embed_model.encode([chunk],    convert_to_numpy=True)[0]
    combo = ((q_vec + a_vec) / 2).astype("float32")
    score = cosine_similarity([combo], [c_vec])[0][0] * 100
    return round(float(score), 2)


def compute_band(score_percent: float) -> str:
    """
        Map a percent score into "green"/"amber"/"red" bands.
    """
    score = score_percent / 100.0  # normalize to 0–1
    if score >= 0.80:
        return "green"
    if score >= 0.50:
        return "amber"
    return "red"


def highlight(chunk: str, answer: str, num_sentences: int = 2,) -> str:
    """
        Find the top-N sentences in `chunk` most similar to `answer`
        and wrap them in bold + larger font HTML spans.
    """
    # 1. Sentence split & embed
    sentences    = nltk.sent_tokenize(chunk)
    sent_embs    = _embed_model.encode(sentences, convert_to_numpy=True)
    answer_emb   = _embed_model.encode([answer], convert_to_numpy=True)[0]

    # 2. Compute similarities
    sims = cosine_similarity([answer_emb], sent_embs)[0]
    top_idxs = np.argsort(sims)[-num_sentences:]

    # 3. Highlight those sentences
    highlighted = chunk
    for idx in sorted(top_idxs, reverse=True):
        sent = sentences[idx]
        highlighted = highlighted.replace(
            sent,
            f'<span style="font-weight:bold; font-size:1.1rem;">{sent}</span>'
        )

    return highlighted
