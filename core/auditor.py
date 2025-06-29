# core/auditor.py

import numpy as np
import torch

# Monkey-patch torch.Tensor.numpy
_orig_tensor_numpy = torch.Tensor.numpy
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        return self.tolist()
torch.Tensor.numpy = _safe_tensor_numpy

import nltk
from sklearn.metrics.pairwise import cosine_similarity


def find_best_chunk(question, model_answer, chunks, embed_model, topic_filter=None):
    if topic_filter:
        filtered = [c for c in chunks if topic_filter.lower() in c.lower()]
        chunks_to_search = filtered or chunks
    else:
        chunks_to_search = chunks

    q_emb = embed_model.encode([question])[0]
    ans_clean = model_answer.strip().split(".")[0]
    a_emb = embed_model.encode([ans_clean])[0]
    combo = (q_emb + a_emb) / 2
    chunk_embs = embed_model.encode(chunks_to_search)
    scores = cosine_similarity([combo], chunk_embs)[0]
    idx = int(np.argmax(scores))
    return chunks_to_search[idx], round(float(scores[idx]) * 100, 2)


def highlight_top_sentences(chunk, model_answer, embed_model):
    # Try NLTK sentence tokenization; if punkt is missing, fallback to splitting on periods
    try:
        sentences = nltk.sent_tokenize(chunk)
    except LookupError:
        sentences = [
            s.strip()
            for s in chunk.replace("\n", " ").split(".")
            if s.strip()
        ]

    # Embed each sentence
    sent_embs = embed_model.encode(sentences)
    ans_clean = model_answer.strip().split(".")[0]
    ans_emb = embed_model.encode([ans_clean])[0]

    # Compute similarity to answer, pick top 2
    sims = [
        (cosine_similarity([ans_emb], [se])[0][0], sent)
        for se, sent in zip(sent_embs, sentences)
    ]
    sims.sort(reverse=True, key=lambda x: x[0])
    top2 = [s for _, s in sims[:2]]

    # Bold the top-2 sentences in the chunk
    for s in top2:
        chunk = chunk.replace(s, f"**{s}**")

    return chunk