# core/auditor.py

import numpy as np
import torch

# --- Monkey-patch torch.Tensor.numpy to catch the RuntimeError ---
_orig_tensor_numpy = torch.Tensor.numpy
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        # fallback: convert to Python list
        return self.tolist()

torch.Tensor.numpy = _safe_tensor_numpy

import nltk
from sklearn.metrics.pairwise import cosine_similarity


def find_best_chunk(question, model_answer, chunks, embed_model, topic_filter=None):
    if topic_filter:
        filterend_chunks = [c for c in chunks if topic_filter.lower() in c.lower()]
        if not filterend_chunks:
            filterend_chunks = chunks
    else:
        filterend_chunks = chunks
    
    question_emb = embed_model.encode([question])[0]
    answer_clean = model_answer.strip().split(".")[0]  # just first sentence
    answer_emb = embed_model.encode([answer_clean])[0]
    combined_emb = (question_emb+answer_emb)/2
    chunks_emb = embed_model.encode(filterend_chunks)
    scores = cosine_similarity([combined_emb], chunks_emb)[0]
    best_index = int(np.argmax(scores))
    return filterend_chunks[best_index], round(float(scores[best_index]) * 100, 2)


def highlight_top_sentences(chunk, model_answer, embed_model):
    sentences = nltk.sent_tokenize(chunk)
    sentences_embs = embed_model.encode(sentences)
    answer_clean = model_answer.strip().split(".")[0]
    answer_emb = embed_model.encode([answer_clean])[0]
    similarities = [(cosine_similarity([answer_emb], [se])[0][0], s) for se,s in zip(sentences_embs, sentences)]
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_sents = [s[1] for s in similarities[:2]]
    for s in top_sents:
        chunk = chunk.replace(s, f'**{s}**')
    return chunk