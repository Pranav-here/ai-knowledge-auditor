# core/vector_store.py

import os
import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
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

INDEX_PATH="data/faiss_index/chunks.index"
CHUNKS_PATH="data/faiss_index/chunks.pkl"


def build_and_save_index(chunks, embed_model):
    os.makedirs("data/faiss_index", exist_ok=True)
    embeddings = embed_model.encode(chunks, show_progress_bar = True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    norm_embs = embeddings /np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(norm_embs)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index_and_chunks():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def query_index(question, answer, embed_model, index, chunks, top_k=1):
    q_emb = embed_model.encode(question)[0]
    a_emb = embed_model.encode([answer])[0]
    combined = (q_emb+a_emb)/2
    norm_combined = combined/np.linalg.norm(combined)

    D, I = index.search(np.array([norm_combined]), top_k)
    best_idx = I[0][0]
    # Cap and scale to get more natural scores
    raw_score = float(D[0][0])
    best_score = round(min(1.0, max(0.0, raw_score)) * 100, 2)
    # best_score = round((raw_score ** 1.3) * 100, 2)  # slight boost to higher scores
    return chunks[best_idx], best_score
