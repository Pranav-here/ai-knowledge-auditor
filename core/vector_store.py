# core/vector_store.py
import os
import faiss
import pickle
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Reranker model
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

INDEX_PATH = "data/faiss_index/chunks.index"
CHUNKS_PATH = "data/faiss_index/chunks.pkl"

# Monkey-patch for torch.Tensor.numpy errors
_orig_tensor_numpy = torch.Tensor.numpy
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        return self.tolist()
torch.Tensor.numpy = _safe_tensor_numpy

def build_and_save_index(chunks, embed_model):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    norm_embs = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
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

def query_index(question, answer, embed_model, index, chunks, top_k=5):
    # Combine question+answer embedding
    q_emb = embed_model.encode(question)[0]
    a_emb = embed_model.encode([answer])[0]
    combined = (q_emb + a_emb) / 2
    norm_combined = combined / np.linalg.norm(combined)
    # FAISS search
    D, I = index.search(np.array([norm_combined]), top_k)
    top_idxs = I[0][:min(len(chunks), top_k)]
    candidates = [chunks[i] for i in top_idxs]
    # Rerank with Cross-Encoder
    pair_inputs = [(answer, chunk) for chunk in candidates]
    rerank_scores = RERANKER.predict(pair_inputs, convert_to_numpy=True)
    best_rerank = int(np.argmax(rerank_scores))
    best_chunk = candidates[best_rerank]
    raw_rerank = float(rerank_scores[best_rerank])
    rerank_prob = expit(raw_rerank)
    # Local sentence-level sim
    sentences = best_chunk.split(". ")
    sent_embs = embed_model.encode(sentences)
    ans_emb_short = embed_model.encode([answer.strip().split(".")[0]])[0]
    sims = (sent_embs @ ans_emb_short) / (
        np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(ans_emb_short)
    )
    local_max = float(sims.max())
    # Global FAISS score
    global_score = float(D[0][0])
    # Optional negation penalty
    if any(neg in best_chunk.lower() for neg in ["not", "no", "cannot", "never"]) and \
       any(affirm in answer.lower() for affirm in ["can", "allowed", "must", "yes"]):
        rerank_prob = max(rerank_prob - 0.40, 0.0)
    # Final blend
    blended = 0.1 * global_score + 0.2 * local_max + 0.7 * rerank_prob
    blended = max(min(blended, 1.0), 0.0)
    trust_score = round(blended * 100, 2)
    return best_chunk, trust_score, global_score, local_max, rerank_prob
