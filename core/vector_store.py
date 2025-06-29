# core/vector_store.py
import os
import faiss
import pickle
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

# Monkey-patch for torch.Tensor.numpy errors
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return torch.Tensor.numpy(self, *args, **kwargs)
    except RuntimeError:
        return self.tolist()

torch.Tensor.numpy = _safe_tensor_numpy

# Reranker model (with graceful fallback if it fails to load)
try:
    RERANKER = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )
except Exception as e:
    print(f"⚠️ Reranker unavailable: {e}")
    RERANKER = None

INDEX_PATH = "data/faiss_index/chunks.index"
CHUNKS_PATH = "data/faiss_index/chunks.pkl"

def build_and_save_index(chunks, embed_model):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    # Normalize and build FAISS index
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
    # Embed question and answer
    q_emb = embed_model.encode([question])[0]
    a_emb = embed_model.encode([answer])[0]
    combined = (q_emb + a_emb) / 2
    norm_combined = combined / np.linalg.norm(combined)

    # FAISS search
    D, I = index.search(np.array([norm_combined]), top_k)
    top_idxs = I[0][: min(len(chunks), top_k)]
    candidates = [chunks[i] for i in top_idxs]

    # Reranking (if available)
    if RERANKER:
        pair_inputs = [(answer, chunk) for chunk in candidates]
        try:
            rerank_scores = RERANKER.predict(pair_inputs, convert_to_numpy=True)
            best_idx = int(np.argmax(rerank_scores))
            best_chunk = candidates[best_idx]
            raw_rerank = float(rerank_scores[best_idx])
            rerank_prob = expit(raw_rerank)
        except Exception as e:
            print(f"⚠️ Rerank failed: {e}")
            best_chunk = candidates[0]
            rerank_prob = 0.0
    else:
        best_chunk = candidates[0]
        rerank_prob = 0.0

    # Local sentence-level similarity
    sentences = best_chunk.split(". ")
    sent_embs = embed_model.encode(sentences)
    ans_short = answer.strip().split(".")[0]
    ans_emb_short = embed_model.encode([ans_short])[0]
    sims = (sent_embs @ ans_emb_short) / (
        np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(ans_emb_short)
    )
    local_max = float(np.max(sims)) if len(sims) > 0 else 0.0

    # Global FAISS score
    global_score = float(D[0][0]) if D.size > 0 else 0.0

    # Final blended trust score (blend: 10% global, 20% local, 70% rerank)
    blended = 0.1 * global_score + 0.2 * local_max + 0.7 * rerank_prob
    blended = max(min(blended, 1.0), 0.0)
    trust_score = round(blended * 100, 2)

    return best_chunk, trust_score, global_score, local_max, rerank_prob
