# Helper file with a simple build_index() that accepts the uploaded PDF and writes auditor.idx alongside it.

import fitz
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(uploaded_file):
    """
        Read the entire pdf as one long string    
    """
    uploaded_file.seek(0)
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        texts = [page.get_text() for page in doc]
    return "\n".join(texts)


def split_into_chunks(context, window=500):
    return [context[i:i+window] for i in range(0, len(context), window)]
    

def build_index(uploaded_file):
    """
        Build a FAISS index from the uploaded PDF and save it to disk.
        returns (index, metadata)
    """
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_into_chunks(text, window=500)

    embeds = _model.encode(chunks, convert_to_numpy=True).astype("float32")
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)

    metadata = {i: {"text": chunks[i]} for i in range(len(chunks))}
    return index, metadata
