# AI Knowledge Auditor – MVP v5
# A chatbot that audits answers from PDF content using combined question-answer similarity

# app.py
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

import io
import hashlib
import streamlit as st
from core.loader import extract_text_from_pdf, chunk_text
from core.embedder import load_embedder, load_summarizer
from core.vector_store import build_and_save_index, load_index_and_chunks, query_index
from core.auditor import highlight_top_sentences, find_best_chunk

# Configure page
st.set_page_config(page_title="AI Knowledge Auditor", page_icon="🧠", layout="centered")

# Load models once into session state
def get_models():
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = load_embedder()
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = load_summarizer()
get_models()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar instructions
with st.sidebar:
    st.title("📚 How It Works")
    st.markdown("""
    1. Upload a PDF  
    2. Enter the **question** you asked an AI model  
    3. Enter the **answer** the model gave  
    4. We'll highlight the most relevant passage from the PDF and calculate a **Trust Score**  
    ---
    The **Trust Score** shows how closely the model's answer aligns with the PDF content using semantic similarity.
    """)

# Main title
st.title("🧠 AI Knowledge Auditor")
st.caption("Upload a PDF and audit AI-generated answers for accuracy and relevance.")

# PDF uploader with caching based on file hash
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.read()
    if len(pdf_bytes) > 200 * 1024 * 1024:
        st.error("🚫 File too large. Please upload a PDF under 200 MB.")
    else:
        file_hash = hashlib.md5(pdf_bytes).hexdigest()
        # Only rebuild index if the file has changed
        if st.session_state.get("pdf_hash") != file_hash:
            st.session_state.pdf_hash = file_hash
            # Build index with spinner
            with st.spinner("Indexing PDF... This may take a moment."):
                text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
                if not text:
                    st.error("🚫 No text found in PDF.")
                else:
                    chunks = chunk_text(text)
                    build_and_save_index(chunks, st.session_state.embed_model)
                    idx, chunks_list = load_index_and_chunks()
                    st.session_state.faiss_index = idx
                    st.session_state.chunk_texts = chunks_list
                    st.success("✅ PDF uploaded and indexed!")
        else:
            st.info("✅ Using cached index for this PDF.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Audit form and result
if st.session_state.get("faiss_index"):
    with st.form(key="audit_form"):
        question = st.text_input("🔍 Your question")
        model_answer = st.text_area("🧠 Model's answer")
        show_summary = st.checkbox("📝 Show summary")
        submitted = st.form_submit_button("Audit Answer")

    if submitted and model_answer:
        # Query index and get raw sub-scores
        chunk, trust_score, global_sim, local_sim, rerank_prob = query_index(
            question,
            model_answer,
            st.session_state.embed_model,
            st.session_state.faiss_index,
            st.session_state.chunk_texts
        )
        highlighted = highlight_top_sentences(chunk, model_answer, st.session_state.embed_model)

        # Optional summary
        summary = None
        if show_summary:
            try:
                summary = st.session_state.summarizer(chunk, max_length=120, min_length=30)[0]['summary_text']
            except Exception as e:
                summary = f"⚠️ Summary failed: {str(e)}"

        # Labels
        label = (
            "✅ Supported" if trust_score >= 75 else
            "⚠️ Partial Support" if trust_score >= 40 else
            "❌ Likely Hallucinated"
        )
        trust_display = f"📊 **Trust Score:** {trust_score}%  \n{label}"

        # Build assistant message
        result = f"📘 **Relevant Chunk:**  \n{highlighted}"
        if summary:
            result += f"\n\n📝 **Summary:**  \n{summary}"
        if trust_score < 40:
            result += "\n\n⚠️ **Low Trust:** Model answer may not be well-supported."

        # Advanced sub-scores
        advanced = (
            "\n\n🔍 **Advanced Scoring Details:**  "
            f"\n- Global similarity: {global_sim:.4f}"
            f"\n- Local similarity: {local_sim:.4f}"
            f"\n- Rerank probability: {rerank_prob:.4f}"
        )

        # Append messages and rerun
        st.session_state.messages.append({"role": "user", "content": f"**Q:** {question}  \n**A:** {model_answer}"})
        st.session_state.messages.append({"role": "assistant", "content": f"{result}{advanced}\n\n{trust_display}"})
        st.rerun()

    if st.button("🗑 Reset Chat"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("📌 Upload a PDF to begin.")
