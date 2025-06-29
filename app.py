# AI Knowledge Auditor – Final MVP
# A chatbot that audits answers from PDF content using combined question-answer similarity

# app.py
import io
import hashlib

import numpy as np
import torch
import streamlit as st

from core.loader import extract_text_from_pdf, chunk_text
from core.embedder import load_embedder, load_summarizer
from core.vector_store import build_and_save_index, load_index_and_chunks, query_index
from core.auditor import highlight_top_sentences, find_best_chunk

# Monkey-patch torch.Tensor.numpy to avoid RuntimeError in Streamlit
_orig_tensor_numpy = torch.Tensor.numpy
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        return self.tolist()
torch.Tensor.numpy = _safe_tensor_numpy

# Configure page
st.set_page_config(page_title="AI Knowledge Auditor", page_icon="🧠", layout="centered")

# Load models once into session state
def get_models():
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = load_embedder()
    if "summarizer" not in st.session_state:
        st.session_state.summarizer = load_summarizer()

get_models()

# Initialize chat history
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

# Main UI
st.title("🧠 AI Knowledge Auditor")
st.caption("Upload a PDF and audit AI-generated answers for accuracy and relevance.")

# PDF upload & indexing
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    if len(pdf_bytes) > 200 * 1024 * 1024:
        st.error("🚫 File too large. Please upload a PDF under 200 MB.")
    else:
        file_hash = hashlib.md5(pdf_bytes).hexdigest()
        if st.session_state.get("pdf_hash") != file_hash:
            st.session_state.pdf_hash = file_hash
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

# Audit form & logic
if st.session_state.get("faiss_index"):
    with st.form("audit_form"):
        question = st.text_input("🔍 Your question")
        model_answer = st.text_area("🧠 Model's answer")
        show_summary = st.checkbox(
            "📝 Show summary",
            disabled=(st.session_state.summarizer is None)
        )
        submitted = st.form_submit_button("Audit Answer")

    if submitted and model_answer:
        # Get best chunk and similarity scores
        chunk, trust_score, global_sim, local_sim, rerank_prob = query_index(
            question,
            model_answer,
            st.session_state.embed_model,
            st.session_state.faiss_index,
            st.session_state.chunk_texts
        )
        highlighted = highlight_top_sentences(chunk, model_answer, st.session_state.embed_model)

        # Summarize if requested and available
        summary = None
        if show_summary and st.session_state.summarizer:
            try:
                summary = st.session_state.summarizer(chunk, max_length=120, min_length=30)[0]["summary_text"]
            except Exception as e:
                summary = f"⚠️ Summary failed: {e}"

        # Trust label
        if trust_score >= 75:
            label = "✅ Supported"
        elif trust_score >= 40:
            label = "⚠️ Partial Support"
        else:
            label = "❌ Likely Hallucinated"
        trust_display = f"📊 **Trust Score:** {trust_score}%  \n{label}"

        # Build the assistant’s message
        result = f"📘 **Relevant Chunk:**  \n{highlighted}"
        if summary:
            result += f"\n\n📝 **Summary:**  \n{summary}"
        if trust_score < 40:
            result += "\n\n⚠️ **Low Trust:** Model answer may not be well-supported."

        advanced = (
            "\n\n🔍 **Advanced Scoring Details:**  "
            f"\n- Global similarity: {global_sim:.4f}"
            f"\n- Local similarity: {local_sim:.4f}"
            f"\n- Rerank probability: {rerank_prob:.4f}"
        )

        # Append and rerun
        st.session_state.messages.append({"role": "user", "content": f"**Q:** {question}  \n**A:** {model_answer}"})
        st.session_state.messages.append({"role": "assistant", "content": f"{result}{advanced}\n\n{trust_display}"})
        st.rerun()

    if st.button("🗑 Reset Chat"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("📌 Upload a PDF to begin.")
