# AI Knowledge Auditor – MVP v4
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

import streamlit as st
from core.loader import extract_text_from_pdf, chunk_text
from core.embedder import load_embedder, load_summarizer
from core.auditor import find_best_chunk, highlight_top_sentences

# Configure page
st.set_page_config(page_title="AI Knowledge Auditor", page_icon="🧠", layout="centered")

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

# Load models once
if "embed_model" not in st.session_state:
    st.session_state.embed_model = load_embedder
if "summarizer" not in st.session_state:
    st.session_state.embed_model = load_summarizer

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_pdf and uploaded_pdf.size > 200 * 1024 *1024:
    st.error("🚫 File too large. Please upload a PDF under 200 MB.")
    uploaded_pdf = None

# Load and store PDF text
if uploaded_pdf and "pdf_text" not in st.session_state:
    text = extract_text_from_pdf(uploaded_pdf)
    if text:
        st.session_state.pdf_text = text
        st.session_state.chunks = chunk_text(text)
        st.success("✅ PDF uploaded and processed!")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Audit form
if uploaded_pdf:
    with st.form(key="audit_form"):
        question = st.text_input("🔍 Your question")
        model_answer = st.text_area("🧠 Model's answer")
        with st.expander("⚙️ Options"):
            show_summary = st.checkbox("📝 Show summary")
        topic_filter = st.text_input("🔎 Topic filter (optional)")
        submitted = st.form_submit_button("Audit Answer")

    if submitted and model_answer:
        chunk, score = find_best_chunk(
            question, model_answer,
            st.session_state.chunks,
            st.session_state.embed_model,
            topic_filter=topic_filter
        )
        highlighted = highlight_top_sentences(chunk, model_answer, st.session_state.embed_model)
        summary = None
        if show_summary:
            try:
                summary = st.session_state.summarizer(chunk, max_length=120, min_length=30)[0]['summary_text']
            except Exception as e:
                summary = f"⚠️ Summary failed: {str(e)}"

        trust_display = f"📊 **Trust Score:** {score}%"
        result = f"📘 **Relevant Chunk:**\n\n{highlighted}"
        if summary:
            result += f"\n\n📝 **Summary:**\n\n{summary}"
        if score < 40:
            result += "\n\n⚠️ **Low Trust:** Model answer may not be well-supported."

        st.session_state.messages.append({"role": "user", "content": f"**Q:** {question}\n**A:** {model_answer}"})
        st.session_state.messages.append({"role": "assistant", "content": f"{result}\n\n{trust_display}"})
        st.rerun()

    if st.button("🗑 Reset Chat"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("📌 Upload a PDF to begin.")