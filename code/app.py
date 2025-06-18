# AI Knowledge Auditor – MVP v2.5
# A chatbot that audits answers from PDF content using combined question-answer similarity

# app.py – at the very top, before anything else
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
import fitz               # PyMuPDF
import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer  # ② torch gets imported after NumPy
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

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

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])

# Extract text from PDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

# Extract keywords (unused currently, but can help with extensions)
def get_keywords(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

# Load SentenceTransformer model once
if "embed_model" not in st.session_state:
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").to("cpu")

if "summarizer" not in st.session_state:
    st.session_state.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Combined similarity for chunk search
def find_best_chunk(question, model_answer, context, topic_filter=None, window=1000, overlap=250):
    chunks = []
    for start in range(0, len(context), overlap):
        end = start+window
        chunks.append(context[start: end])
    if topic_filter:
        filtered_chunks = [chunk for chunk in chunks if topic_filter.lower() in chunk.lower()]
        if not filtered_chunks:
            filtered_chunks = chunks
    else:
        filtered_chunks = chunks
    question_emb = st.session_state.embed_model.encode([question])[0]
    answer_emb = st.session_state.embed_model.encode([model_answer])[0]
    combined_emb = (question_emb + answer_emb) / 2
    chunk_embs = st.session_state.embed_model.encode(filtered_chunks)
    scores = cosine_similarity([combined_emb], chunk_embs)[0]
    best_index = int(np.argmax(scores))
    best_chunk = filtered_chunks[best_index]
    best_score = round(float(scores[best_index]) * 100, 2)
    return best_chunk, best_score

# Highlight top 2 semantically relevant sentences
def highlight_top_sentences(chunk, model_answer):
    sentences = nltk.sent_tokenize(chunk)
    sentence_embeddings = st.session_state.embed_model.encode(sentences)
    answer_emb = st.session_state.embed_model.encode([model_answer])[0]
    similarities = [(cosine_similarity([answer_emb], [sent_emb])[0][0], sent) for sent_emb, sent in zip(sentence_embeddings, sentences)]
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sent[1] for sent in similarities[:2]]
    for sentence in top_sentences:
        chunk = chunk.replace(sentence, f'**{sentence}**')
    return chunk

def summarize_chunk(chunk):
    try:
        summary = st.session_state.summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        return summary.strip().replace("\n", " ")
    except Exception as e:
        return f"⚠️ Summary failed: {str(e)}"

# Load and store PDF text
if uploaded_pdf and "pdf_text" not in st.session_state:
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("✅ PDF uploaded and processed!")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Form for auditing after PDF upload
if uploaded_pdf:
    if "form_question" not in st.session_state:
        st.session_state.form_question = ""
    if "form_answer" not in st.session_state:
        st.session_state.form_answer = ""

    with st.form(key="audit_form"):
        question = st.text_input("🔍 What was the question you asked the model?", value=st.session_state.form_question, key="form_question")
        model_answer = st.text_area("🧠 What answer did the model give?", value=st.session_state.form_answer, key="form_answer")
        show_summary = st.checkbox("📝 Show summary of this chunk")
        topic_filter = st.text_input("🔎 Optional Topic Filter (e.g., 'machine learning')", key="topic_filter")
        submitted = st.form_submit_button("Audit Answer")

    if submitted and model_answer:
        chunk, trust_score = find_best_chunk(question, model_answer, st.session_state.pdf_text, topic_filter=topic_filter)
        highlighted = highlight_top_sentences(chunk, model_answer)
        summary = summarize_chunk(chunk) if show_summary else None

        response_md = f"📘 **Most Relevant Passage:**\n\n{highlighted}"
        if show_summary and summary:
            response_md += f"\n\n📝 **Summary:**\n\n{summary}"

        trust_display = f"📊 Trust Score\n\n**{trust_score}%**"
        warning_text = ""
        if trust_score < 40:
            warning_text = "⚠️ Low Trust Warning\nThe model's answer may not be well-supported by the document."

        st.session_state.messages.append({
            "role": "user",
            "content": f"Q: {question}\nA: {model_answer}"
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{response_md}\n\n{trust_display}\n\n{warning_text}"
        })

        st.session_state.reset_inputs = True
        st.session_state.submitted = True
        st.rerun()



else:
    st.info("📌 Upload a PDF to get started.")
