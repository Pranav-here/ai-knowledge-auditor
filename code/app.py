# AI Knowledge Auditor – MVP v1.7
# A chatbot that audits answers from PDF content using keyword matching and a trust score

import streamlit as st
import fitz  # PyMuPDF: Used to read text from PDFs
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data required for tokenizing and stopword filtering
nltk.download('punkt')
nltk.download('stopwords')

# Configure Streamlit page layout and metadata
st.set_page_config(page_title="AI Knowledge Auditor", page_icon="🧠", layout="centered")

# Sidebar with usage instructions
with st.sidebar:
    st.title("📚 How It Works")
    st.markdown("""
    1. Upload a PDF  
    2. Enter a AI generated answer
    3. We'll highlight a relevant passage and calculate a **Trust Score**  
    ---
    **Trust Score** shows how closely the answer aligns with the PDF content using semantic similarity.
    """)

# App Title and Description
st.title("🧠 AI Knowledge Auditor")
st.caption("Upload a PDF and ask a question. We’ll evaluate the accuracy of the answer using the document.")

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# PDF file uploader
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])

# Extract text from all pages of a PDF using PyMuPDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

# Get keywords from a question by tokenizing and removing stopwords
def get_keywords(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

if "embed_model" not in st.session_state:
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Load a Pretrained Sentence transformer


# Search for the best-matching chunk of PDF text based on keyword overlap
def find_best_chunk(question, context, window=500):
    chunks = [context[i:i+window] for i in range(0, len(context), window)]
    question_emb = st.session_state.embed_model.encode([question])[0]
    chunk_embs = st.session_state.emb_model.encode([chunks])
    scores = cosine_similarity([question_emb], chunk_embs)[0]
    best_index = int(np.argmax(scores))
    best_chunk = chunks[best_index]
    best_score = round(float(scores[best_index]) * 100, 2)
    return best_chunk, best_score


# Highlight matching keywords in bold for visual clarity
def highlight_keywords(chunk, keywords):
    for kw in set(keywords):
        chunk = re.sub(rf'\b({kw})\b', r'**\1**', chunk, flags=re.IGNORECASE)
    return chunk

# Extract and store PDF text once per upload
if uploaded_pdf and "pdf_text" not in st.session_state:
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("✅ PDF uploaded and processed!")

# Render past user and assistant messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Main question-answering logic (appears after PDF is uploaded)
if uploaded_pdf:
    question = st.chat_input("Ask a question about the PDF...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})  # Store user message

        keywords = get_keywords(question)  # Step 1: extract important keywords
        chunk, trust_score = find_best_chunk(question, st.session_state.pdf_text)  # Step 2: find best match

        highlighted = highlight_keywords(chunk, keywords)  # Step 3: bold matched keywords

        # Step 4: Format assistant response and display in chat bubble
        response_md = f"📘 **Answer:**\n\n{highlighted}"

        with st.chat_message("assistant"):
            st.markdown(response_md)
            st.metric(label="Trust Score", value=f"{trust_score}%")  # Visual trust metric

        # Step 5: Save assistant message with trust score to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{response_md}\n\n📊 **Trust Score:** `{trust_score}%`"
        })

        st.rerun()  # Refresh to show updated messages

else:
    st.info("📌 Upload a PDF to get started.")
