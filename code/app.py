# AI Knowledge Auditor – MVP v1.8
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
    2. Enter the **question** you asked an AI model  
    3. Enter the **answer** the model gave  
    4. We'll highlight the most relevant passage from the PDF and calculate a **Trust Score**  
    ---
    The **Trust Score** shows how closely the model's answer aligns with the PDF content using semantic similarity.
    """)


# App Title and Description
st.title("🧠 AI Knowledge Auditor")
st.caption("Upload a PDF and audit AI-generated answers for accuracy and relevance.")

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
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Load a Pretrained Sentence transformer


# Search for the best-matching chunk of PDF text based on keyword overlap
def find_best_chunk(question, context, window=500):
    chunks = [context[i:i+window] for i in range(0, len(context), window)]
    question_emb = st.session_state.embed_model.encode([question])[0]
    chunk_embs = st.session_state.embed_model.encode(chunks)
    scores = cosine_similarity([question_emb], chunk_embs)[0]
    best_index = int(np.argmax(scores))
    best_chunk = chunks[best_index]
    best_score = round(float(scores[best_index]) * 100, 2)
    return best_chunk, best_score


# Highlight sentences in bold for visual clarity
def highlight_top_sentences(chunk, model_answer):
    sentences = nltk.sent_tokenize(chunk)  # split sentences
    sentence_embeddings = st.session_state.embed_model.encode(sentences)  # embed the sentences
    # compute cosine similarity between questions and each sentence
    question_emb = st.session_state.embed_model.encode([model_answer])[0]
    similarities = []
    for i in range(len(sentences)):
        sim_score = cosine_similarity([question_emb], [sentence_embeddings[i]])[0][0]
        similarities.append((sim_score, sentences[i]))

    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sent[1] for sent in similarities[:2]]


    for sentence in top_sentences:
        highlighted_chunk = chunk.replace(sentence, '**'+sentence+'**')
    
    return highlighted_chunk


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
    # question = st.chat_input("Ask a question about the PDF...")

    if uploaded_pdf:
        # Initialize form inputs in session_state
        if "form_question" not in st.session_state:
            st.session_state.form_question = ""
        if "form_answer" not in st.session_state:
            st.session_state.form_answer = ""

        with st.form(key="audit_form"):
            question = st.text_input("🔍 What was the question you asked the model?", value=st.session_state.form_question, key="form_question")
            model_answer = st.text_area("🧠 What answer did the model give?", value=st.session_state.form_answer, key="form_answer")
            submitted = st.form_submit_button("Audit Answer")


        if submitted and model_answer:
            st.session_state.messages.append({"role": "user", "content": f"Q: {question}\nA: {model_answer}"})

            chunk, trust_score = find_best_chunk(model_answer, st.session_state.pdf_text)
            highlighted = highlight_top_sentences(chunk, model_answer)

            response_md = f"📘 **Most Relevant Passage:**\n\n{highlighted}"

            with st.chat_message("assistant"):
                st.markdown(response_md)
                st.metric(label="Trust Score", value=f"{trust_score}%")
                if trust_score < 40:
                    st.warning("⚠️ Low trust score. The model's answer may not be well-supported by the document.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{response_md}\n\n📊 **Trust Score:** `{trust_score}%`"
            })

            st.rerun()


else:
    st.info("📌 Upload a PDF to get started.")
