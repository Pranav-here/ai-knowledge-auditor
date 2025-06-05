# 🧠 AI Knowledge Auditor – MVP v1.5
# A chatbot that answers questions based on a PDF using keyword matching and gives a "Trust Score"

import streamlit as st
import fitz  # PyMuPDF: Used to extract text from PDF files
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 🔁 Download NLTK data (only runs once; used for tokenizing and filtering out common words)
nltk.download('punkt')
nltk.download('stopwords')

# 🚀 Set up the main Streamlit page
st.set_page_config(page_title="AI Knowledge Auditor", page_icon="🧠", layout="centered")
st.title("🧠 AI Knowledge Auditor")
st.caption("Upload a PDF and ask a question about its content. We'll show a contextual answer with a trust score.")

# 💬 Store chat messages (questions + responses) between user and assistant
if "messages" not in st.session_state:
    st.session_state.messages = []

# 📄 Let user upload a PDF file
uploaded_pdf = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

# 🧾 Extract all text from uploaded PDF (page by page)
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

# 🔍 Get keywords from a user's question (removes common English words)
def get_keywords(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in stop_words]

# 🧠 Find the most relevant chunk of text in the PDF that matches the user's question
def find_best_chunk(question, context, window=500):
    keywords = get_keywords(question)
    chunks = [context[i:i+window] for i in range(0, len(context), window)]  # break PDF text into chunks
    best_chunk, max_matches = "", 0
    for chunk in chunks:
        match_count = sum(1 for kw in keywords if kw in chunk.lower())
        if match_count > max_matches:
            best_chunk = chunk
            max_matches = match_count
    return best_chunk, max_matches, len(keywords)

# ✨ Highlight (bold) matching keywords in the selected text chunk
def highlight_keywords(chunk, keywords):
    for kw in set(keywords):
        chunk = re.sub(rf'\b({kw})\b', r'**\1**', chunk, flags=re.IGNORECASE)
    return chunk

# ✅ If PDF is uploaded for the first time, extract and store its text
if uploaded_pdf and "pdf_text" not in st.session_state:
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("✅ PDF uploaded and processed!")

# 💬 Show chat history (questions and AI responses) as messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 🧠 Once PDF is uploaded, allow user to type questions
if uploaded_pdf:
    question = st.chat_input("Ask your question here...")
    if question:
        # 1️⃣ Save user question
        st.session_state.messages.append({"role": "user", "content": question})

        # 2️⃣ Extract keywords from the question
        keywords = get_keywords(question)

        # 3️⃣ Find the best-matching text chunk from the PDF
        chunk, match_count, total_keywords = find_best_chunk(question, st.session_state.pdf_text, window=600)

        # 4️⃣ Calculate a simple "trust score" based on how many keywords matched
        trust_score = round((match_count / total_keywords) * 100, 2) if total_keywords > 0 else 0.0

        # 5️⃣ Highlight the matching keywords in the chosen chunk
        highlighted = highlight_keywords(chunk, keywords)

        # 6️⃣ Build the final assistant response
        answer = (
            f"📘 **Answer based on the PDF:**\n\n"
            f"{highlighted}\n\n"
            f"📊 **Trust Score:** `{trust_score}%`"
        )

        # 7️⃣ Save assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 🔄 Refresh the app to show new message
        st.rerun()
else:
    st.info("📌 Please upload a PDF to begin.")
