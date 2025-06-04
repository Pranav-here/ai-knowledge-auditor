# Import statements
import streamlit as st
import fitz  # This is PyMuPDF, used to open and read text from PDF files

# Set up the app’s title, icon, and layout style
st.set_page_config(page_title="AI Knowledge Auditor", page_icon="🧠", layout="centered")

st.title("🧠 AI Knowledge Auditor")

# Small helpful instruction just below the title
st.caption("Upload a PDF, then ask a question about its content.")

# This helps us keep track of the whole chat (user questions + assistant replies)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Loop through all messages in history and show them in the chat box format
# It shows both the user's and assistant's past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # 'role' is either 'user' or 'assistant'
        st.markdown(msg["content"])     # 'content' is the actual text of the message

# File uploader for PDF — user uploads a file here
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# A helper function to extract all the text from the uploaded PDF
# It reads each page and combines the text
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

# If user uploaded a PDF AND we haven’t already processed/stored its text
if uploaded_pdf is not None and "pdf_text" not in st.session_state:
    # Extract text from the uploaded PDF and save it in the session state
    st.session_state.pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("PDF uploaded successfully!")  # Show green success message

# Only show the chat input if PDF has been uploaded
if uploaded_pdf is not None:
    # Show a textbox at the bottom where user can ask a question
    question = st.chat_input("Ask a question about the PDF...")
    
    # If the user types something and hits Enter
    if question:
        # Save the user’s message to the chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # We only take the first 3000 characters of the PDF text (to keep things fast)
        context = st.session_state.pdf_text[:3000]

        # Here, we’d normally use a real LLM to answer — but for now, we just show some sample text
        answer = f"📘 **Answer based on the PDF:**\n\n{context[:300]}..."

        # Save the assistant’s response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Rerun the app so the new messages show up in the chat
        st.rerun()
else:
    # If no PDF yet, show a small prompt to remind user to upload one
    st.caption("⬆️ Upload a PDF to begin.")
