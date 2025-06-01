import streamlit as st

st.title("AI Knowledge Auditor")
st.write("Upload a pdf and ask a question related to its content")

uploaded_files = st.file_uploader("**Upload PDF**", type=['pdf'])

st.chat_input("Ask your question here")

if uploaded_files is not None:
    pass
