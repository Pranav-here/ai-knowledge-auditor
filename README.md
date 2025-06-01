# 🧠 AI Knowledge Auditor

**Trust & Traceability for LLM Answers**

## 🚀 Overview

**AI Knowledge Auditor** is a Streamlit-based tool that verifies how *truthful and grounded* a large language model’s answer is, using uploaded domain-specific documents. It runs an audit pipeline that retrieves evidence from your sources, compares them with the LLM’s output, and generates a **trust score**—highlighting mismatches and potential hallucinations.

This isn't just another chatbot. It's an **LLM reliability tool** for high-stakes fields like **medicine, law, policy, and education**.

## 🎯 Problem Statement

Modern LLMs (e.g., GPT, Mistral) are powerful but can **hallucinate**—confidently stating false or unverified information. This makes them risky to trust in scenarios where factual accuracy matters.

This project tackles a key question:  
> “How can we **audit** an LLM’s response to ensure it's *based on retrieved facts*, not fiction?”

## 🔍 What This Tool Does

- Accepts **PDF documents** as ground truth (e.g., medical guidelines, legal papers)
- Lets users **ask questions** based on the uploaded files
- Uses a **RAG (Retrieval-Augmented Generation)** pipeline to fetch context from documents
- Compares the **LLM-generated answer** against retrieved source chunks
- Computes a **Trust Score** (0–100) and flags hallucinated or off-topic content
- Shows a clean UI: **Question → Answer → Evidence → Trust Score → Highlights**

## 🛠 Tech Stack

| Layer        | Tool/Lib                    |
|--------------|-----------------------------|
| Frontend     | Streamlit                   |
| LLM          | Mistral-7B (HuggingFace)    |
| Retrieval    | LangChain + FAISS           |
| Embeddings   | Sentence Transformers       |
| Comparison   | Cosine similarity + ROUGE   |
| Optional     | LangChain Tracing, LLMonitor|

## 💡 Key Features

- 🔍 Upload & chunk PDFs into FAISS vector store
- 💬 Ask natural language questions grounded in those docs
- 🤖 LLM-generated answers with **trust analysis**
- 🧠 Highlight hallucinated segments and off-topic fluff
- 📄 Exportable PDF audit report (coming soon)

## 🎯 Use Cases

- 🩺 Verifying AI medical advice
- ⚖️ Auditing LLM-generated legal summaries
- 📚 Educational tutors with real-time fact checks
- 📊 Traceability layer for any LLM app

## 📦 Future Roadmap

- [ ] Source citations + click-to-verify chunks  
- [ ] Support for multi-model auditing (GPT vs Mistral)  
- [ ] Domain-specific fine-tuning (e.g., healthcare)  
- [ ] API access for external apps  
- [ ] Chat history with trust evolution tracking  

## 📌 Status

> Alpha build – core RAG + trust scoring system in development. UI scaffold complete.  
Targeting full MVP by mid-July 2025.

---

**Demo, docs, and tutorials coming soon.**  
Built by [Pranav Kuchibhotla](https://pranavkuchibhotla.com)
