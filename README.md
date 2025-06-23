# AI Knowledge Auditor

AI Knowledge Auditor is a lightweight Streamlit application that checks how well an AI‑generated answer aligns with a source PDF.  
It combines dense retrieval, cross‑encoder re‑ranking, and a simple heuristic blend to produce a confidence score and highlights the supporting passage.

Live Demo: Will be out soon!

## Features

* Drag‑and‑drop PDF upload  
* File‑hash-based caching to avoid re‑indexing unchanged PDFs  
* Indexing progress spinner for large documents  
* Automatic text extraction and adaptive chunking  
* FAISS vector index built with `all‑mpnet‑base‑v2` embeddings  
* Cross‑encoder re‑ranking (`ms‑marco‑MiniLM‑L‑6‑v2`) for sharper relevance judgment  
* Sentence‑level highlighting of the best supporting lines  
* Optional passage summarisation (DistilBART CNN‑12‑6)  
* Advanced scoring breakdown:  
  - Global similarity  
  - Local similarity  
  - Rerank probability  
* Clear Trust Score labelling: Supported, Partial Support, or Likely Hallucinated  
* Chat‑style history with reset

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/pranav-here/ai‑knowledge‑auditor.git
cd ai‑knowledge‑auditor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (first run only)

```bash
python - <<'PY'
import nltk
nltk.download("punkt")
nltk.download("stopwords")
PY
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, upload a PDF, and start auditing.

## Folder structure

```
.
├── app.py
├── core/
│   ├── loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── auditor.py
├── data/
│   └── faiss_index/        # auto‑generated
├── requirements.txt
└── README.md              # this file
```

## Tips

* Large PDFs may take time to embed; a spinner indicates progress.  
* The Trust Score is heuristic; adjust the weighting blend in `core/vector_store.py` for custom behavior.  
* If `faiss-cpu` fails on Apple Silicon or Windows, check FAISS documentation for compatible wheels.

## License

This project is released under the MIT License.
