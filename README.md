# AI Knowledge Auditor

AI Knowledge Auditor is a lightweight Streamlit application that checks how well an AI‑generated answer aligns with a source PDF.  
It combines dense retrieval, cross‑encoder re‑ranking, and a simple heuristic blend to produce a confidence score and highlights the supporting passage.

Live Demo: Will be out soon!

## Features

* Drag‑and‑drop PDF upload  
* Automatic text extraction and adaptive chunking  
* FAISS vector index built with `all‑mpnet‑base‑v2` embeddings  
* Cross‑encoder re‑ranking (`ms‑marco‑MiniLM‑L‑6‑v2`) for sharper relevance judgment  
* Sentence‑level highlighting of the best supporting lines  
* Optional passage summarisation (DistilBART CNN‑12‑6)  
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
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (first run only)

```python
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
│   ├── __init__.py
│   ├── loader.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── auditor.py
├── data/
│   └── faiss_index/        # auto‑generated
├── requirements.txt
└── README.md
```

## Tips

* Large PDFs may take time to embed; progress is displayed in the terminal.  
* The Trust Score is heuristic. Adjust the weighting blend in `core/vector_store.py` for different behaviour.  
* If `faiss-cpu` fails to install on Apple Silicon or Windows, see the FAISS documentation for pre‑built wheels.

## License

This project is released under the MIT License.
