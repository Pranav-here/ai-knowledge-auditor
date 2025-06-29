![Python Version](https://img.shields.io/badge/python-3.8%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

# AI Knowledge Auditor

AI Knowledge Auditor is a lightweight auditing tool for evaluating AI-generated answers against authoritative PDF sources. By combining dense retrieval with FAISS, cross-encoder re-ranking, and sentence-level analysis, it produces a clear confidence score and highlights the most relevant passages.

## Live Demo
A hosted version is available at:

[CHECK IT OUT](https://ai-knowledge-auditor.streamlit.app/)

## Key Features

- **Adaptive PDF Processing**: Automatically extracts text from any PDF, splits content into optimally sized chunks, and caches embeddings by file hash to avoid repeated indexing.
- **Dense Retrieval with FAISS**: Builds a vector index using `sentence-transformers/all-mpnet-base-v2` embeddings to retrieve globally relevant chunks based on combined question-and-answer vectors.
- **Cross-Encoder Re-Ranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to refine initial FAISS results, improving precision by scoring answer–chunk pairs directly.
- **Local Sentence Highlighting**: Computes sentence-level similarity to emphasize the two most relevant sentences within the best chunk.
- **Optional Summarization**: Offers concise summaries of the chosen passage via `sshleifer/distilbart-cnn-12-6` for quick context.
- **Trust Score Breakdown**:
  - **Global Similarity**: Measures broad alignment between question+answer and chunk using FAISS inner product.
  - **Local Similarity**: Captures sentence-level relevance, ensuring key details are present.
  - **Re-Rank Probability**: Cross-encoder score normalized to [0,1], bringing fine-grained semantic judgment.
- **Heuristic Blending**: Combines the three metrics (0.1×global + 0.2×local + 0.7×rereank) into a single percentage score, with labels:
  - Supported (≥75%)
  - Partial Support (40–75%)
  - Likely Hallucinated (<40%)
- **Chat-Style Interface**: Maintains question–answer history with reset capability for iterative auditing sessions.

## Architecture Overview

1. **PDF Loader**: Uses PyMuPDF to extract raw text, then applies adaptive chunking based on document length.
2. **Embedder**: Loads a dual-model pipeline:
   - SentenceTransformer for chunk embeddings
   - CrossEncoder for pairwise relevance re-ranking
3. **Vector Store**: Constructs a FAISS index of normalized chunk embeddings to efficiently retrieve top candidates.
4. **Query Pipeline**:
   - Encode question and answer separately, average their vectors, normalize, and query FAISS.
   - Re-rank top-K chunks with the CrossEncoder to select the single best passage.
   - Compute local sentence similarities for highlighting.
   - Apply optional negation penalty to correct false affirmations.
   - Blend metrics into a unified Trust Score.
5. **User Interface**: Streamlit front end provides file upload, form fields for question and model answer, scoring display, and interactive history.

## Why This Approach Works

- **Global Retrieval** quickly narrows down to broadly relevant sections, ensuring coverage of the overall context.
- **Re-Ranking** refines the initial hits by directly comparing the AI answer to each passage, boosting precision and minimizing irrelevant chunks.
- **Local Similarity** ensures that the final output isn’t just topically related but contains the exact details needed to support or refute the answer.

The combination of these layers delivers a robust yet efficient auditing workflow suitable for documents of any size. Custom weighting in `core/vector_store.py` can be adjusted for different trade-offs between breadth and precision.

## Contribution and Feedback

Contributions are welcome. Please open an issue or pull request for improvements, bug fixes, or feature requests.
