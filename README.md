# DocChat — RAG-based Q&A Over Research Papers

A Retrieval-Augmented Generation (RAG) system that answers questions about
research papers (Attention Is All You Need, BERT, RAG) using a local LLM.
Built entirely with free, open-source, locally-run components — no API keys required.

## What it does

Ask a question like *"How does multi-head attention work?"* and the system:
1. Searches a vector database of paper excerpts for relevant passages
2. Feeds those passages + your question to a local LLM
3. Returns a grounded answer with source citations (paper + page number)

## Architecture
PDFs → Chunking → Embeddings → Chroma (Vector DB)
↓
User Question → Embed → Retrieve top-k chunks
↓
Prompt (context + question) → LLM (Ollama/phi3) → Answer
↓
FastAPI REST endpoint

## Tech Stack

- **Language**: Python 3.11
- **Document processing**: PyPDF, LangChain text splitters
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`, runs locally)
- **Vector database**: ChromaDB (persistent, local)
- **LLM**: Ollama running `phi3:mini` (local, no API costs)
- **API**: FastAPI + Uvicorn
- **Containerization**: Docker

## Project Structure
docchat/
├── data/raw/              # Source PDFs
├── chroma_db/              # Persisted vector database
├── src/
│   ├── ingest.py            # Load & chunk PDFs
│   ├── embed.py              # Embedding model wrapper
│   ├── build_vectorstore.py  # Build/rebuild the vector DB
│   ├── query_vectorstore.py  # Semantic search over the DB
│   ├── rag_pipeline.py       # Full retrieval + generation pipeline
│   └── main.py                # FastAPI application
├── requirements.txt
└── Dockerfile

## Running Locally (without Docker)

1. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

2. Install [Ollama](https://ollama.com) and pull the model:
```bash
   ollama pull phi3:mini
```

3. Build the vector database (one-time, or after adding new PDFs):
```bash
   python src/build_vectorstore.py
```

4. Start the API:
```bash
   cd src
   uvicorn main:app --reload
```

5. Query it:
```bash
   curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is multi-head attention?"}'
```

   Or visit `http://127.0.0.1:8000/docs` for interactive API docs.

## Running with Docker

1. Make sure Ollama is running on your host machine with `phi3:mini` pulled.

2. Build the image:
```bash
   docker build -t docchat:latest .
```

3. Run the container:
```bash
   docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 docchat:latest
```

4. Query it the same way as above.

## Example

**Request:**
```json
{"question": "What pretraining tasks does BERT use?"}
```

**Response:**
```json
{
  "question": "What pretraining tasks does BERT use?",
  "answer": "BERT is pretrained using a Masked Language Model (MLM) objective and Next Sentence Prediction (NSP)...",
  "sources": [
    {"source": "data/raw/bert.pdf", "page": 5},
    {"source": "data/raw/bert.pdf", "page": 13}
  ]
}
```

## Adding Your Own Documents

1. Drop PDF files into `data/raw/`
2. Re-run `python src/build_vectorstore.py` to rebuild the index
3. Restart the API

## Key Design Decisions

- **Local-first**: every component (embeddings, LLM, vector DB) runs locally
  with no external API dependencies, making this free to run and easy to demo.
- **Separation of indexing and serving**: building the vector store (slow,
  run occasionally) is decoupled from querying (fast, run per-request) —
  mirroring real production RAG architectures.
- **Source attribution**: every answer includes the source document and
  page number, improving trust and verifiability.

## Possible Improvements

- Swap `phi3:mini` for a larger model (e.g., Llama 3 8B) for better answer quality
- Add a simple web UI (Streamlit/React)
- Support more document formats (Word, HTML, Markdown)
- Add evaluation metrics (retrieval precision, answer relevance)
- Deploy to a cloud platform with a hosted LLM API for production use