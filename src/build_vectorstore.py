"""
build_vectorstore.py
---------------------
Step 3 of the RAG pipeline: Build the persistent Vector Store.

What this script does:
1. Loads PDFs and splits into chunks (reusing ingest.py)
2. Loads the embedding model (reusing embed.py)
3. Embeds all chunks
4. Stores chunks + embeddings + metadata in a Chroma vector database,
   persisted to disk in ./chroma_db/

Why a persistent vector store:
- Embedding 220 chunks took only ~2 seconds here, but real projects can
  have thousands/millions of chunks taking hours.
- We do this expensive work ONCE (offline/batch), then the query side
  just loads the saved index instantly.
- This mirrors real production architecture: an "indexing pipeline"
  (run periodically when documents change) separate from the
  "serving/query pipeline" (run per user request, must be fast).

Run this script whenever:
- You add new PDFs to data/raw/
- You change CHUNK_SIZE/CHUNK_OVERLAP in ingest.py
"""

import chromadb
from chromadb.utils import embedding_functions

from .ingest import load_pdfs, split_documents, RAW_DATA_DIR
from .embed import EMBEDDING_MODEL_NAME


# Where Chroma will store its database files on disk.
# This folder will contain SQLite files + index data - it's how the
# "database" persists between script runs.
import os
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

# Name of the "collection" - think of this like a table name in SQL.
# A single Chroma database can hold multiple collections.
COLLECTION_NAME = "research_papers"


def get_embedding_function():
    """
    Chroma needs to know HOW to convert text into vectors whenever
    we add or query documents. Instead of manually calling
    sentence-transformers ourselves, we wrap it in Chroma's
    SentenceTransformerEmbeddingFunction.

    This means: whenever we call collection.add(documents=[...]) or
    collection.query(query_texts=[...]), Chroma automatically runs
    the text through this same model for us.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def build_vectorstore():
    print("=== Step 1: Loading and chunking documents ===")
    documents = load_pdfs(RAW_DATA_DIR)
    chunks = split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    print("\n=== Step 2: Connecting to Chroma (persistent) ===")
    # PersistentClient saves data to disk at CHROMA_DB_DIR.
    # If this folder already has data, it loads the existing database.
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    embedding_fn = get_embedding_function()

    # get_or_create_collection: if "research_papers" already exists
    # (from a previous run), reuse it. Otherwise create it fresh.
    # We delete and recreate it here to avoid duplicate entries when
    # re-running this script during development.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' (rebuilding fresh)")
    except Exception:
        pass  # collection didn't exist yet, that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print("\n=== Step 3: Adding chunks to the vector store ===")
    print("(Chroma will embed each chunk automatically using the embedding function)")

    # Chroma's add() requires:
    # - documents: list of raw text strings
    # - metadatas: list of dicts (one per document) - extra info for filtering/display
    # - ids: list of unique string IDs (one per document) - required, like a primary key
    documents_text = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", -1),
        }
        for chunk in chunks
    ]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Chroma can be slow if you add thousands at once without batching,
    # but 220 is small enough to add in one call.
    collection.add(
        documents=documents_text,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"\nDone! Collection '{COLLECTION_NAME}' now contains {collection.count()} chunks")
    print(f"Stored persistently at: ./{CHROMA_DB_DIR}/")


if __name__ == "__main__":
    build_vectorstore()