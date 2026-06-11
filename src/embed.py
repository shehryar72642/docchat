"""
embed.py
--------
Step 2 of the RAG pipeline: Generate Embeddings.

What this script does:
1. Re-runs ingestion (load PDFs -> split into chunks)
2. Loads a local embedding model (all-MiniLM-L6-v2)
3. Converts each text chunk into a 384-dimensional vector
4. Prints stats and a similarity demo so we can SEE embeddings working

Why we need this step:
- Embeddings turn text into numbers that capture MEANING, not just words.
- Two chunks discussing the same concept in different words will have
  vectors that are close together (high cosine similarity).
- This is what lets us do "semantic search" later: find relevant chunks
  for a user's question even if they don't share exact keywords.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import load_pdfs, split_documents, RAW_DATA_DIR

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Model name on Hugging Face Hub. The first time this runs, it will
# download the model (~80MB) and cache it locally in ~/.cache/huggingface
# Subsequent runs reuse the cached copy - no re-download.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the sentence-transformer model.

    SentenceTransformer wraps a transformer neural network (built on
    PyTorch) plus a "pooling" layer that compresses the network's
    per-word outputs into a single fixed-size vector per input text.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def generate_embeddings(model, texts):
    """
    Convert a list of text strings into a list of vectors.

    model.encode():
    - Tokenizes each text (splits into sub-word pieces the model understands)
    - Runs them through the transformer network
    - Pools the output into one fixed-size vector (384 numbers here)

    Returns a NumPy array of shape (num_texts, 384).
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,   # visual progress bar since 220 chunks takes a moment
        convert_to_numpy=True,    # return as NumPy array (easier to do math on)
    )
    return embeddings


def demo_similarity_search(model, chunks, embeddings):
    """
    Demonstrate semantic search WITHOUT a vector database (yet).

    We take a sample question, embed it the same way we embedded the
    chunks, then compute cosine similarity between the question vector
    and EVERY chunk vector. Cosine similarity ranges from -1 to 1:
    - 1.0  = identical direction (very similar meaning)
    - 0.0  = unrelated
    - -1.0 = opposite meaning

    We then show the top 3 most similar chunks - this is the core
    operation a vector database performs at scale.
    """
    query = "How does the attention mechanism work in transformers?"
    print(f"\n=== Semantic Search Demo ===")
    print(f"Query: {query}\n")

    # Embed the query using the SAME model -> same 384-dim space
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Compute similarity between query vector and all 220 chunk vectors
    # Result shape: (1, 220) -> one row, 220 similarity scores
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get indices of the top 3 highest similarity scores, descending order
    top_indices = similarities.argsort()[::-1][:3]

    for rank, idx in enumerate(top_indices, start=1):
        score = similarities[idx]
        chunk = chunks[idx]
        print(f"--- Rank {rank} (score={score:.4f}) ---")
        print(f"Source: {chunk.metadata.get('source')}, Page: {chunk.metadata.get('page')}")
        print(chunk.page_content[:200])
        print()


def main():
    print("=== Re-running ingestion ===")
    documents = load_pdfs(RAW_DATA_DIR)
    chunks = split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    print("\n=== Loading embedding model ===")
    model = load_embedding_model()

    print("\n=== Generating embeddings ===")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = generate_embeddings(model, texts)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"  -> {embeddings.shape[0]} chunks, each represented as a {embeddings.shape[1]}-dimensional vector")

    # Show a tiny preview of one embedding vector
    print(f"\nFirst 10 numbers of chunk #10's embedding:")
    print(embeddings[10][:10])

    demo_similarity_search(model, chunks, embeddings)


if __name__ == "__main__":
    main()