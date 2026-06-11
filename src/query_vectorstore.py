"""
query_vectorstore.py
---------------------
Step 4 of the RAG pipeline: Query the Vector Store.

What this script does:
1. Connects to the EXISTING Chroma database on disk (no re-embedding
   of documents - that was already done by build_vectorstore.py)
2. Takes a user question
3. Embeds the question and searches for the most similar chunks
4. Prints results with source/page metadata

This is the "retrieval" half of "Retrieval-Augmented Generation".
In Step 7, we'll take these retrieved chunks and feed them to an LLM
to generate a final answer.
"""

import chromadb

from build_vectorstore import CHROMA_DB_DIR, COLLECTION_NAME, get_embedding_function


def query_vectorstore(question: str, n_results: int = 3):
    """
    Search the vector store for chunks most relevant to `question`.

    Args:
        question: the user's natural language question
        n_results: how many top chunks to retrieve

    Returns:
        Chroma's query result dict, containing documents, metadatas,
        and distances (lower distance = more similar).
    """
    # Connect to the SAME persistent database created by build_vectorstore.py
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    embedding_fn = get_embedding_function()

    # get_collection (not create) - we expect it to already exist.
    # If it doesn't, this raises an error reminding us to run
    # build_vectorstore.py first.
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # collection.query():
    # - Embeds `question` using the same embedding function
    # - Searches Chroma's internal index for the n_results closest vectors
    # - Returns matching documents + their metadata + similarity distances
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    return results


def print_results(question: str, results: dict):
    print(f"Query: {question}\n")

    # results structure: each value is a list-of-lists because Chroma
    # supports batch queries (multiple questions at once). We only sent
    # one question, so we index [0] to get that question's results.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        print(f"--- Rank {rank} (distance={dist:.4f}) ---")
        print(f"Source: {meta['source']}, Page: {meta['page']}")
        print(doc[:250])
        print()


if __name__ == "__main__":
    # A few test questions to try
    test_questions = [
        "How does the attention mechanism work in transformers?",
        "What datasets were used to evaluate BERT?",
        "How is the retriever trained in RAG?",
    ]

    for q in test_questions:
        results = query_vectorstore(q, n_results=2)
        print_results(q, results)
        print("=" * 60)