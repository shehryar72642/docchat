"""
rag_pipeline.py
----------------
Step 5 of the RAG pipeline: Retrieval-Augmented Generation (full pipeline).

What this script does:
1. Takes a user question
2. Retrieves relevant chunks from Chroma (retrieval - already built)
3. Constructs a prompt combining the question + retrieved context
4. Sends the prompt to a local LLM (phi3:mini via Ollama) to generate
   an answer (generation - new in this step)
5. Prints the answer along with the sources used

This is the complete "RAG" pattern: Retrieval-Augmented Generation.
The LLM's answer is "grounded" in our documents rather than relying
purely on what it memorized during training.
"""

import os
import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)

from .query_vectorstore import query_vectorstore


# The LLM model we pulled via `ollama pull phi3:mini`
LLM_MODEL = "phi3:mini"

# How many chunks to retrieve as context for each question.
# More context = more information for the LLM, but also more tokens
# to process (slower) and risk of including irrelevant info.
N_RESULTS = 3


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Construct the prompt sent to the LLM.

    This is "prompt engineering" - how we phrase this instruction
    significantly affects answer quality. Key elements:
    - Clear instruction to use ONLY the provided context
    - Instruction to say "I don't know" if the answer isn't in context
      (this reduces hallucination)
    - The context itself, clearly delimited
    - The actual question at the end
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""You are a helpful research assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say "I don't have enough information in the provided documents to answer this."

Context:
{context_text}

Question: {question}

Answer:"""

    return prompt


def retrieve_chunks(question: str, n_results: int = N_RESULTS) -> list[dict]:
    """
    Wrapper around query_vectorstore() that reshapes Chroma's output
    into a simpler list of dicts - easier to work with when building
    the prompt.
    """
    results = query_vectorstore(question, n_results=n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "distance": dist,
        })

    return chunks


def generate_answer(prompt: str) -> str:
    """
    Send the prompt to the local LLM via Ollama and return its response.

    ollama.chat():
    - Connects to the Ollama server running in the background
      (started automatically when you opened the Ollama app)
    - 'messages' follows the same format as OpenAI's chat API:
      a list of {"role": ..., "content": ...} dicts
    - 'role': 'user' means this message is from the user/application
      (as opposed to 'system' or 'assistant')

    Returns just the text content of the model's reply.
    """
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response["message"]["content"]


def answer_question(question: str) -> dict:
    """
    Full RAG pipeline for a single question.

    Returns a dict with the answer and the sources used, so the
    caller (e.g., a future API endpoint) can display both.
    """
    print(f"Question: {question}")

    print("Retrieving relevant chunks...")
    chunks = retrieve_chunks(question)

    print("Building prompt...")
    prompt = build_prompt(question, chunks)

    print("Generating answer...")
    answer = generate_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": c["source"], "page": c["page"]}
            for c in chunks
        ],
    }


def main():
    test_questions = [
        "What is the main idea of the attention mechanism?",
        "What pretraining tasks does BERT use?",
    ]

    for question in test_questions:
        result = answer_question(question)

        print("\n" + "=" * 60)
        print(f"ANSWER:\n{result['answer']}")
        print("\nSOURCES:")
        for src in result["sources"]:
            print(f"  - {src['source']} (page {src['page']})")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()