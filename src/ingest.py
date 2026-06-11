"""
ingest.py
---------
Step 1 of the RAG pipeline: Document Ingestion.

What this script does:
1. Loads all PDF files from data/raw/
2. Extracts raw text from each PDF
3. Splits the text into smaller overlapping "chunks"
4. Prints stats so we can sanity-check the output

Why we need this step:
- LLMs and embedding models have a limited input size (context window).
  We can't feed an entire 15-page paper in one go.
- Splitting into chunks lets us later embed and search at a granular level -
  e.g., find the one paragraph that discusses "multi-head attention"
  instead of the whole paper.
- Overlap between chunks prevents losing context at chunk boundaries
  (e.g., a sentence that gets cut in half).
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- Configuration ----
import os
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# CHUNK_SIZE: max number of characters per chunk.
# 1000 chars is roughly 150-250 words - small enough for embedding models,
# large enough to retain meaningful context.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: how many characters from the end of one chunk
# are repeated at the start of the next chunk.
# This ensures a sentence/idea split across two chunks isn't lost entirely
# in either one.
CHUNK_OVERLAP = 150


def load_pdfs(directory: str):
    """
    Load every PDF in `directory` and return a list of LangChain
    'Document' objects.

    Each PDF page becomes its own Document object with:
      - page_content: the extracted text of that page
      - metadata: dict containing info like {'source': filename, 'page': page_number}

    Why metadata matters:
    Later, when the system answers a question, we can tell the user
    *which paper and page* the answer came from - crucial for trust
    and verifiability in real RAG systems.
    """
    all_documents = []

    for filename in os.listdir(directory):
        if not filename.endswith(".pdf"):
            continue  # skip non-PDF files

        filepath = os.path.join(directory, filename)
        print(f"Loading: {filename}")

        # PyPDFLoader reads the PDF and creates one Document per page
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        all_documents.extend(documents)
        print(f"  -> {len(documents)} pages loaded")

    return all_documents


def split_documents(documents):
    """
    Split a list of Document objects into smaller chunks.

    RecursiveCharacterTextSplitter tries to split on natural boundaries
    in this priority order: paragraphs, then lines, then sentences/words,
    then characters.
    This keeps chunks semantically coherent rather than cutting
    mid-word/mid-sentence whenever possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # measure chunk size by character count
    )

    chunks = splitter.split_documents(documents)
    return chunks


def main():
    print("=== Step 1: Loading PDFs ===")
    documents = load_pdfs(RAW_DATA_DIR)
    print(f"\nTotal pages loaded across all PDFs: {len(documents)}")

    print("\n=== Step 2: Splitting into chunks ===")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # ---- Sanity checks: print a sample chunk ----
    print("\n=== Sample chunk (chunk #10) ===")
    sample = chunks[10]
    print(f"Source: {sample.metadata.get('source')}")
    print(f"Page: {sample.metadata.get('page')}")
    print(f"Content length: {len(sample.page_content)} characters")
    print("Content preview:")
    print(sample.page_content[:300])
    print("...")


if __name__ == "__main__":
    main()