"""
main.py
-------
Step 6 of the RAG pipeline: API Layer (FastAPI).

What this script does:
- Defines a web API with one main endpoint: POST /ask
- Wraps our existing rag_pipeline.answer_question() function
- Validates incoming requests and outgoing responses using Pydantic models
- Provides automatic interactive API docs at /docs

Why an API matters:
- Our RAG logic (rag_pipeline.py) is currently only callable from Python.
- An API exposes it over HTTP, so ANY client - a web frontend, mobile app,
  curl command, or another service - can use it, regardless of language.
- This is the standard way ML/AI functionality gets integrated into
  real products.

How to run this:
    uvicorn main:app --reload
(run from inside the src/ directory, or adjust the module path)
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import answer_question


# Create the FastAPI application instance.
# This object is what Uvicorn runs, and where we attach our routes.
app = FastAPI(
    title="DocChat API",
    description="A RAG-based Q&A API over research papers (Attention, BERT, RAG)",
    version="0.1.0",
)


class AskRequest(BaseModel):
    """
    Defines the expected JSON body for POST /ask requests.

    Pydantic automatically validates incoming JSON against this schema:
    - If 'question' is missing or not a string, FastAPI returns a
      422 error automatically, before our code even runs.

    Example valid request body:
        {"question": "What is BERT?"}
    """
    question: str


class Source(BaseModel):
    """Represents one source document/page used to answer a question."""
    source: str
    page: int


class AskResponse(BaseModel):
    """
    Defines the JSON shape we return from POST /ask.

    Declaring this explicitly (rather than returning a raw dict) means:
    - FastAPI validates our OWN output matches this shape (catches bugs)
    - The /docs page shows clients exactly what to expect back
    """
    question: str
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    """
    Simple health-check endpoint.
    Useful for verifying the API is running, and for deployment
    platforms that ping a URL to check if the service is alive.
    """
    return {"status": "ok", "message": "DocChat API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Main endpoint: accepts a question, runs the full RAG pipeline,
    and returns the generated answer with its sources.

    FastAPI automatically:
    1. Parses the incoming JSON into an AskRequest object
    2. Calls this function with `request` populated
    3. Validates our return value matches AskResponse
    4. Serializes the result back to JSON
    """
    result = answer_question(request.question)
    return result