# Dockerfile
# -----------
# Builds a container image for the DocChat FastAPI application.
#
# This image contains: Python, our dependencies, our code, and the
# pre-built Chroma vector database. It does NOT contain Ollama -
# the LLM runs separately and this container connects to it over
# the network (see OLLAMA_HOST env var).

# Base image: official Python 3.11 slim image.
# "slim" = minimal Debian-based image with just enough to run Python -
# smaller and faster to build than the full python:3.11 image.
FROM python:3.11-slim

# Set the working directory INSIDE the container.
# All subsequent commands (COPY, RUN, CMD) run relative to this path.
WORKDIR /app

# Copy only requirements.txt first, then install.
# WHY separately from copying all code: Docker caches each step (layer).
# If only our Python code changes (not requirements.txt), Docker reuses
# the cached "pip install" layer instead of re-running it - much faster
# rebuilds during development.
COPY requirements.txt .

# Install CPU-only PyTorch FIRST, from PyTorch's dedicated CPU index.
# WHY: the default PyPI version of torch bundles full NVIDIA/CUDA GPU
# support (~400MB+), which we don't need (no GPU here, and our Mac is
# Apple Silicon anyway). The CPU-only build is ~100-150MB - much faster
# and more reliable to download. Once torch is installed, the next step
# (installing sentence-transformers, which depends on torch) will see
# this requirement is already satisfied and skip re-downloading it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of our dependencies.
# --default-timeout=120 increases pip's per-chunk download timeout
# (default ~15s) to handle slower/unstable connections better.
RUN pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# Pre-download the embedding model into the image at build time.
# This means the model is baked into the image itself - no network
# needed at runtime. Cached to the default HuggingFace cache location.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Now copy the rest of the application code and data.
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

# Document that the container listens on port 8000.
# This is informational (doesn't actually publish the port - that
# happens with `docker run -p`).
EXPOSE 8000

# The command that runs when the container starts.
# --host 0.0.0.0 is REQUIRED in containers: it tells uvicorn to accept
# connections from outside the container, not just "localhost" (which
# inside a container would only mean the container itself).
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]