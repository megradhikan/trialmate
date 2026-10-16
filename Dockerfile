FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download embedding model at build time (saves startup time)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Build the FAISS clause index at build time so cold starts are fast
RUN python -m matching.embedder --trials data/processed/trials.json --output data/processed/faiss_index

# Default data setup
RUN mkdir -p data/raw data/processed data/samples .llm_cache

EXPOSE 8000

# mock = rule-based rationales, no API key needed. Set to anthropic/openai + key for LLM explanations.
ENV LLM_PROVIDER=mock
ENV EMBEDDING_PROVIDER=sentence_transformers
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2

# Render/Heroku-style platforms inject $PORT
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
