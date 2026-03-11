FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download embedding model at build time (optional, saves startup time)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Default data setup
RUN mkdir -p data/raw data/processed data/samples .llm_cache

EXPOSE 8000

ENV LLM_PROVIDER=anthropic
ENV EMBEDDING_PROVIDER=sentence_transformers
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV VECTOR_STORE=faiss
ENV DEBUG_MODE=false

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
