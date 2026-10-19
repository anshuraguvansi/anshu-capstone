# MP2: Mini-RAG

A Sherlock Holmes question-answering pipeline using OpenAI embeddings and Qdrant. I used **uv** to manage Python and dependencies. `pyproject.toml` serves the dependency-list role of `requirements.txt`; `uv.lock` pins the resolved versions.

## Setup

With uv installed, run from the repository root:

```sh
cd mp2
uv sync --python 3.14
```

Create `.env` with your credentials and Qdrant URL:

```dotenv
OPENAI_API_KEY=your-openai-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-key
```

## Run

```sh
uv run --env-file .env python mp2_rag.py ingest    # Build the collection first
uv run --env-file .env python mp2_rag.py ask       # Interactive questions
uv run --env-file .env python mp2_rag.py validate  # Validate all five questions
```
