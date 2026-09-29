EMBED_MODEL = "text-embedding-3-small"

_client_instance = None


def _openai():
    """Lazy-init OpenAI client so imports don't require the key."""
    global _client_instance
    if _client_instance is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The 'openai' package is required for embedding. Run 'uv sync' first."
            ) from error
        _client_instance = OpenAI()
    return _client_instance


def embed_batch(texts: list[str]) -> list[list[float]]:
    """One API call, list of vectors back."""
    resp = _openai().embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
