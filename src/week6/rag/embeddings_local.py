from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(text: str) -> list[float]:
    """Embed one string. Returns 384 floats."""
    return _get_model().encode(text, normalize_embeddings=True).tolist()
