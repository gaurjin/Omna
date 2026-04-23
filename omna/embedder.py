"""omna.embedder — FastEmbed wrapper with model caching."""
from __future__ import annotations

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_cache: dict = {}


def _get_model(model_name: str):
    """Return a cached TextEmbedding instance, loading it on first call."""
    if model_name not in _cache:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed is required for embedding. "
                "Install it with:  pip install omna[embed]"
            ) from None
        _cache[model_name] = TextEmbedding(model_name=model_name)
    return _cache[model_name]


def embed(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Embed a list of texts and return a list of float vectors.

    Uses FastEmbed locally — no API key required. The model is downloaded
    once (~130 MB for the default) and cached on disk by FastEmbed.

    Args:
        texts: Strings to embed.
        model_name: Any model name supported by FastEmbed.

    Returns:
        List of vectors, one per input text. Vector length depends on the model
        (384 for the default BAAI/bge-small-en-v1.5).
    """
    model = _get_model(model_name)
    return [vec.tolist() for vec in model.embed(texts)]


def embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """Return the vector dimension for *model_name*."""
    return len(embed(["probe"], model_name=model_name)[0])
