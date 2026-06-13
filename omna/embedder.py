"""omna.embedder — FastEmbed wrapper with hardware-aware acceleration."""
from __future__ import annotations

import platform

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def _maybe_prefix(texts: list[str], kind: str, model_name: str) -> list[str]:
    """Apply nomic's task-instruction prefixes (it was trained to require them:
    ``search_document:`` on indexed text, ``search_query:`` on the query).
    Other models (e.g. bge) don't use prefixes, so this is a no-op for them.
    `kind` is "document" (default, index build) or "query" (search/filter)."""
    if "nomic" in model_name.lower():
        prefix = "search_query: " if kind == "query" else "search_document: "
        return [prefix + t for t in texts]
    return texts

# Public cache dict — keyed by model name (preserves existing test surface).
_cache: dict = {}

_model_instance = None
_model_name_cached: str | None = None


def get_best_providers() -> list[str]:
    """Dynamically detect the best hardware accelerator available."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_embedding_model(model_name: str = DEFAULT_MODEL):
    """Create a TextEmbedding model with the best available providers."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise ImportError(
            "fastembed is required for embed(), search() and filter(). "
            'Install it with:  pip install "omna[embed]"'
        ) from None

    providers = get_best_providers()
    is_cpu_only = providers[0] == "CPUExecutionProvider"
    model = TextEmbedding(
        model_name=model_name,
        providers=providers,
        # parallel=0 (max workers) only for CPU;
        # None (single process) for CoreML/CUDA to prevent crashing.
        parallel=0 if is_cpu_only else None,
    )
    print("Warming up CoreML... (first run only, ~30s)")
    list(model.embed(["warmup"]))
    print("Ready.")
    return model


def _get_model(model_name: str = DEFAULT_MODEL):
    """Return a cached TextEmbedding instance, reloading only on model change."""
    global _model_instance, _model_name_cached
    if model_name not in _cache:
        _model_instance = create_embedding_model(model_name)
        _model_name_cached = model_name
        _cache[model_name] = _model_instance
    else:
        _model_instance = _cache[model_name]
        _model_name_cached = model_name
    return _cache[model_name]


def embed_texts(texts: list[str], batch_size: int = 32, chunk_size: int = 2_000,
                kind: str = "document"):
    """Embed texts in chunks to prevent CoreML/GPU from being overwhelmed.

    Args:
        texts: Strings to embed.
        batch_size: Internal ONNX batch size passed to FastEmbed.
        chunk_size: Number of texts per chunk. Default 2 000 keeps peak RAM under control.
        kind: "document" (default — index build) or "query"; controls the
            nomic task prefix (no-op for non-nomic models).

    Returns:
        List of raw numpy vectors, one per input text.
    """
    import gc
    import math

    model = _get_model()
    texts = _maybe_prefix(texts, kind, DEFAULT_MODEL)

    if len(texts) <= chunk_size:
        vectors = list(model.embed(texts, batch_size=batch_size))
        gc.collect()
        return vectors

    total_chunks = math.ceil(len(texts) / chunk_size)
    all_vectors: list = []
    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(texts))
        print(f"Embedding chunk {i + 1}/{total_chunks}...", flush=True)
        all_vectors.extend(model.embed(texts[start:end], batch_size=batch_size))
        gc.collect()
    return all_vectors


def embed(texts: list[str], model_name: str = DEFAULT_MODEL,
          kind: str = "document") -> list[list[float]]:
    """Embed a list of texts and return a list of float vectors.

    Uses FastEmbed locally — no API key required. The model downloads once on
    first use and is cached on disk by FastEmbed.

    Args:
        texts: Strings to embed.
        model_name: Any model name supported by FastEmbed.
        kind: "document" (default) or "query" — controls the nomic task prefix.
            search()/filter() pass "query"; index build uses "document".

    Returns:
        List of vectors, one per input text. Vector length depends on the model
        (768 for the default nomic-embed-text-v1.5).
    """
    model = _get_model(model_name)
    texts = _maybe_prefix(texts, kind, model_name)
    return [vec.tolist() for vec in model.embed(texts, batch_size=512)]


def embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """Return the vector dimension for *model_name*."""
    return len(embed(["probe"], model_name=model_name)[0])
