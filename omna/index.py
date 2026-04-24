"""omna.index — save and load embedding indexes as Parquet files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

EMBEDDING_COL = "_omna_embedding"

# In-memory cache: path string → (df, numpy array)
_cache: dict[str, tuple[pl.DataFrame, np.ndarray]] = {}


def save(df: pl.DataFrame, embeddings: list[list[float]], path: str | Path) -> None:
    """Persist *df* together with *embeddings* to a Parquet file at *path*.

    Each row in *df* must correspond to exactly one vector in *embeddings*.
    The vectors are stored in a column named '_omna_embedding' as List[Float32].
    Existing files at *path* are overwritten.

    Args:
        df: The source DataFrame (any columns).
        embeddings: One float vector per row of *df*.
        path: Destination file path (will be created or overwritten).
    """
    if len(df) != len(embeddings):
        raise ValueError(
            f"df has {len(df)} rows but embeddings has {len(embeddings)} entries"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed = df.with_columns(
        pl.Series(name=EMBEDDING_COL, values=embeddings, dtype=pl.List(pl.Float32))
    )
    indexed.write_parquet(path)
    # Clear cache for this path so next load picks up fresh data
    _cache.pop(str(path), None)


def load(path: str | Path) -> tuple[pl.DataFrame, np.ndarray]:
    """Load a Parquet file written by :func:`save`.

    First call reads from disk and caches in memory.
    Every subsequent call returns the cached version instantly.

    Returns:
        A tuple of (df, embeddings) where *df* is the original DataFrame
        (without the embedding column) and *embeddings* is a numpy float32 array
        of shape (n_rows, embedding_dim).
    """
    path = Path(path)
    key = str(path)

    if key in _cache:
        return _cache[key]

    if not path.exists():
        raise FileNotFoundError(f"No index found at {path}")

    full = pl.read_parquet(path)
    embeddings = np.array(full[EMBEDDING_COL].to_list(), dtype=np.float32)
    df = full.drop(EMBEDDING_COL)

    _cache[key] = (df, embeddings)
    return df, embeddings
