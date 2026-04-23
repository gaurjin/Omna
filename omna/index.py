"""omna.index — save and load embedding indexes as Parquet files."""
from __future__ import annotations

from pathlib import Path

import polars as pl

EMBEDDING_COL = "_omna_embedding"


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


def load(path: str | Path) -> tuple[pl.DataFrame, list[list[float]]]:
    """Load a Parquet file written by :func:`save`.

    Returns:
        A tuple of (df, embeddings) where *df* is the original DataFrame
        (without the embedding column) and *embeddings* is a list of float vectors.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No index found at {path}")
    full = pl.read_parquet(path)
    embeddings: list[list[float]] = full[EMBEDDING_COL].to_list()
    df = full.drop(EMBEDDING_COL)
    return df, embeddings
