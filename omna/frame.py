"""Public df.omna namespace — all user-facing DataFrame methods live here."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from omna import index
from _omna import top_k_flat, top_k_flat_np  # noqa: F401

_AUDIT_PATH = Path(".omna") / "pii_audit.parquet"


def _default_index_path(column: str) -> Path:
    return Path(".omna") / f"{column}.parquet"


@pl.api.register_dataframe_namespace("omna")
class OmnaFrame:
    """Omna namespace attached to every Polars DataFrame as df.omna."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def embed(self, column: str, index_path: str | Path | None = None) -> pl.DataFrame:
        """Vectorize *column* and persist the embedding index to disk.

        The index is saved as a Parquet file at .omna/{column}.parquet (or
        *index_path* if given). Re-running overwrites the existing index.
        Call this once; search() and filter() read the saved file.

        Args:
            column: Name of the string column to embed.
            index_path: Override the default save location.

        Returns:
            The original DataFrame unchanged.
        """
        from omna import embedder
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        texts = self._df[column].cast(pl.String).to_list()
        vectors = embedder.embed(texts)
        path = Path(index_path) if index_path else _default_index_path(column)
        index.save(self._df, vectors, path)
        return self._df

    def search(self, query: str, on: str, k: int = 10,
               index_path: str | Path | None = None) -> pl.DataFrame:
        """Return the *k* rows most semantically similar to *query*.

        Requires df.omna.embed(on) to have been called first.

        Args:
            query: Natural-language search string.
            on: Column name that was previously embedded.
            k: Number of results (default 10).
            index_path: Override the default index location.

        Returns:
            DataFrame of the top k matching rows plus a '_score' column (0-1).
        """
        from omna import embedder
        path = Path(index_path) if index_path else _default_index_path(on)
        if not path.exists():
            raise FileNotFoundError(
                f"No index for column '{on}'. Run df.omna.embed('{on}') first."
            )
        df, embeddings = index.load(path)
        query_vec = np.array(embedder.embed([query])[0], dtype=np.float32)
        dim = embeddings.shape[1]
        flat_emb = np.ascontiguousarray(embeddings)
        hits = top_k_flat_np(query_vec, flat_emb, dim, k)
        if not hits:
            return df.clear()
        result = df[list(h[0] for h in hits)].with_columns(
            pl.Series("_score", [h[1] for h in hits], dtype=pl.Float32)
        )
        return result

    def filter(self, concept: str, on: str, threshold: float = 0.3,
               index_path: str | Path | None = None) -> pl.DataFrame:
        """Keep rows whose *on* column semantically matches *concept*.

        Requires df.omna.embed(on) to have been called first.

        Args:
            concept: Concept or phrase to match against.
            on: Column name that was previously embedded.
            threshold: Minimum cosine similarity score to keep a row (default 0.3).
            index_path: Override the default index location.

        Returns:
            Filtered DataFrame sorted by similarity, no score column.
        """
        from omna import embedder
        path = Path(index_path) if index_path else _default_index_path(on)
        if not path.exists():
            raise FileNotFoundError(
                f"No index for column '{on}'. Run df.omna.embed('{on}') first."
            )
        df, embeddings = index.load(path)
        concept_vec = np.array(embedder.embed([concept])[0], dtype=np.float32)
        dim = embeddings.shape[1]
        flat_emb = np.ascontiguousarray(embeddings)
        hits = top_k_flat_np(concept_vec, flat_emb, dim, len(df))
        above = [h for h in hits if h[1] >= threshold]
        if not above:
            return df.clear()
        return df[[h[0] for h in above]]

    def mask_pii(self, audit_path: str | Path | None = None) -> pl.DataFrame:
        """Redact PII in all string columns and save an audit log to disk.

        Args:
            audit_path: Override the default audit log location.

        Returns:
            New DataFrame with PII redacted. The original is not modified.
        """
        from omna import pii
        masked_df, audit_df = pii.mask(self._df)
        log_path = Path(audit_path) if audit_path else _AUDIT_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)
        audit_df.write_parquet(log_path)
        return masked_df

    def pii_report(self) -> pl.DataFrame:
        """Scan all string columns and return a PII findings report.

        Returns:
            DataFrame of PII findings. Empty (same schema) when none found.
        """
        from omna import pii
        return pii.report(self._df)

    def ask(self, question: str, model: str | None = None) -> str:
        """Answer a natural-language question about this DataFrame using Claude.

        Args:
            question: Any natural-language question about the data.
            model: Claude model ID. Defaults to claude-haiku-4-5-20251001.

        Returns:
            Claude's answer as a string.
        """
        from omna import ask as ask_mod
        kwargs = {"model": model} if model else {}
        return ask_mod.query(self._df, question, **kwargs)
