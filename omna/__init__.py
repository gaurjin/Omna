"""Omna — semantic search, PII masking and schema understanding for Polars."""
from __future__ import annotations

import polars as pl

from omna.frame import OmnaFrame  # noqa: F401 — registers df.omna namespace
from omna import understand as _understand_mod


def understand(df: pl.DataFrame) -> pl.DataFrame:
    """Infer schema and label each column in *df*.

    Returns a one-row-per-column DataFrame with:
    column, dtype, null_pct, unique_count, label, sample.

    Labels: email, phone, name, id, date, text, numeric, boolean, category, unknown.
    No LLM call — runs fully offline.

    Args:
        df: Any Polars DataFrame.
    """
    return _understand_mod.describe(df)


__all__ = ["OmnaFrame", "understand"]
