"""Omna — semantic search, PII masking and schema understanding for Polars."""
from __future__ import annotations

__version__ = "0.1.0"

import polars as pl

from omna.frame import OmnaFrame  # noqa: F401 — registers df.omna namespace
from omna import understand as _understand_mod


def understand(df: pl.DataFrame) -> None:
    """Print a rich schema summary for *df* and return None.

    Infers a semantic label for each column (email, phone, name, id, date,
    text, numeric, boolean, category, unknown) without any LLM call.
    The table is printed to stdout via rich. Nothing is returned — use
    ``understand_df`` when you need the raw result DataFrame.

    Args:
        df: Any Polars DataFrame.
    """
    result = _understand_mod.describe(df)
    _understand_mod._print_understand(result, len(df))


def understand_df(df: pl.DataFrame) -> pl.DataFrame:
    """Return a one-row-per-column schema DataFrame for *df*, silently.

    Same inference logic as ``understand`` but prints nothing. Useful in
    scripts and tests that need to inspect the result programmatically.

    Args:
        df: Any Polars DataFrame.

    Returns:
        DataFrame with columns: column, dtype, null_pct, unique_count,
        label, sample.
    """
    return _understand_mod.describe(df)


__all__ = ["OmnaFrame", "understand", "understand_df"]
