"""omna.understand — schema inference and column labelling."""
from __future__ import annotations

import re

import polars as pl

_EMAIL_RE = re.compile(r'^[\w.+\-]+@[\w\-]+\.[a-z]{2,}$', re.IGNORECASE)
_PHONE_RE = re.compile(r'^[\d\s()\-+\.]{7,20}$')

_DATE_TYPES = (pl.Date, pl.Datetime, pl.Time, pl.Duration)


def _infer_label(name: str, dtype: pl.DataType, samples: list) -> str:
    """Return a semantic label for one column."""
    name_l = name.lower()

    # dtype-first: unambiguous types
    if dtype == pl.Boolean:
        return "boolean"
    if any(isinstance(dtype, t) for t in _DATE_TYPES):
        return "date"
    if dtype.is_numeric():
        if name_l in ("id",) or name_l.endswith("_id") or name_l.endswith("id"):
            return "id"
        return "numeric"

    # string column — try name keywords first, then sample patterns
    if any(k in name_l for k in ("email", "mail")):
        return "email"
    if any(k in name_l for k in ("phone", "tel", "mobile", "cell", "fax")):
        return "phone"
    if any(k in name_l for k in ("name", "person", "author", "first", "last", "full")):
        return "name"
    if name_l == "id" or name_l.endswith("_id"):
        return "id"
    if any(k in name_l for k in ("date", "time", "stamp", "created", "updated")):
        return "date"
    if any(k in name_l for k in ("text", "body", "content", "description",
                                  "message", "comment", "note", "summary",
                                  "review", "bio", "detail")):
        return "text"

    # sample-based fallback for string values
    str_samples = [s for s in samples if isinstance(s, str) and s]
    if str_samples:
        if all(_EMAIL_RE.match(s) for s in str_samples):
            return "email"
        if all(_PHONE_RE.match(s) for s in str_samples):
            return "phone"
        avg_len = sum(len(s) for s in str_samples) / len(str_samples)
        if avg_len > 60:
            return "text"
        return "category"

    return "unknown"


def describe(df: pl.DataFrame) -> pl.DataFrame:
    """Infer schema and label each column in *df*.

    Analyzes column name, dtype, null rate, cardinality, and sample values to
    assign a semantic label without any LLM call. Fast and offline.

    Labels assigned: email, phone, name, id, date, text, numeric, boolean,
    category, unknown.

    Args:
        df: Any Polars DataFrame.

    Returns:
        One-row-per-column DataFrame with columns:
        column, dtype, null_pct, unique_count, label, sample.
        *sample* is the first three non-null values joined by ", ".
    """
    n = len(df)
    rows: list[dict] = []

    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        null_count = series.null_count()
        null_pct = round(100.0 * null_count / n, 1) if n > 0 else 0.0

        non_null = series.drop_nulls()
        unique_count = int(non_null.n_unique()) if len(non_null) > 0 else 0
        samples = non_null.head(3).to_list()

        rows.append({
            "column": col,
            "dtype": str(dtype),
            "null_pct": null_pct,
            "unique_count": unique_count,
            "label": _infer_label(col, dtype, samples),
            "sample": ", ".join(repr(v) for v in samples),
        })

    if not rows:
        return pl.DataFrame(schema={
            "column": pl.String, "dtype": pl.String,
            "null_pct": pl.Float64, "unique_count": pl.Int64,
            "label": pl.String, "sample": pl.String,
        })

    return pl.DataFrame(rows).cast({"null_pct": pl.Float64, "unique_count": pl.Int64})
