"""
omna/pii.py — PII detection and masking, powered by the omna-core engine.

This module routes ENTIRELY to the unified L1–L6 Rust engine (`omna_pii_mask`
wheel — the same kernel the Omna Mac app and browser extension ship): high
core-PII recall, 220+ secret rules, checksum-validated IDs, no heavy Python ML
dependencies, and reversible Shield tokens (`[PERSON_1]`, `[EMAIL_1]`, …) —
except secrets/credentials, which are ALWAYS irreversibly `[REDACTED:<KIND>]`.

The engine ships as a compiled wheel (`omna_pii_mask`). If it is not installed,
every function here raises a friendly ImportError naming it.
"""

from __future__ import annotations

import datetime
import os
import random
import concurrent.futures
from typing import Optional

import polars as pl


# ---------------------------------------------------------------------------
# omna-core engine access
# ---------------------------------------------------------------------------

def _core_engine_available() -> bool:
    try:
        import omna_pii_mask  # noqa: F401
        return True
    except ImportError:
        return False


def _require_core() -> None:
    if not _core_engine_available():
        raise ImportError(
            "Omna's PII engine (the `omna_pii_mask` wheel) is not installed. "
            "It ships as a compiled wheel built from omna-workspace "
            "(target/wheels/omna_pii_mask-*.whl)."
        )


def _detect_entities(text: str, column: Optional[str] = None, model: bool = False) -> list[str]:
    """Entity-type names the engine finds in `text` (e.g. ['PERSON','EMAIL']).

    `column` (when given) is prepended as a ``"col: value"`` structure prior —
    the same cue the mask step uses — so a labeled name column ("name: Alice
    Smith") is detected where the bare value would not be. `model=True` adds L3
    (the AI model) so contextual PII (bare prose names) is detected too."""
    if not text or not isinstance(text, str):
        return []
    import omna_pii_mask
    probe = f"{column}: {text}" if column else text
    return [s["entity"] for s in omna_pii_mask.detect(probe, model=model)]


# Hit-rate threshold for flagging a column as containing PII.
_HIT_RATE_THRESHOLD = 0.10


def _mask_batch_core(
    texts: list[str],
    replacement: str = "<REDACTED>",  # ignored; token semantics are the engine's
    column: Optional[str] = None,
    model: bool = False,
) -> list[str]:
    """Mask a batch via the omna-core unified engine. Picklable for the pool.

    `column` (when given) is prepended as a ``"col: value"`` label cue — the
    same structure prior the native row pipeline gets — then stripped from the
    masked output. If a span ever swallowed the label, the cell is re-masked
    without the prior (defensive). `model=True` enables L3 (the on-device AI
    model) for contextual PII like bare prose names."""
    import omna_pii_mask
    out = []
    prefix = f"{column}: " if column else ""
    for t in texts:
        if not t or not isinstance(t, str):
            out.append(t)
            continue
        masked = omna_pii_mask.mask(prefix + t, model=model)["masked"]
        if prefix:
            if masked.startswith(prefix):
                masked = masked[len(prefix):]
            else:
                masked = omna_pii_mask.mask(t, model=model)["masked"]
        out.append(masked)
    return out


# ---------------------------------------------------------------------------
# Public API — called from frame.py
# ---------------------------------------------------------------------------

def detect_pii_columns(df: pl.DataFrame, sample_size: int = 1000, model: bool = False) -> dict[str, list[str]]:
    """
    Scan string columns for PII using sample-based detection on the omna-core
    engine. A column is flagged when more than `_HIT_RATE_THRESHOLD` (10%) of
    sampled rows contain at least one entity. The column name rides along as a
    structure prior (so a "name" column is recognised). The engine is precise
    (no over-firing), so no entity-type allow-list or repeated-value guard is
    needed — both were heuristics for the old noisy-NER detection path.

    Returns a dict mapping column name → sorted list of PII entity types found.
    """
    _require_core()
    pii_columns: dict[str, list[str]] = {}
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]

    for col in string_cols:
        non_null = [v for v in df[col].to_list() if v and isinstance(v, str)]
        if not non_null:
            continue
        sample = random.sample(non_null, min(sample_size, len(non_null)))
        entity_types: set[str] = set()
        hits = 0
        for text in sample:
            ents = _detect_entities(text, col, model)
            if ents:
                hits += 1
                entity_types.update(ents)
        hit_rate = hits / len(sample) if sample else 0.0
        if hit_rate > _HIT_RATE_THRESHOLD:
            pii_columns[col] = sorted(entity_types)

    return pii_columns


def pii_report(df: pl.DataFrame) -> pl.DataFrame:
    """
    Scan the DataFrame for PII (sample-based, 1,000 rows/column) via the
    omna-core engine and return a report DataFrame:

      column | pii_types | sample_size | rows_with_pii | flagged | avg_confidence

    Nothing is modified. A column is `flagged` when > 10% of sampled rows
    contain at least one entity.
    """
    _require_core()
    import omna_pii_mask

    rows = []
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    sample_size = 1000

    for col in string_cols:
        non_null = [v for v in df[col].to_list() if v and isinstance(v, str)]
        if not non_null:
            continue
        sample = random.sample(non_null, min(sample_size, len(non_null)))
        entity_types: set[str] = set()
        conf_scores: list[float] = []
        hits = 0
        for text in sample:
            spans = omna_pii_mask.detect(f"{col}: {text}")
            if spans:
                hits += 1
                entity_types.update(s["entity"] for s in spans)
                conf_scores.extend(float(s["confidence"]) for s in spans)
        hit_rate = hits / len(sample) if sample else 0.0
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
        rows.append({
            "column": col,
            "pii_types": ", ".join(sorted(entity_types)) if entity_types else "",
            "sample_size": len(sample),
            "rows_with_pii": hits,
            "flagged": hit_rate > _HIT_RATE_THRESHOLD,
            "avg_confidence": round(avg_conf, 3),
        })

    if not rows:
        return pl.DataFrame(schema=_REPORT_SCHEMA)
    return pl.DataFrame(rows)


def mask_pii(
    df: pl.DataFrame,
    columns: Optional[list[str]] = None,
    replacement: str = "<REDACTED>",
    audit_path: Optional[str] = None,
    model: bool = False,
) -> pl.DataFrame:
    """
    Mask PII in all string columns (or the specified columns) with the unified
    omna-core engine: reversible `[PERSON_1]`-style Shield tokens, secrets
    always irreversibly `[REDACTED:<KIND>]`, checksum-validated IDs, 220+ secret
    rules. Same kernel as the Omna Mac app and browser extension.

    Parameters
    ----------
    df : pl.DataFrame
    columns : column names to mask, or None to auto-detect
    replacement : retained for signature compatibility; IGNORED (token
        semantics are the engine's)
    audit_path : path to write an audit log (CSV), or None to skip
    model : if True, enable L3 — the on-device AI model that catches contextual
        PII regex can't (bare prose names, addresses). The model (~809 MB)
        downloads once on first use. Runs single-process so the model loads
        once (not per CPU core).

    Returns a new DataFrame with PII masked.
    """
    _require_core()
    if columns is None:
        columns = list(detect_pii_columns(df, sample_size=1000, model=model).keys())
    columns = [c for c in columns if c in df.columns and df[c].dtype == pl.Utf8]
    if not columns:
        return df

    # model=True loads a ~809 MB model — keep it to ONE worker so it loads once,
    # not once per core. L1+L2 (model=False) parallelises freely.
    n_workers = 1 if model else (os.cpu_count() or 1)
    batch_size = max(500, 50_000 // n_workers)

    masked_df = df.clone()
    audit_rows = []

    # Deduplicate — mask each unique value exactly once.
    col_meta: dict[str, tuple[list, list]] = {}
    for col in columns:
        values = df[col].to_list()
        seen: dict[str, int] = {}
        unique_vals: list[str] = []
        for v in values:
            if v is not None and v not in seen:
                seen[v] = len(unique_vals)
                unique_vals.append(v)
        col_meta[col] = (values, unique_vals)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        col_futures: dict[str, list] = {}
        for col in columns:
            _, unique_vals = col_meta[col]
            batches = [
                unique_vals[i : i + batch_size]
                for i in range(0, len(unique_vals), batch_size)
            ]
            # Column name rides along as a structure prior (see _mask_batch_core).
            col_futures[col] = [
                pool.submit(_mask_batch_core, b, replacement, col, model) for b in batches
            ]

        for col in columns:
            values, unique_vals = col_meta[col]
            masked_unique: list[str] = []
            for f in col_futures[col]:
                masked_unique.extend(f.result())
            mask_map = dict(zip(unique_vals, masked_unique))
            masked_values = [mask_map[v] if v is not None else None for v in values]
            masked_df = masked_df.with_columns(pl.Series(name=col, values=masked_values))
            changed = sum(1 for a, b in zip(values, masked_values) if a != b)
            audit_rows.append({
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "column": col,
                "rows_scanned": len(values),
                "rows_masked": changed,
                "replacement": "shield-tokens",
                "engine": "core",
            })

    if audit_path and audit_rows:
        pl.DataFrame(audit_rows).write_csv(audit_path)

    return masked_df


# ---------------------------------------------------------------------------
# Backwards-compatible aliases — keeps existing tests/imports working
# ---------------------------------------------------------------------------

#: Schema used by pii_report() — exported for tests
_REPORT_SCHEMA = {
    "column": pl.Utf8,
    "pii_types": pl.Utf8,
    "sample_size": pl.Int64,
    "rows_with_pii": pl.Int64,
    "flagged": pl.Boolean,
    "avg_confidence": pl.Float64,
}

report = pii_report
mask = mask_pii
