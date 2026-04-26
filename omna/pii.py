"""
omna/pii.py — PII detection and masking with audit logging.

Performance design:
  1. Analyzer is cached per worker process (built once, reused).
  2. Sample-based column detection: scan 1,000 random rows first.
  3. Deduplication: each unique text is masked exactly once; results are
     expanded back to all rows.  Big win on columns with repeated values.
  4. Single ProcessPoolExecutor shared across all columns — workers pay the
     spaCy startup cost once and stay alive for every subsequent column.
  5. All column batches submitted simultaneously so ProfileName and Text
     overlap instead of running back-to-back.
  6. fast=True mode: regex-only masking (no spaCy).  Catches email, phone,
     SSN, credit card, URL in ~1-3 seconds per column.  Misses person names
     in prose.  Recommended for long-text review columns.
"""

from __future__ import annotations

import os
import re
import random
import datetime
import concurrent.futures
from typing import Optional

import polars as pl

# Matches spans that are already government-redacted (XXXX, XXXX XXXX, etc.)
# We skip these so mask_pii() doesn't double-redact the government's own tokens.
_XXXX_SPAN_RE = re.compile(r'^(XX+\s*)+$')

# ---------------------------------------------------------------------------
# Fast regex-only patterns (used when fast=True in mask_pii)
#
# Catches: email, phone (US/intl), SSN, credit card, URL.
# Does NOT catch: person names, locations, organisations — those need spaCy.
# Compiled once at import time so workers share the compiled objects.
# ---------------------------------------------------------------------------

_FAST_PATTERNS: list[re.Pattern] = [
    # Email address
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    # US phone: (555) 867-5309, 555-867-5309, +1 555 867 5309, etc.
    re.compile(r'(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)'),
    # US SSN: 123-45-6789
    re.compile(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'),
    # Credit card: 4 groups of 4 digits (with optional separators)
    re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    # URL starting with http/https
    re.compile(r'https?://[^\s]+'),
]


def _mask_text_fast(text: str, replacement: str = "<REDACTED>") -> str:
    """Regex-only masking — ~50x faster than Presidio, no spaCy NER required."""
    if not text or not isinstance(text, str):
        return text
    spans: list[tuple[int, int]] = []
    for pat in _FAST_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    if not spans:
        return text
    # Sort descending and merge overlaps so we replace right-to-left safely
    spans.sort(key=lambda x: x[0], reverse=True)
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1][0] = min(merged[-1][0], start)
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    chars = list(text)
    for start, end in merged:
        chars[start:end] = list(replacement)
    return "".join(chars)


def _mask_batch_fast(texts: list[str], replacement: str = "<REDACTED>") -> list[str]:
    """Regex-only batch masking. Picklable for multiprocessing."""
    return [_mask_text_fast(t, replacement) for t in texts]


# ---------------------------------------------------------------------------
# Entity-type allow-list for pii_report()
#
# Presidio proxies spaCy NER, which happily tags short alphanumeric codes
# (B006K2ZZ7K → PERSON 0.85, B001GVISJM → LOCATION 0.85).  Only these
# entity types represent actual personal data; the rest (LOCATION, NRP,
# DATE_TIME, ORG …) are NLP concepts, not PII.
# ---------------------------------------------------------------------------

_REAL_PII_TYPES: frozenset[str] = frozenset({
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN",
    "IBAN_CODE", "MEDICAL_LICENSE", "US_PASSPORT", "US_DRIVER_LICENSE",
    "UK_NHS", "SG_NRIC_FIN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
    "IN_PAN", "IN_AADHAAR",
})

# Presidio's phone recogniser scores ~0.4; 0.35 is its own internal floor.
_MIN_SCORE = 0.35
_HIT_RATE_THRESHOLD = 0.10

# ---------------------------------------------------------------------------
# Process-local analyzer cache
# Each worker process builds ONE analyzer and reuses it for all rows.
# We never pass the analyzer across process boundaries (avoids pickle errors).
# ---------------------------------------------------------------------------

_ANALYZER = None  # module-level, lives inside each worker process

# spaCy pipeline components that Presidio never uses.
# Disabling them gives ~2x throughput with identical NER accuracy:
# the ner component in en_core_web_lg has its own internal contextual
# representations and does not depend on the shared tok2vec.
_SPACY_UNUSED_PIPES = ("tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer")


def _get_analyzer():
    """Return the process-local Presidio analyzer, building it if needed."""
    global _ANALYZER
    if _ANALYZER is None:
        from presidio_analyzer import AnalyzerEngine
        _ANALYZER = AnalyzerEngine()
        # Disable unused spaCy pipeline stages for ~2x speedup.
        nlp = _ANALYZER.nlp_engine.nlp.get("en")
        if nlp is not None:
            keep = [p for p in nlp.pipe_names if p not in _SPACY_UNUSED_PIPES]
            nlp.select_pipes(enable=keep)
    return _ANALYZER


# ---------------------------------------------------------------------------
# Worker functions — these run inside child processes
# ---------------------------------------------------------------------------

def _analyze_text(text: str) -> list[str]:
    """
    Detect PII entity types in a single string.
    Returns a list of entity type strings, e.g. ['PERSON', 'EMAIL_ADDRESS'].
    """
    if not text or not isinstance(text, str):
        return []
    analyzer = _get_analyzer()
    results = analyzer.analyze(text=text, language="en")
    return [r.entity_type for r in results]


def _mask_text(text: str, replacement: str = "<REDACTED>") -> str:
    """
    Mask all PII in a single string.
    Returns the masked string.
    """
    if not text or not isinstance(text, str):
        return text
    analyzer = _get_analyzer()
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text
    # Apply the same filters as pii_report: only genuine PII entity types
    # at a meaningful confidence score.  Without this, spaCy mislabels things
    # like "all hours" (DATE_TIME) or "XXXX" tokens as redactable entities.
    real_results = [
        r for r in results
        if r.entity_type in _REAL_PII_TYPES and r.score >= _MIN_SCORE
    ]
    if not real_results:
        return text
    # Replace right-to-left so earlier character positions stay valid.
    results_sorted = sorted(real_results, key=lambda r: r.start, reverse=True)
    chars = list(text)
    for r in results_sorted:
        span_text = text[r.start:r.end]
        if _XXXX_SPAN_RE.match(span_text.strip()):
            continue  # already government-redacted — don't double-redact
        chars[r.start:r.end] = list(replacement)
    return "".join(chars)


def _analyze_batch(texts: list[str]) -> list[list[str]]:
    """
    Analyze a batch of texts. Runs inside a worker process.
    Returns a list of entity-type lists, one per input text.
    """
    return [_analyze_text(t) for t in texts]


def _mask_batch(texts: list[str], replacement: str = "<REDACTED>") -> list[str]:
    """
    Mask a batch of texts. Runs inside a worker process.
    Returns a list of masked strings, one per input text.
    """
    return [_mask_text(t, replacement) for t in texts]


# ---------------------------------------------------------------------------
# Column-level helpers
# ---------------------------------------------------------------------------

def _sample_column_for_pii(values: list[str], sample_size: int = 1000) -> bool:
    """
    Sample up to `sample_size` non-null values from the column.
    Returns True if ≥95% of sampled rows contain at least one PII entity.
    This lets us skip scanning all 500k rows when a column is obviously PII.
    """
    non_null = [v for v in values if v and isinstance(v, str)]
    if not non_null:
        return False
    sample = random.sample(non_null, min(sample_size, len(non_null)))
    hits = sum(1 for text in sample if _analyze_text(text))
    return (hits / len(sample)) >= 0.95


def _parallel_map(fn, items: list, batch_size: int = 500) -> list:
    """
    Split `items` into batches, run `fn` on each batch in a separate
    worker process (one process per CPU core), then flatten results.

    `fn` must be a module-level function (picklable).
    We do NOT pass the analyzer — each worker builds its own.
    """
    n_workers = os.cpu_count() or 1

    # Split into batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(fn, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    # as_completed returns out of order — we need to preserve row order.
    # Redo with map() which preserves order:
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch_result in pool.map(fn, batches):
            results.extend(batch_result)

    return results


# ---------------------------------------------------------------------------
# Public API — called from frame.py
# ---------------------------------------------------------------------------

def detect_pii_columns(df: pl.DataFrame, sample_size: int = 1000) -> dict[str, list[str]]:
    """
    Scan string columns for PII using sample-based detection.

    Uses the same filtering logic as pii_report():
      - Only _REAL_PII_TYPES entities count (excludes LOCATION, NRP, DATE_TIME …)
      - Score must be >= _MIN_SCORE (0.35) to suppress near-zero noise
      - Hit rate must exceed _HIT_RATE_THRESHOLD (10%) of sampled rows
      - At least 2 *distinct* text values must have triggered a hit — this
        prevents a single repeated value (e.g. the same product code appearing
        several times in a small slice) from falsely flagging the whole column.

    Returns a dict mapping column name → sorted list of PII entity types found.
    """
    pii_columns: dict[str, list[str]] = {}
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    analyzer = _get_analyzer()

    for col in string_cols:
        values = df[col].to_list()
        non_null = [v for v in values if v and isinstance(v, str)]
        if not non_null:
            continue

        sample = random.sample(non_null, min(sample_size, len(non_null)))
        entity_types: set[str] = set()
        hits = 0
        hit_values: set[str] = set()

        for text in sample:
            results = analyzer.analyze(text=text, language="en")
            real_hits = [
                r for r in results
                if r.entity_type in _REAL_PII_TYPES and r.score >= _MIN_SCORE
            ]
            if real_hits:
                hits += 1
                hit_values.add(text)
                entity_types.update(r.entity_type for r in real_hits)

        hit_rate = hits / len(sample) if sample else 0.0
        # Require > 1 distinct text values with hits (prevents one repeated
        # false-positive value from flagging the column), but skip that guard
        # when hit_rate is very high (> 50%) — a column where half the rows
        # contain PII is clearly PII even if they all share the same value.
        distinct_hits = len(hit_values)
        if hit_rate > _HIT_RATE_THRESHOLD and (distinct_hits > 1 or hit_rate > 0.5):
            pii_columns[col] = sorted(entity_types)

    return pii_columns


def pii_report(df: pl.DataFrame) -> pl.DataFrame:
    """
    Scan the DataFrame for PII and return a report DataFrame.

    Uses sample-based detection (1,000 rows per column) so it is fast
    even on very large DataFrames.

    False-positive suppression strategy:
      1. Entity-type filter — only entities in _REAL_PII_TYPES are counted.
         spaCy NER fires on alphanumeric codes (B006K2ZZ7K → PERSON 0.85,
         B001GVISJM → LOCATION 0.85); LOCATION, NRP, DATE_TIME etc. are
         discarded.  This drops ProductId-style columns from ~11% hit rate to
         ~5% before the threshold is even applied.
      2. Minimum score — results with score < _MIN_SCORE (0.35) are ignored.
         Presidio's phone recogniser scores 0.4; this floor catches it while
         still discarding near-zero noise.
      3. Hit-rate threshold — a column is only flagged when > 10% of sampled
         rows contain at least one qualifying entity.

    Returns a Polars DataFrame with columns:
      column | pii_types | sample_size | rows_with_pii | flagged | avg_confidence
    """
    rows = []
    string_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    sample_size = 1000

    analyzer = _get_analyzer()

    for col in string_cols:
        values = df[col].to_list()
        non_null = [v for v in values if v and isinstance(v, str)]
        if not non_null:
            continue

        sample = random.sample(non_null, min(sample_size, len(non_null)))
        entity_types: set[str] = set()
        hits = 0
        conf_scores: list[float] = []

        for text in sample:
            results = analyzer.analyze(text=text, language="en")
            real_hits = [
                r for r in results
                if r.entity_type in _REAL_PII_TYPES and r.score >= _MIN_SCORE
            ]
            if real_hits:
                hits += 1
                conf_scores.extend(r.score for r in real_hits)
                entity_types.update(r.entity_type for r in real_hits)

        hit_rate = hits / len(sample) if sample else 0.0
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
        flagged = hit_rate > _HIT_RATE_THRESHOLD

        rows.append({
            "column": col,
            "pii_types": ", ".join(sorted(entity_types)) if entity_types else "",
            "sample_size": len(sample),
            "rows_with_pii": hits,
            "flagged": flagged,
            "avg_confidence": round(avg_conf, 3),
        })

    if not rows:
        return pl.DataFrame(schema={
            "column": pl.Utf8,
            "pii_types": pl.Utf8,
            "sample_size": pl.Int64,
            "rows_with_pii": pl.Int64,
            "flagged": pl.Boolean,
            "avg_confidence": pl.Float64,
        })

    return pl.DataFrame(rows)


def mask_pii(
    df: pl.DataFrame,
    columns: Optional[list[str]] = None,
    replacement: str = "<REDACTED>",
    audit_path: Optional[str] = None,
    fast: bool = False,
) -> pl.DataFrame:
    """
    Mask PII in all string columns (or the specified columns).

    Parameters
    ----------
    df : pl.DataFrame
    columns : list of column names to mask, or None to auto-detect
    replacement : string to replace PII with (default "<REDACTED>")
    audit_path : path to write audit log (CSV), or None to skip
    fast : bool, default False
        If True, use regex-only masking instead of full Presidio + spaCy.
        Catches email, phone, SSN, credit card, URL.
        Does NOT catch person names written in prose.
        ~10-50x faster on long-text columns — recommended for review/body text.
        If False (default), use full Presidio with spaCy NER for maximum recall.

    Returns a new DataFrame with PII masked.
    """
    if columns is None:
        detected = detect_pii_columns(df, sample_size=1000)
        columns = list(detected.keys())

    columns = [c for c in columns if c in df.columns and df[c].dtype == pl.Utf8]
    if not columns:
        return df

    batch_fn = _mask_batch_fast if fast else _mask_batch
    n_workers = os.cpu_count() or 1
    # Large batches minimise IPC overhead; workers stay busy per batch.
    batch_size = max(500, 50_000 // n_workers)

    masked_df = df.clone()
    audit_rows = []

    # Build deduplication maps upfront — mask each unique text exactly once.
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

    # Single pool shared across all columns — workers pay the spaCy startup
    # cost once.  Submit batches for ALL columns simultaneously so ProfileName
    # (fast) and Text (slow) process in true parallel overlap.
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        col_futures: dict[str, list] = {}
        for col in columns:
            _, unique_vals = col_meta[col]
            batches = [
                unique_vals[i : i + batch_size]
                for i in range(0, len(unique_vals), batch_size)
            ]
            col_futures[col] = [pool.submit(batch_fn, b, replacement) for b in batches]

        # Collect results in column order (order within each column preserved)
        for col in columns:
            values, unique_vals = col_meta[col]
            masked_unique: list[str] = []
            for f in col_futures[col]:
                masked_unique.extend(f.result())

            mask_map = dict(zip(unique_vals, masked_unique))
            masked_values = [mask_map[v] if v is not None else None for v in values]

            masked_df = masked_df.with_columns(
                pl.Series(name=col, values=masked_values)
            )
            changed = sum(1 for a, b in zip(values, masked_values) if a != b)
            audit_rows.append({
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "column": col,
                "rows_scanned": len(values),
                "rows_masked": changed,
                "replacement": replacement,
            })

    if audit_path and audit_rows:
        pl.DataFrame(audit_rows).write_csv(audit_path)

    return masked_df


# ---------------------------------------------------------------------------
# Backwards-compatible aliases — keeps existing tests passing
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

# Old names that tests import directly
report = pii_report
mask = mask_pii