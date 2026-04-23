"""omna.pii — Presidio-powered PII detection and redaction."""
from __future__ import annotations

from functools import lru_cache

import polars as pl

_REPORT_SCHEMA = {
    "column": pl.String,
    "row": pl.Int64,
    "entity_type": pl.String,
    "text": pl.String,
    "score": pl.Float64,
}

# DATE_TIME and NRP produce too many false positives on normal business text.
_PII_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "US_BANK_NUMBER",
    "US_ITIN", "IP_ADDRESS", "IBAN_CODE", "CRYPTO", "MEDICAL_LICENSE",
    "LOCATION", "URL", "MAC_ADDRESS", "UK_NHS",
]

_PII_INSTALL_MSG = (
    "Presidio and spaCy are required for PII detection. "
    "Install them with:  pip install omna[pii]"
)


@lru_cache(maxsize=1)
def _analyzer():
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        raise ImportError(_PII_INSTALL_MSG) from None
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def _anonymizer():
    try:
        from presidio_anonymizer import AnonymizerEngine
    except ImportError:
        raise ImportError(_PII_INSTALL_MSG) from None
    return AnonymizerEngine()


def _scan(text: str) -> list:
    """Return Presidio results for one string. Empty list if text is blank."""
    if not text or not text.strip():
        return []
    return _analyzer().analyze(text=text, language="en", entities=_PII_ENTITIES)


def _redact(text: str, results: list) -> str:
    """Replace each PII span with its <ENTITY_TYPE> token."""
    if not results:
        return text
    try:
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        raise ImportError(_PII_INSTALL_MSG) from None
    operators = {
        r.entity_type: OperatorConfig("replace", {"new_value": f"<{r.entity_type}>"})
        for r in results
    }
    return _anonymizer().anonymize(
        text=text, analyzer_results=results, operators=operators
    ).text


def _string_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == pl.String]


def report(df: pl.DataFrame) -> pl.DataFrame:
    """Scan all string columns in *df* and return a PII findings report.

    Does not modify the DataFrame. Use this to preview what mask() will redact.

    Each row in the returned DataFrame is one detected PII entity:

    - column: which DataFrame column the entity was found in
    - row: integer row index
    - entity_type: Presidio label (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, …)
    - text: the original PII string
    - score: Presidio confidence (0–1)

    Args:
        df: DataFrame to scan. Non-string columns are skipped.

    Returns:
        DataFrame with schema (column, row, entity_type, text, score).
        Empty DataFrame with the same schema when no PII is found.
    """
    findings: list[dict] = []
    for col in _string_columns(df):
        for i, cell in enumerate(df[col].to_list()):
            if cell is None:
                continue
            for r in _scan(cell):
                findings.append({
                    "column": col,
                    "row": i,
                    "entity_type": r.entity_type,
                    "text": cell[r.start:r.end],
                    "score": round(float(r.score), 4),
                })
    if not findings:
        return pl.DataFrame(schema=_REPORT_SCHEMA)
    return pl.DataFrame(findings).cast({"row": pl.Int64, "score": pl.Float64})


def mask(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Redact PII in all string columns of *df*.

    Scans every string column, replaces each detected PII span with its
    <ENTITY_TYPE> token, and produces an audit log with the same schema
    as report() so you can verify what was redacted.

    Args:
        df: DataFrame to redact.

    Returns:
        (masked_df, audit_df) — masked_df has PII replaced with tokens;
        audit_df lists every redaction (column, row, entity_type, text, score).
    """
    audit: list[dict] = []
    new_columns: dict[str, list] = {}

    for col in _string_columns(df):
        masked_cells: list[str | None] = []
        for i, cell in enumerate(df[col].to_list()):
            if cell is None:
                masked_cells.append(None)
                continue
            results = _scan(cell)
            for r in results:
                audit.append({
                    "column": col,
                    "row": i,
                    "entity_type": r.entity_type,
                    "text": cell[r.start:r.end],
                    "score": round(float(r.score), 4),
                })
            masked_cells.append(_redact(cell, results))
        new_columns[col] = masked_cells

    masked_df = df.with_columns([
        pl.Series(col, vals, dtype=pl.String)
        for col, vals in new_columns.items()
    ])

    if not audit:
        audit_df = pl.DataFrame(schema=_REPORT_SCHEMA)
    else:
        audit_df = pl.DataFrame(audit).cast({"row": pl.Int64, "score": pl.Float64})

    return masked_df, audit_df
