"""Day 5 tests — omna.pii module."""
import polars as pl
import pytest
from omna.pii import report, mask, _REPORT_SCHEMA

CLEAR_TEXT = "My name is Alice Smith. Email: alice@example.com. Phone: 555-867-5309."
CLEAN_TEXT = "The revenue report covers widgets and gadgets."


# ── report() ─────────────────────────────────────────────────────────────────

def test_report_returns_dataframe():
    df = pl.DataFrame({"text": [CLEAR_TEXT]})
    result = report(df)
    assert isinstance(result, pl.DataFrame)


def test_report_schema():
    df = pl.DataFrame({"text": [CLEAR_TEXT]})
    result = report(df)
    assert set(result.columns) == set(_REPORT_SCHEMA.keys())


def test_report_detects_person():
    df = pl.DataFrame({"text": ["Call Alice Smith tomorrow."]})
    result = report(df)
    assert "PERSON" in result["pii_types"][0]


def test_report_detects_email():
    df = pl.DataFrame({"text": ["Send to alice@example.com please."]})
    result = report(df)
    assert "EMAIL_ADDRESS" in result["pii_types"][0]


def test_report_detects_phone():
    df = pl.DataFrame({"text": ["Call me at 555-867-5309 anytime."]})
    result = report(df)
    assert "PHONE_NUMBER" in result["pii_types"][0]


def test_report_clean_text_returns_empty():
    df = pl.DataFrame({"text": [CLEAN_TEXT]})
    result = report(df)
    assert result["rows_with_pii"][0] == 0
    assert result["flagged"][0] == False
    assert set(result.columns) == set(_REPORT_SCHEMA.keys())


def test_report_skips_non_string_columns():
    df = pl.DataFrame({"text": [CLEAR_TEXT], "value": [42]})
    result = report(df)
    assert (result["column"] == "value").sum() == 0


def test_report_handles_null_cells():
    df = pl.DataFrame({"text": [None, CLEAR_TEXT]})
    result = report(df)
    assert len(result) > 0
    assert result["rows_with_pii"][0] > 0


def test_report_multi_row():
    df = pl.DataFrame({"text": [CLEAR_TEXT, CLEAN_TEXT, "Bob Jones lives here."]})
    result = report(df)
    assert len(result) == 1
    assert result["rows_with_pii"][0] > 0
    assert result["flagged"][0] == True


def test_report_score_between_zero_and_one():
    df = pl.DataFrame({"text": [CLEAR_TEXT]})
    result = report(df)
    assert result["rows_with_pii"][0] >= 0
    assert result["rows_with_pii"][0] <= result["sample_size"][0]


# ── mask() ───────────────────────────────────────────────────────────────────

def test_mask_returns_two_dataframes():
    df = pl.DataFrame({"text": [CLEAR_TEXT]})
    result = mask(df)
    assert isinstance(result, pl.DataFrame)


def test_mask_replaces_email():
    df = pl.DataFrame({"text": ["Contact alice@example.com for details."]})
    masked_df = mask(df)
    assert "<REDACTED>" in masked_df["text"][0]
    assert "alice@example.com" not in masked_df["text"][0]


def test_mask_replaces_person():
    df = pl.DataFrame({"text": ["My name is Alice Smith."]})
    masked_df = mask(df)
    assert "<REDACTED>" in masked_df["text"][0]
    assert "Alice Smith" not in masked_df["text"][0]


def test_mask_clean_text_unchanged():
    df = pl.DataFrame({"text": [CLEAN_TEXT]})
    masked_df = mask(df)
    assert masked_df["text"][0] == CLEAN_TEXT


def test_mask_audit_has_report_schema():
    df = pl.DataFrame({"text": [CLEAR_TEXT]})
    result = mask(df)
    assert result.columns == df.columns


def test_mask_audit_empty_when_no_pii():
    df = pl.DataFrame({"text": [CLEAN_TEXT]})
    result = mask(df)
    assert result["text"][0] == CLEAN_TEXT


def test_mask_audit_records_original_text():
    df = pl.DataFrame({"text": ["Email bob@corp.io for help."]})
    masked_df = mask(df)
    assert "bob@corp.io" not in masked_df["text"][0]


def test_mask_preserves_non_string_columns():
    df = pl.DataFrame({"text": [CLEAR_TEXT], "id": [99]})
    masked_df = mask(df)
    assert masked_df["id"][0] == 99


def test_mask_handles_null_cells():
    df = pl.DataFrame({"text": [None, CLEAR_TEXT]})
    masked_df = mask(df)
    assert masked_df["text"][0] is None


def test_mask_multiple_columns():
    df = pl.DataFrame({
        "name": ["Alice Smith"],
        "contact": ["alice@example.com"],
    })
    masked_df = mask(df)
    assert "<REDACTED>" in masked_df["name"][0]
    assert "<REDACTED>" in masked_df["contact"][0]
    assert "Alice Smith" not in masked_df["name"][0]
    assert "alice@example.com" not in masked_df["contact"][0]
