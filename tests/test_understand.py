"""Day 6 tests — omna.understand() and understand.describe()."""
import polars as pl
import pytest
import omna
from omna.understand import describe, _infer_label


# ── describe() return shape ──────────────────────────────────────────────────

def test_returns_dataframe():
    df = pl.DataFrame({"name": ["Alice"], "age": [30]})
    result = describe(df)
    assert isinstance(result, pl.DataFrame)


def test_one_row_per_column():
    df = pl.DataFrame({"a": [1], "b": ["x"], "c": [True]})
    assert len(describe(df)) == 3


def test_expected_columns():
    df = pl.DataFrame({"x": [1]})
    result = describe(df)
    assert set(result.columns) == {"column", "dtype", "null_pct", "unique_count", "label", "sample"}


def test_empty_dataframe_returns_schema_only():
    df = pl.DataFrame(schema={"text": pl.String, "val": pl.Int64})
    result = describe(df)
    assert set(result.columns) == {"column", "dtype", "null_pct", "unique_count", "label", "sample"}


# ── label inference ───────────────────────────────────────────────────────────

def test_boolean_column_labelled_boolean():
    df = pl.DataFrame({"active": [True, False]})
    row = describe(df).filter(pl.col("column") == "active")
    assert row["label"][0] == "boolean"


def test_integer_column_labelled_numeric():
    df = pl.DataFrame({"score": [10, 20, 30]})
    row = describe(df).filter(pl.col("column") == "score")
    assert row["label"][0] == "numeric"


def test_id_column_by_name():
    df = pl.DataFrame({"user_id": [1, 2, 3]})
    row = describe(df).filter(pl.col("column") == "user_id")
    assert row["label"][0] == "id"


def test_email_column_by_name():
    df = pl.DataFrame({"email": ["a@b.com", "c@d.org"]})
    row = describe(df).filter(pl.col("column") == "email")
    assert row["label"][0] == "email"


def test_email_column_by_sample_values():
    df = pl.DataFrame({"contact": ["alice@example.com", "bob@corp.io"]})
    row = describe(df).filter(pl.col("column") == "contact")
    assert row["label"][0] == "email"


def test_phone_column_by_name():
    df = pl.DataFrame({"phone": ["555-1234", "555-5678"]})
    row = describe(df).filter(pl.col("column") == "phone")
    assert row["label"][0] == "phone"


def test_name_column_by_name():
    df = pl.DataFrame({"full_name": ["Alice Smith", "Bob Jones"]})
    row = describe(df).filter(pl.col("column") == "full_name")
    assert row["label"][0] == "name"


def test_date_column_by_dtype():
    df = pl.DataFrame({"created": [1, 2]}).with_columns(
        pl.col("created").cast(pl.Date)
    )
    row = describe(df).filter(pl.col("column") == "created")
    assert row["label"][0] == "date"


def test_text_column_by_name():
    df = pl.DataFrame({"description": ["long text here"]})
    row = describe(df).filter(pl.col("column") == "description")
    assert row["label"][0] == "text"


def test_long_string_values_labelled_text():
    long_val = "x" * 80
    df = pl.DataFrame({"notes": [long_val, long_val]})
    row = describe(df).filter(pl.col("column") == "notes")
    assert row["label"][0] == "text"


def test_short_string_values_labelled_category():
    df = pl.DataFrame({"status": ["open", "closed", "open"]})
    row = describe(df).filter(pl.col("column") == "status")
    assert row["label"][0] == "category"


# ── null_pct and unique_count ─────────────────────────────────────────────────

def test_null_pct_correct():
    df = pl.DataFrame({"x": [1, None, None, None]})
    row = describe(df).filter(pl.col("column") == "x")
    assert row["null_pct"][0] == 75.0


def test_null_pct_zero_when_no_nulls():
    df = pl.DataFrame({"x": [1, 2, 3]})
    row = describe(df).filter(pl.col("column") == "x")
    assert row["null_pct"][0] == 0.0


def test_unique_count():
    df = pl.DataFrame({"x": ["a", "b", "a", "c"]})
    row = describe(df).filter(pl.col("column") == "x")
    assert row["unique_count"][0] == 3


# ── sample field ──────────────────────────────────────────────────────────────

def test_sample_contains_first_values():
    df = pl.DataFrame({"x": ["alpha", "beta", "gamma", "delta"]})
    row = describe(df).filter(pl.col("column") == "x")
    sample = row["sample"][0]
    assert "alpha" in sample
    assert "beta" in sample
    assert "gamma" in sample
    assert "delta" not in sample  # only first 3


def test_sample_skips_nulls():
    df = pl.DataFrame({"x": [None, None, "first", "second"]})
    row = describe(df).filter(pl.col("column") == "x")
    assert "first" in row["sample"][0]


# ── omna.understand() top-level function ────────────────────────────────────

def test_top_level_understand_returns_dataframe():
    df = pl.DataFrame({"email": ["a@b.com"], "age": [25]})
    result = omna.understand(df)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 2


def test_top_level_understand_labels_email():
    df = pl.DataFrame({"email_address": ["a@b.com"]})
    result = omna.understand(df)
    assert result["label"][0] == "email"
