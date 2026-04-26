"""Day 5 tests — df.omna.mask_pii() and df.omna.pii_report() integration."""
import polars as pl
import pytest
import omna  # noqa: F401

CLEAR = "Alice Smith called 555-867-5309 about alice@example.com."
CLEAN = "The revenue report covers widgets and gadgets."


@pytest.fixture
def pii_df():
    return pl.DataFrame({
        "notes": [CLEAR, CLEAN, "Bob Jones owed $500."],
        "id": [1, 2, 3],
    })


# ── pii_report() ─────────────────────────────────────────────────────────────

def test_pii_report_returns_dataframe(pii_df):
    result = pii_df.omna.pii_report()
    assert isinstance(result, pl.DataFrame)


def test_pii_report_finds_pii(pii_df):
    result = pii_df.omna.pii_report()
    assert len(result) > 0


def test_pii_report_does_not_modify_df(pii_df):
    original_notes = pii_df["notes"].to_list()
    pii_df.omna.pii_report()
    assert pii_df["notes"].to_list() == original_notes


def test_pii_report_columns(pii_df):
    result = pii_df.omna.pii_report()
    assert {"column", "pii_types", "sample_size", "rows_with_pii", "flagged"}.issubset(result.columns)


def test_pii_report_clean_df_is_empty():
    df = pl.DataFrame({"notes": [CLEAN, "bolts nuts screws only."]})
    result = df.omna.pii_report()
    assert result["rows_with_pii"][0] == 0
    assert result["flagged"][0] == False


# ── mask_pii() ────────────────────────────────────────────────────────────────

def test_mask_pii_returns_dataframe(pii_df, tmp_path):
    result = pii_df.omna.mask_pii(audit_path=tmp_path / "audit.parquet")
    assert isinstance(result, pl.DataFrame)


def test_mask_pii_same_shape(pii_df, tmp_path):
    result = pii_df.omna.mask_pii(audit_path=tmp_path / "audit.parquet")
    assert result.shape == pii_df.shape


def test_mask_pii_redacts_pii_row(pii_df, tmp_path):
    result = pii_df.omna.mask_pii(audit_path=tmp_path / "audit.parquet")
    assert "Alice Smith" not in result["notes"][0]
    assert "alice@example.com" not in result["notes"][0]


def test_mask_pii_leaves_clean_row_unchanged(pii_df, tmp_path):
    result = pii_df.omna.mask_pii(audit_path=tmp_path / "audit.parquet")
    assert result["notes"][1] == CLEAN


def test_mask_pii_preserves_non_string_columns(pii_df, tmp_path):
    result = pii_df.omna.mask_pii(audit_path=tmp_path / "audit.parquet")
    assert result["id"].to_list() == [1, 2, 3]


def test_mask_pii_writes_audit_file(pii_df, tmp_path):
    audit = tmp_path / "audit.parquet"
    pii_df.omna.mask_pii(audit_path=audit)
    assert audit.exists()


def test_mask_pii_audit_is_readable_parquet(pii_df, tmp_path):
    audit = tmp_path / "audit.csv"
    pii_df.omna.mask_pii(audit_path=audit)
    log = pl.read_csv(audit)
    assert "column" in log.columns


def test_mask_pii_does_not_modify_original(pii_df, tmp_path):
    original = pii_df["notes"].to_list()
    pii_df.omna.mask_pii(audit_path=tmp_path / "audit.parquet")
    assert pii_df["notes"].to_list() == original
