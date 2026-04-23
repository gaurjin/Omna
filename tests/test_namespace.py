"""Day 1 smoke tests — namespace registration only."""
import pytest
import polars as pl
import omna  # noqa: F401 — side-effect: registers df.omna


def test_namespace_is_registered():
    df = pl.DataFrame({"text": ["hello", "world"]})
    assert hasattr(df, "omna")


def test_all_methods_exist():
    df = pl.DataFrame({"text": ["hello"]})
    ns = df.omna
    for method in ("embed", "search", "filter", "mask_pii", "pii_report", "ask"):
        assert callable(getattr(ns, method)), f"df.omna.{method} is missing"


def test_search_without_index_raises():
    df = pl.DataFrame({"text": ["hello"]})
    with pytest.raises(FileNotFoundError):
        df.omna.search("query", on="text", index_path="/tmp/omna_no_such_index.parquet")


def test_mask_pii_returns_dataframe():
    df = pl.DataFrame({"text": ["hello world"]})
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        result = df.omna.mask_pii(audit_path=pathlib.Path(tmp) / "audit.parquet")
        assert isinstance(result, pl.DataFrame)
