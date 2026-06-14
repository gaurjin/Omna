"""Edge-case regression guards discovered while hardening for the 0.2.0 release.

The security round-trip is the important one: secrets must NEVER be reversible
(they must not land in the token map), or restore() would un-redact a credential.
"""
import importlib.util

import polars as pl
import pytest

import omna  # noqa: F401 — registers df.omna namespace

_HAS_ENGINE = importlib.util.find_spec("omna_pii_mask") is not None
pytestmark = pytest.mark.skipif(not _HAS_ENGINE, reason="omna-pii-mask engine not installed")


# ── Security: reversibility for PII, irreversibility for secrets ───────────────

def test_secret_is_never_restorable():
    import omna_pii_mask as eng
    r = eng.mask("Reach me at john@acme.com; key AKIAIOSFODNN7EXAMPLE")
    # The secret must not be in the reversible token map...
    assert not any("AKIA" in original for original in r["tokens"].values()), \
        f"secret leaked into the reversible token map: {r['tokens']}"
    # ...and restoring must not bring it back, while the email DOES come back.
    restored = eng.restore(r["masked"], r["tokens"])
    assert "AKIAIOSFODNN7EXAMPLE" not in restored, f"secret restored: {restored}"
    assert "john@acme.com" in restored, f"PII not restored: {restored}"


def test_pii_round_trip_is_lossless():
    import omna_pii_mask as eng
    original = "Call Jane at 415-555-0198 or email jane@x.com"
    r = eng.mask(original)
    assert eng.restore(r["masked"], r["tokens"]) == original


def test_masking_is_deterministic():
    import omna_pii_mask as eng
    text = "SSN 123-45-6789 email a@b.com"
    assert eng.mask(text)["masked"] == eng.mask(text)["masked"]


# ── DataFrame behavioral edges ─────────────────────────────────────────────────

def test_mask_pii_preserves_none_cells():
    out = pl.DataFrame({"note": ["email a@b.com", None, "plain"]}).omna.mask_pii()
    assert out["note"][1] is None
    assert "a@b.com" not in (out["note"][0] or "")


def test_mask_pii_no_string_columns_returns_unchanged():
    df = pl.DataFrame({"n": [1, 2, 3]})
    assert df.omna.mask_pii().equals(df)


def test_mask_pii_all_null_column_no_crash():
    df = pl.DataFrame({"note": [None, None]}, schema={"note": pl.Utf8})
    assert df.omna.mask_pii().shape == df.shape


def test_pii_report_empty_dataframe_no_crash():
    pl.DataFrame({"note": []}, schema={"note": pl.Utf8}).omna.pii_report()
