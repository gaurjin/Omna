"""Tests for the opt-in omna-core unified engine path (workspace #94).

The omna_core wheel is not on PyPI yet — every test here skips cleanly when
it isn't installed, so CI without the wheel stays green.
"""

import polars as pl
import pytest

from omna.pii import _core_engine_available, mask_pii

core = pytest.importorskip("omna_core") if _core_engine_available() else None

pytestmark = pytest.mark.skipif(
    not _core_engine_available(), reason="omna_core wheel not installed"
)


def _df():
    return pl.DataFrame(
        {
            "email": ["bob@corp.com", "carol@corp.com"],
            "ssn": ["123-45-6749", "234-56-7891"],
            "notes": ["api_key: AKIAIOSFODNN7EXAMPLE", "nothing here"],
        }
    )


def test_core_masks_with_shield_tokens():
    out = mask_pii(_df(), columns=["email", "ssn"])
    assert out["email"].to_list()[0].startswith("[EMAIL_")
    assert out["ssn"].to_list()[0].startswith("[GOV_ID_")


def test_core_redacts_secrets_irreversibly():
    out = mask_pii(_df(), columns=["notes"])
    cell = out["notes"].to_list()[0]
    assert "AKIA" not in cell
    assert "[REDACTED:" in cell


def test_core_is_the_only_engine():
    """Presidio + spaCy were removed 2026-06-13 — the unified Rust engine is
    the sole masking path (no `engine=` parameter, no legacy fallback). The
    engine ported Presidio's useful rules into L1 and replaced its spaCy NER
    with the L3 model, so the dependency is gone with no loss of capability.
    """
    import inspect

    from omna.pii import mask_pii as mp

    assert "engine" not in inspect.signature(mp).parameters
    out = mask_pii(_df(), columns=["email"])
    assert out["email"].to_list()[0].startswith("[EMAIL_")


def test_namespace_mask_pii_works(tmp_path):
    """The df.omna.mask_pii namespace method masks via the core engine."""
    out = _df().omna.mask_pii(audit_path=tmp_path / "a.parquet")
    assert out["email"].to_list()[0].startswith("[EMAIL_")


def test_ask_serialize_masks_sample_rows():
    from omna import ask

    df = pl.DataFrame({"email": ["alice@corp.com"], "ssn": ["123-45-6749"]})
    s = ask._serialize(df)
    assert "alice@corp.com" not in s
    assert "123-45-6749" not in s
    # Schema/stats still present — only the sample rows are masked.
    assert "Shape: 1 rows" in s


def test_ask_serialize_mask_rows_false_optout():
    from omna import ask

    df = pl.DataFrame({"email": ["alice@corp.com"]})
    s = ask._serialize(df, mask_rows=False)
    assert "alice@corp.com" in s
