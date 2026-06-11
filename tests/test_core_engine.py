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


def test_engine_core_masks_with_shield_tokens():
    out = mask_pii(_df(), columns=["email", "ssn"], engine="core")
    assert out["email"].to_list()[0].startswith("[EMAIL_")
    assert out["ssn"].to_list()[0].startswith("[GOV_ID_")


def test_engine_core_redacts_secrets_irreversibly():
    out = mask_pii(_df(), columns=["notes"], engine="core")
    cell = out["notes"].to_list()[0]
    assert "AKIA" not in cell
    assert "[REDACTED:" in cell


def test_engine_core_default_is_still_presidio():
    import inspect

    from omna.pii import mask_pii as mp

    sig = inspect.signature(mp)
    assert sig.parameters["engine"].default == "presidio", (
        "the default must not flip before the >=0.95 Gretel gate passes "
        "(measured 0.524 on 2026-06-10 — see benchmarks.json core_engine note)"
    )


def test_engine_rejects_unknown_name():
    with pytest.raises(ValueError):
        mask_pii(_df(), columns=["email"], engine="nope")


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
