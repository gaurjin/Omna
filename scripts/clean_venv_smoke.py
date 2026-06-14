"""Clean-venv smoke test for a PyPI-style install of omna + omna-pii-mask.

Run with the clean-venv interpreter from a directory OUTSIDE the source tree so
it exercises the INSTALLED wheel, not ./omna/:

    cd /tmp && /tmp/omna-fresh/bin/python ~/Developer/Omna/scripts/clean_venv_smoke.py

Must be a real file (not piped via stdin): mask_pii(model=False) uses a
ProcessPoolExecutor, and macOS 'spawn' re-imports __main__ in workers.

Verifies, against the installed wheels:
  1. bare import omna loads no heavy deps
  2. real hybrid search ranks an exact code #1
  3. mask_pii(model=False) redacts emails / SSNs / secrets
  4. mask_pii(model=True) redacts a bare prose name (the L3-only win)
"""
import sys

import polars as pl
import omna


def check_bare_import():
    leaked = [m for m in sys.modules
              if any(h in m for h in ("fastembed", "anthropic", "omna_pii_mask", "onnxruntime"))]
    assert not leaked, f"heavy deps leaked at import: {leaked}"
    print(f"[1] bare import OK — omna {omna.__version__} from {omna.__file__}")


def check_hybrid_search():
    df = pl.DataFrame({"text": [
        "claim denied for water damage in basement",
        "replacement part XJ9000 is on backorder",
        "invoice paid in full, thank you",
        "customer reports a leak under the sink",
    ]})
    df = df.omna.embed("text")
    out = df.omna.search("XJ9000", on="text", k=1)
    assert "XJ9000" in out["text"][0], out["text"].to_list()
    print(f"[2] hybrid search OK — exact code ranked #1: {out['text'][0]!r}")


def check_mask_regex():
    df = pl.DataFrame({"note": [
        "Email john@acme.com, SSN 123-45-6789, call 415-555-0198",
        "AWS key AKIAIOSFODNN7EXAMPLE in config",
    ]})
    m = df.omna.mask_pii(model=False)
    r0, r1 = m["note"][0], m["note"][1]
    assert "john@acme.com" not in r0 and "123-45-6789" not in r0, r0
    assert "AKIAIOSFODNN7EXAMPLE" not in r1, r1
    print(f"[3] mask model=False OK — row0={r0!r}")


def check_mask_model():
    import omna_pii_mask
    omna_pii_mask.download_model()  # idempotent; downloads ~809 MB once
    df = pl.DataFrame({"note": ["Please contact Jane Doe about the overdue balance."]})
    m = df.omna.mask_pii(model=True)
    assert "Jane Doe" not in m["note"][0], m["note"][0]
    print(f"[4] mask model=True OK — bare name redacted: {m['note'][0]!r}")


if __name__ == "__main__":
    check_bare_import()
    check_hybrid_search()
    check_mask_regex()
    check_mask_model()
    print("\nALL CLEAN-VENV SMOKE CHECKS PASSED")
