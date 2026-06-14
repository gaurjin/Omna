"""The bare `pip install omna` ships no fastembed — the lazy import must raise
an actionable error naming the extra to install (mirrors ask.py). PII masking
now needs the compiled `omna_pii_mask` wheel; its absence raises a named error too."""
import sys

import pytest


def test_embedder_missing_fastembed_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # forces ImportError
    from omna import embedder

    with pytest.raises(ImportError, match=r"pip install .?omna\[embed\]"):
        embedder.create_embedding_model()


def test_pii_missing_core_wheel_names_it(monkeypatch):
    """With the omna_pii_mask wheel absent, the PII functions raise an ImportError
    that names the wheel."""
    monkeypatch.setitem(sys.modules, "omna_pii_mask", None)  # forces ImportError
    from omna import pii

    with pytest.raises(ImportError, match=r"omna_pii_mask"):
        pii._require_core()
