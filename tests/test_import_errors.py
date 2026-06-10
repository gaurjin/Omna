"""The bare `pip install omna` ships no fastembed/presidio — the lazy imports
must raise an actionable error naming the extra to install (mirrors ask.py)."""
import sys

import pytest


def test_embedder_missing_fastembed_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # forces ImportError
    from omna import embedder

    with pytest.raises(ImportError, match=r"pip install .?omna\[embed\]"):
        embedder.create_embedding_model()


def test_pii_missing_presidio_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    from omna import pii

    monkeypatch.setattr(pii, "_ANALYZER", None)  # reset process-local cache
    with pytest.raises(ImportError, match=r"pip install .?omna\[pii\]"):
        pii._get_analyzer()
