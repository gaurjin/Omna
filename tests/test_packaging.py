"""Packaging invariants that must hold before any PyPI publish.

These guard the mistakes that only surface at upload/install time: a stale
version that collides with what's already on PyPI, a version dunder that drifts
from the wheel metadata, and a `[pii]` extra that doesn't actually pull the
engine wheel it routes to.
"""
import importlib.metadata as md
import tomllib
from pathlib import Path

import omna

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_dunder_version_matches_installed_metadata():
    """omna.__version__ must equal the version pip/PyPI sees for the wheel."""
    assert omna.__version__ == md.version("omna")


def test_pyproject_version_matches_dunder():
    """pyproject is the source of truth; the package dunder must match it."""
    assert _pyproject()["project"]["version"] == omna.__version__


def test_not_republishing_an_existing_pypi_version():
    """0.1.0 is already on PyPI; uploading it again fails. Guard the bump."""
    assert omna.__version__ != "0.1.0", "Bump the version — 0.1.0 is taken on PyPI"


def test_pii_extra_depends_on_engine_wheel():
    """pip install omna[pii] must pull the omna-pii-mask engine wheel."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any("omna-pii-mask" in d for d in extras["pii"]), \
        "[pii] extra must depend on omna-pii-mask"
