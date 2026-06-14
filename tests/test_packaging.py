"""Packaging invariants that must hold before any PyPI publish.

These guard the mistakes that only surface at upload/install time: a stale
version that collides with what's already on PyPI, a version dunder that drifts
from the wheel metadata, and a `[pii]` extra that doesn't actually pull the
engine wheel it routes to.
"""
import importlib.metadata as md
import json
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import omna

ROOT = Path(__file__).resolve().parent.parent


def _pypi_released_versions(project: str, timeout: float = 5.0) -> set[str] | None:
    """Versions of *project* already on PyPI, or None if PyPI is unreachable.

    Returning None (not an empty set) lets callers skip rather than falsely pass
    when offline — an empty set would wrongly assert "nothing is published".
    """
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return set(json.load(resp)["releases"].keys())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return set()  # project not registered → nothing published yet
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_dunder_version_matches_installed_metadata():
    """omna.__version__ must equal the version pip/PyPI sees for the wheel."""
    assert omna.__version__ == md.version("omna")


def test_pyproject_version_matches_dunder():
    """pyproject is the source of truth; the package dunder must match it."""
    assert _pyproject()["project"]["version"] == omna.__version__


def test_current_version_not_already_on_pypi():
    """PyPI rejects re-uploading an existing version (HTTP 400). Assert the
    current version isn't already published. Queries the live PyPI release set
    (skips if offline) so it stays honest as new versions are released — unlike
    a hardcoded ``!= "0.1.0"`` check, which would green-light re-uploading 0.2.0.
    """
    released = _pypi_released_versions("omna")
    if released is None:
        pytest.skip("PyPI unreachable — cannot verify version availability")
    assert omna.__version__ not in released, (
        f"omna {omna.__version__} is already on PyPI {sorted(released)} — "
        "bump the version before publishing (re-upload fails with HTTP 400)."
    )


def test_pii_extra_depends_on_engine_wheel():
    """pip install omna[pii] must pull the omna-pii-mask engine wheel."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any("omna-pii-mask" in d for d in extras["pii"]), \
        "[pii] extra must depend on omna-pii-mask"
