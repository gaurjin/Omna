"""Verify that `import omna` is fast and does not load heavy optional deps."""
import subprocess
import sys


def _run(code: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_heavy_deps_not_loaded_at_import():
    """fastembed, presidio, spacy, and anthropic must not appear in sys.modules
    after a bare `import omna`."""
    result = _run(
        "import omna, sys; "
        "heavy = [m for m in sys.modules "
        "         if any(h in m for h in ('fastembed','presidio','anthropic','spacy'))];"
        "print(','.join(heavy))"
    )
    assert result.returncode == 0, f"import omna failed:\n{result.stderr}"
    leaked = result.stdout.strip()
    assert leaked == "", (
        f"Heavy optional deps leaked into `import omna`:\n  {leaked}\n"
        "These must be lazily imported inside their respective functions."
    )


def test_import_omna_under_200ms():
    """`import omna` must complete in under 200ms (heavy deps are lazy-loaded)."""
    result = _run(
        "import time; "
        "t = time.perf_counter(); "
        "import omna; "
        "print(f'{time.perf_counter() - t:.4f}')"
    )
    assert result.returncode == 0, f"import omna failed:\n{result.stderr}"
    elapsed = float(result.stdout.strip())
    assert elapsed < 0.2, (
        f"`import omna` took {elapsed:.3f}s — expected < 0.2s.\n"
        "A heavy dependency is likely being imported at module level."
    )
