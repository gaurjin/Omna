# Phase B — PyPI Publish-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the `omna` Polars plugin to a "one command away from PyPI" state — verified — so that `pip install "omna[embed,pii,ask]"` yields a working install (hybrid search + PII masking incl. `model=True`) on macOS + Linux, Python 3.10+. Prepare the `omna-pii-mask` engine wheel (the `[pii]` dependency) for publish too. Leave both irreversible uploads for the user.

**Architecture:** `omna` is a maturin-built package (pure-Python `omna/` package + a Rust `_omna` cosine kernel). PII routes to the separate compiled `omna_pii_mask` wheel (built from `omna-workspace/bindings/omna-core-py`, abi3-py39, links ONNX Runtime, runs L3 in-process). PyPI metadata cannot carry git-URL deps, so `pip install omna[pii]` only works if `omna-pii-mask` is also on PyPI → both packages must be prepared. The actual `twine upload`s are the hard stop.

**Tech Stack:** Python 3.10+, maturin 1.13, uv, twine, PyO3, GitHub Actions (PyO3/maturin-action).

---

## Ground-truth findings (verified against real code, 2026-06-13)

- `omna` **0.1.0 is already on PyPI** under `gaurav <gaurjin@gmail.com>` (only 0.1.0 exists). **Re-uploading 0.1.0 fails** → must bump. `omna-pii-mask` is 404 (name free).
- `pyproject.toml` `[pii]` extra is **empty** (`pii = []`) → `pip install omna[pii]` installs nothing. Must depend on `omna-pii-mask`.
- Engine module `omna_pii_mask` exposes `detect`, `mask`, `download_model`, `restore`, `version`; L1/L2 work; `mask(model=True)` runs L3 in-process. **Old plan (separate Microsoft onnxruntime + light Rust wheel + Python BIOES decoder) is OBSOLETE.**
- Version skew in the engine: binding `pyproject.toml` says `0.2.0`, `Cargo.toml` says `0.2.2`, runtime `version()` returns `0.2.2+<hash>.dirty`. **PyPI rejects local `+...` segments** → engine wheel must build from a clean (committed) tree, and the three versions must agree.
- venv has **no pip** (uv-managed) → use `uv pip`. `twine` not installed anywhere → install into venv. uv 0.11.7 + maturin 1.13.1 present on PATH.
- No PyPI/TestPyPI tokens in env → TestPyPI step is **skipped & documented**, not run.
- No pytest config → bare `pytest` wrongly collects `scripts/smoke_test.py` (collection error). Baseline `pytest tests/` = **142 passed**.
- `OMNA_MODEL_DIR` lets the engine reuse the Mac app's downloaded L3 model (avoids 809 MB re-download).

## File structure (created / modified)

- Modify: `~/Developer/Omna/pyproject.toml` — version bump, `[pii]` dep, classifiers, pytest config, urls.
- Modify: `~/Developer/Omna/omna/__init__.py` — version stays single-sourced; add metadata-consistency test target.
- Modify: `~/Developer/Omna/omna/pii.py:37-44` — `_require_core()` message → `pip install "omna[pii]"`.
- Create: `~/Developer/Omna/tests/test_packaging.py` — version/extra/metadata consistency tests.
- Modify: `~/Developer/omna-workspace/bindings/omna-core-py/pyproject.toml` + `Cargo.toml` — version align, metadata, classifiers, readme.
- Create: `~/Developer/omna-workspace/bindings/omna-core-py/README.md` — engine wheel long description.
- Create: `~/Developer/omna-workspace/.github/workflows/release-pypi.yml` — manylinux + macOS abi3 wheels.
- Create: `~/Developer/Omna/docs/RELEASING-PYPI.md` — publish runbook (mirrors omna-workspace/docs/RELEASING.md).
- Modify (docs, final): omna `CLAUDE.md`, `ARCHITECTURE.md`, `README.md`, `CHANGELOG.md`; omna-workspace `CLAUDE.md`, `USER_JOURNEY.md`, `MASTER.md`, `omna-vision.md`, `ARCHITECTURE.md`.

---

## Task 1: pytest hygiene — scope collection to tests/

**Files:** Modify `~/Developer/Omna/pyproject.toml`

- [ ] **Step 1:** Add a pytest config block so `pytest` alone doesn't choke on `scripts/smoke_test.py`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2:** Run `cd ~/Developer/Omna && .venv/bin/python -m pytest -q` (no path) — Expected: 142 passed, no collection error.
- [ ] **Step 3:** Commit `chore(test): scope pytest to tests/ via testpaths`.

## Task 2: Version-consistency test (TDD) + bump to 0.2.0

**Files:** Create `tests/test_packaging.py`; Modify `pyproject.toml`, `omna/__init__.py`

- [ ] **Step 1: Write failing test** in `tests/test_packaging.py`:

```python
"""Packaging invariants that must hold before any PyPI publish."""
import importlib.metadata as md
import tomllib
from pathlib import Path

import omna

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_dunder_version_matches_installed_metadata():
    """omna.__version__ must equal the version pip/PyPI sees."""
    assert omna.__version__ == md.version("omna")


def test_pyproject_version_matches_dunder():
    """The source of truth in pyproject must match the package dunder."""
    assert _pyproject()["project"]["version"] == omna.__version__


def test_not_republishing_an_existing_pypi_version():
    """0.1.0 is already on PyPI; publishing it again fails. Guard the bump."""
    assert omna.__version__ != "0.1.0", "Bump the version — 0.1.0 is taken on PyPI"


def test_pii_extra_depends_on_engine_wheel():
    """pip install omna[pii] must pull the omna-pii-mask engine wheel."""
    extras = _pyproject()["project"]["optional-dependencies"]
    assert any("omna-pii-mask" in d for d in extras["pii"]), \
        "[pii] extra must depend on omna-pii-mask"
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_packaging.py -q` — Expected: FAIL (version still 0.1.0; pii extra empty).
- [ ] **Step 3:** Bump `pyproject.toml` `version = "0.1.0"` → `version = "0.2.0"` and `omna/__init__.py` `__version__ = "0.1.0"` → `"0.2.0"`. (Rationale: 0.1.0 is taken; hybrid search is a new feature → minor bump; engine swap is a major behavioral change. Pre-1.0 minor bump is correct.)
- [ ] **Step 4:** Reinstall editable so installed metadata reflects 0.2.0: `cd ~/Developer/Omna && maturin develop --release` (per CLAUDE.md: always `--release`).
- [ ] **Step 5:** Re-run the test (after Task 3 adds the pii dep) — Expected: PASS.
- [ ] **Step 6:** Commit `feat: bump omna to 0.2.0 (0.1.0 is taken on PyPI) + version-consistency tests`.

## Task 3: Make `[pii]` extra depend on the engine wheel

**Files:** Modify `pyproject.toml:63`, `omna/pii.py:37-44`, `tests/test_import_errors.py` (verify still passes)

- [ ] **Step 1:** Change `pii = []` to:

```toml
# df.omna.mask_pii() / pii_report() route to the compiled omna-pii-mask wheel
# (the unified L1–L6 Rust engine; links ONNX Runtime; L3 runs in-process).
pii = [
    "omna-pii-mask>=0.2,<0.3",
]
```

- [ ] **Step 2:** Update `omna/pii.py` `_require_core()` message to name the extra (PyPI-accurate):

```python
    if not _core_engine_available():
        raise ImportError(
            "Omna's PII engine is not installed. Install it with:  "
            'pip install "omna[pii]"  (pulls the compiled omna-pii-mask wheel — '
            "the unified L1–L6 Rust engine)."
        )
```

- [ ] **Step 3:** Run `.venv/bin/python -m pytest tests/test_import_errors.py tests/test_packaging.py -q` — Expected: PASS (the existing test matches `r"omna_pii_mask"`; new message still contains `omna-pii-mask`). If the regex `r"omna_pii_mask"` no longer matches the new message, relax it to `r"omna[-_]pii[-_]mask"` in the test.
- [ ] **Step 4:** Full suite `.venv/bin/python -m pytest tests/ -q` — Expected: ≥142 passed.
- [ ] **Step 5:** Commit `fix(pkg): [pii] extra now depends on omna-pii-mask; error names the extra`.

## Task 4: Strengthen omna packaging metadata

**Files:** Modify `pyproject.toml`

- [ ] **Step 1:** Add OS + changelog signals. Add to `classifiers`:

```toml
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
```

- [ ] **Step 2:** Add a Changelog URL under `[project.urls]`:

```toml
Changelog  = "https://github.com/gaurjin/omna/blob/main/CHANGELOG.md"
```

- [ ] **Step 3:** Confirm `readme = "README.md"` renders for PyPI (verified by `twine check` in Task 7). No code change unless twine warns.
- [ ] **Step 4:** Commit `chore(pkg): add OS classifiers + changelog url`.

## Task 5: Fix the engine wheel (omna-pii-mask) packaging

**Files:** Modify `~/Developer/omna-workspace/bindings/omna-core-py/pyproject.toml`, `Cargo.toml`; Create `README.md` there.

- [ ] **Step 1:** Determine the engine `version()` source (read `src/` of the binding). Align all three to one clean version. Set binding `pyproject.toml` `version` and `Cargo.toml` `[package].version` to **`0.2.2`** (the engine's current Cargo version). Verify a committed (non-dirty) build makes `omna_pii_mask.version()` return exactly `0.2.2` (no `+local` segment).
- [ ] **Step 2:** Flesh out binding `pyproject.toml` metadata:

```toml
[project]
name = "omna-pii-mask"
version = "0.2.2"
description = "Omna unified PII/secret detection engine (L1–L6) — compiled Rust kernel with in-process on-device AI. Binary-only."
readme = "README.md"
requires-python = ">=3.9"
license = { text = "Proprietary" }
authors = [{ name = "gaurav", email = "gaurjin@gmail.com" }]
keywords = ["pii", "redaction", "privacy", "ner", "secrets", "rust"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: Other/Proprietary License",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3",
    "Programming Language :: Rust",
    "Topic :: Security",
    "Topic :: Text Processing :: Linguistic",
]

[project.urls]
Homepage = "https://github.com/gaurjin/omna"

[tool.maturin]
module-name = "omna_pii_mask"
```

- [ ] **Step 3:** Create `bindings/omna-core-py/README.md` — short, honest description (binary-only engine, what it does, that it's the `[pii]` backend for `omna`). No fabricated numbers.
- [ ] **Step 4:** Commit in omna-workspace `fix(pkg): align omna-pii-mask versions to 0.2.2 + publish metadata`.

## Task 6: Build both wheels locally + clean-venv smoke test (the core verification)

**Files:** none (build/verify); may add `scripts/clean_venv_smoke.sh` for reproducibility.

- [ ] **Step 1:** Build the engine wheel from a clean tree: `cd ~/Developer/omna-workspace/bindings/omna-core-py && maturin build --release --strip --out ~/Developer/Omna/dist-test`. Confirm filename has **no `+` local segment** and is `cp39-abi3`.
- [ ] **Step 2:** Build the omna wheel: `cd ~/Developer/Omna && maturin build --release --strip --out dist-test`. Confirm `omna-0.2.0-*.whl` produced.
- [ ] **Step 3:** Create a brand-new clean venv: `uv venv /tmp/omna-clean --python 3.11`.
- [ ] **Step 4: bare-import discipline** — install ONLY the omna wheel, assert no heavy deps load:

```bash
uv pip install --python /tmp/omna-clean ~/Developer/Omna/dist-test/omna-0.2.0-*.whl
/tmp/omna-clean/bin/python -c "import omna, sys; \
  leaked=[m for m in sys.modules if any(h in m for h in ('fastembed','anthropic','omna_pii_mask','onnxruntime'))]; \
  assert not leaked, leaked; print('bare import clean:', omna.__version__)"
```

Expected: `bare import clean: 0.2.0`.

- [ ] **Step 5: full extras install** — install the engine wheel + `[embed]` deps into the clean venv (simulate `omna[embed,pii]`; the engine wheel stands in for the not-yet-published PyPI dep):

```bash
uv pip install --python /tmp/omna-clean \
  ~/Developer/Omna/dist-test/omna_pii_mask-0.2.2-*.whl \
  ~/Developer/Omna/dist-test/omna-0.2.0-*.whl \
  fastembed numpy onnxruntime
```

- [ ] **Step 6: real hybrid search** — must rank an exact code #1:

```bash
/tmp/omna-clean/bin/python - <<'PY'
import polars as pl, omna
df = pl.DataFrame({"text": ["claim denied for water damage","part XJ9000 backordered","invoice paid in full"]})
df = df.omna.embed("text")                       # builds index (downloads model once)
out = df.omna.search("XJ9000", on="text", k=1)   # hybrid default
assert "XJ9000" in out["text"][0], out["text"].to_list()
print("hybrid search OK:", out["text"][0])
PY
```

Expected: row containing `XJ9000` ranked first.

- [ ] **Step 7: real PII mask, regex layer** (`model=False`):

```bash
/tmp/omna-clean/bin/python - <<'PY'
import polars as pl, omna
df = pl.DataFrame({"note": ["Email john@acme.com, SSN 123-45-6789"]})
m = df.omna.mask_pii(columns=["note"])
assert "john@acme.com" not in m["note"][0] and "123-45-6789" not in m["note"][0], m["note"][0]
print("mask model=False OK:", m["note"][0])
PY
```

Expected: email + SSN redacted to tokens.

- [ ] **Step 8: real PII mask, L3 model** (`model=True`) — reuse the Mac app's model via `OMNA_MODEL_DIR` if present, else allow the one-time download:

```bash
OMNA_MODEL_DIR="${OMNA_MODEL_DIR:-$HOME/Library/Application Support/Omna/models}" \
/tmp/omna-clean/bin/python - <<'PY'
import polars as pl, omna
df = pl.DataFrame({"note": ["Please contact Jane Doe about the overdue balance."]})
m = df.omna.mask_pii(columns=["note"], model=True)
assert "Jane Doe" not in m["note"][0], m["note"][0]
print("mask model=True OK:", m["note"][0])
PY
```

Expected: bare prose name `Jane Doe` redacted (the L3-only win). If the model can't be located/downloaded in the sandbox, record the failure honestly and fall back to verifying `download_model()` is callable + L1/L2 path; do NOT claim success.

- [ ] **Step 9:** Commit `test: clean-venv smoke (bare import, hybrid search, mask model=False/True)` (add the smoke script if created).

## Task 7: twine check + publish dry-run (NO real upload)

**Files:** none.

- [ ] **Step 1:** Install twine into the venv: `uv pip install --python ~/Developer/Omna/.venv twine`.
- [ ] **Step 2:** `~/Developer/Omna/.venv/bin/twine check ~/Developer/Omna/dist-test/*.whl` — Expected: `PASSED` for both wheels (README renders).
- [ ] **Step 3:** TestPyPI: only if a TestPyPI token is in env. It is **not** (verified) → record "skipped: no TestPyPI token" in the runbook. Do NOT prompt.
- [ ] **Step 4:** Never run real `twine upload`. Document the exact command in the runbook (Task 9) as the single remaining step.

## Task 8: Validate the GitHub release workflows

**Files:** review `~/Developer/Omna/.github/workflows/release.yml`; Create `~/Developer/omna-workspace/.github/workflows/release-pypi.yml`.

- [ ] **Step 1:** Lint omna `release.yml` with `actionlint` if available (`brew list actionlint` / `which actionlint`); else manual review. Confirm: matrix covers macOS aarch64+x86_64 × py3.10/3.11/3.12, Linux manylinux x86_64+aarch64; fetches private Rust source via `ENGINE_PAT`; publishes via `PYPI_API_TOKEN`; no sdist. Note any gap (e.g. it builds non-abi3 per-version wheels — fine for omna since it embeds `_omna`).
- [ ] **Step 2:** Author `omna-workspace/.github/workflows/release-pypi.yml` mirroring it: build `omna-pii-mask` **abi3** wheels (one per platform, not per Python) for macOS aarch64+x86_64 and manylinux x86_64+aarch64; upload artifacts; publish job gated on a token (leave the publish step present but document that the user supplies `PYPI_API_TOKEN`). The engine builds entirely within omna-workspace (path deps to `../../core`, `../../l3-ort`) — no external source fetch needed.
- [ ] **Step 3:** Lint the new workflow with actionlint (or manual). Trigger on tag `omna-pii-mask-v*.*.*` to keep it independent of the omna app/lib tags. Do NOT push a tag.
- [ ] **Step 4:** Commit each in its repo.

## Task 9: Publishing runbook

**Files:** Create `~/Developer/Omna/docs/RELEASING-PYPI.md` (mirror `omna-workspace/docs/RELEASING.md`).

- [ ] **Step 1:** Document, in order: (1) publish `omna-pii-mask` FIRST (so `omna[pii]` resolves) — note the proprietary-IP tradeoff of putting the engine binary on public PyPI; (2) then `omna`. Both via tag-triggered CI **or** manual `twine upload`. Include the exact manual commands, the token each needs, the TestPyPI-skipped note, and the post-publish `pip install "omna[embed,pii,ask]"` verification.
- [ ] **Step 2:** Commit `docs: PyPI publishing runbook`.

## Task 10: Self-review loop (per chunk) — REQUIRED

After Tasks 1–4 (omna packaging), after Tasks 5–6 (engine + smoke), and after Tasks 7–9 (dry-run + docs), spawn TWO fresh subagents:

- [ ] **(a) Spec-compliance reviewer:** given this plan + the spec, confirm nothing missing/extra; verify each goal item is met with evidence.
- [ ] **(b) Adversarial reviewer:** try to BREAK it — wrong version, broken/unresolvable dep, platform gap (abi3? manylinux tag?), version skew, README that won't render, a smoke test that passes for the wrong reason, fabricated numbers in docs.
- [ ] Act on findings; re-review until both pass. Record what was found/fixed.

## Task 11: Docs, cleanup, memory, push (the finish)

- [ ] **Step 1:** omna: finalize `CHANGELOG.md` (`[Unreleased]` → `## [0.2.0] - 2026-06-13`, real numbers only, keep before/after tables), update `CLAUDE.md` (build status, version, [pii] dep), `ARCHITECTURE.md`, `README.md` install commands.
- [ ] **Step 2:** omna-workspace: update `CLAUDE.md`, `USER_JOURNEY.md`, `MASTER.md`, `omna-vision.md`, `ARCHITECTURE.md` to reflect omna-pii-mask publish-readiness.
- [ ] **Step 3:** Remove stale/contradictory lines across all touched docs. No hype, no fabricated numbers; verify every number against `benchmarks.json`/`docs/benchmark.md`.
- [ ] **Step 4:** Delete `dist-test/` build scratch; leave the repo clean.
- [ ] **Step 5:** Save lessons to `~/.claude/projects/-Users-gaurav-Developer/memory/` (note: spec path) + update its `MEMORY.md`.
- [ ] **Step 6:** `git push` omna and omna-workspace to their remotes (direct to main allowed). Commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. **Do NOT touch Omna-engine.**

## THE HARD STOP

Do **not** run the real `twine upload` / push a release tag that publishes. Prepare everything so only that one command remains; document it and the token it needs. Everything else (code, builds, tests, dry-runs, commits, pushes) is done autonomously.
