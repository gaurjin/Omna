# Releasing Omna to PyPI — the one runbook

Goal: publish the `omna` Python library so `pip install "omna[embed,pii,ask]"`
works on macOS + Linux, Python 3.10+, with hybrid search and PII masking
(including `model=True`).

## The model (read once)

Omna is **two** distributions on PyPI:

| Distribution | What it is | Wheels | Source dist (sdist)? |
|---|---|---|---|
| `omna` | Python layer + the `_omna` cosine kernel | per-Python (cp310/311/312) × macOS/Linux | **No** — Rust source is private |
| `omna-pii-mask` | the compiled L1–L6 detection engine | **abi3** (one per platform, covers py3.9+) | **No** — proprietary, binary-only |

**Rule — publish order matters:** `omna[pii]` declares `omna-pii-mask>=0.2,<0.3`.
PyPI metadata cannot carry a git URL, so **`omna-pii-mask` must be on PyPI first**
(or in the same sitting) or `pip install "omna[pii]"` / `"omna[all]"` fails to
resolve for everyone.

> ⚠️ **IP decision (engine only):** publishing `omna-pii-mask` puts the compiled
> engine binary on **public** PyPI — anyone can `pip install omna-pii-mask` and
> use it for free. It stays binary-only (no sdist, source closed), but the binary
> itself is downloadable. Decide this is acceptable before uploading the engine.
> Publishing `omna` alone does **not** expose the engine.

## State at time of writing (2026-06-13)

- `omna` **0.1.0 is already on PyPI** (yours). 0.1.0 cannot be re-uploaded — this
  release is **0.2.0**. (`tests/test_packaging.py` queries PyPI and fails if the
  current version is already published.)
- `omna-pii-mask` is **not yet on PyPI** (404) — this will be its first release,
  **0.2.2**.
- **TestPyPI dry-run was skipped**: no TestPyPI token is configured in the
  environment. To do one, set `TWINE_*` for TestPyPI and run the upload with
  `--repository testpypi` (reversible — TestPyPI is a throwaway index).

## Pre-flight (already done, re-runnable)

```bash
cd ~/Developer/Omna
.venv/bin/python -m pytest -q                       # 147 pass
# build both wheels into dist-test/ (gitignored scratch)
maturin build --release --strip --out dist-test     # omna  -> omna-0.2.0-*.whl
(cd ~/Developer/omna-workspace/bindings/omna-core-py && \
   maturin build --release --strip --out ~/Developer/Omna/dist-test)  # engine -> omna_pii_mask-0.2.2-cp39-abi3-*.whl
.venv/bin/twine check dist-test/*.whl               # both PASSED
# end-to-end in a fresh venv (bare import, hybrid search, mask model=False/True):
cd /tmp && /tmp/omna-fresh/bin/python ~/Developer/Omna/scripts/clean_venv_smoke.py
```

## Path A — GitHub Actions (recommended; builds all platforms)

Each repo has a tag-triggered release workflow. CI builds every wheel (you can't
build manylinux locally on a Mac) and publishes via the `PYPI_API_TOKEN` secret
in each repo's `pypi` environment.

1. **Engine first** — in `~/Developer/omna-workspace`:
   ```bash
   git tag omna-pii-mask-v0.2.2 && git push origin omna-pii-mask-v0.2.2
   ```
   (`.github/workflows/release-pypi.yml`: macOS aarch64+x86_64 abi3, manylinux
   x86_64+aarch64 abi3 → publish. Validate the aarch64-linux cell on this first
   run — it cross-builds under QEMU with `ort` downloading an aarch64 runtime.)

2. **Then the library** — in `~/Developer/Omna`:
   ```bash
   git tag v0.2.0 && git push origin v0.2.0
   ```
   (`.github/workflows/release.yml`: fetches the private Rust kernel via
   `ENGINE_PAT`, builds cp310/311/312 × macOS/Linux → publish.)

**Secrets each repo needs** (Settings → Secrets, and Environments → `pypi`):
- both repos: `PYPI_API_TOKEN` (the upload token, in the `pypi` environment).
- `omna` repo only: `ENGINE_PAT` — a PAT that can clone the private
  `gaurjin/Omna-engine`. Without it, every `omna` build job fails at the
  "Fetch Rust source" step. (The engine repo builds from its own in-tree
  source, so it needs no PAT.)

## Path B — manual twine (macOS wheels only)

Use only if you accept macOS-only wheels for this release (Linux users would fall
back to… nothing, since there's no sdist). Prefer Path A.

```bash
cd ~/Developer/Omna
# 1) engine FIRST
.venv/bin/twine upload dist-test/omna_pii_mask-0.2.2-*.whl
# 2) then the library
.venv/bin/twine upload dist-test/omna-0.2.0-*.whl
```

`twine upload` needs a **real PyPI API token** (`__token__` / `pypi-...`), via
`~/.pypirc` or `TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-...`. **This is the
single irreversible step — it permanently claims the version.**

## Verify after publishing

```bash
python -m venv /tmp/verify && /tmp/verify/bin/pip install "omna[embed,pii,ask]"
cd /tmp && /tmp/verify/bin/python ~/Developer/Omna/scripts/clean_venv_smoke.py
```

That's the whole loop: **bump → build → twine check → engine first → then omna → verify.**
