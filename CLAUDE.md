# Omna — CLAUDE.md

## What this project is
Omna is a hybrid Python + Rust library that adds semantic
search, PII masking, and schema understanding to Polars
DataFrames. It installs as a Polars namespace plugin so
developers can call df.omna.search(), df.omna.mask_pii(),
and omna.understand(df).

## Tech stack
- Python: Polars namespace, FastEmbed, index persistence
- PII: the compiled `omna_pii_mask` wheel (unified L1–L6 Rust engine — no heavy
  Python ML deps). `omna/pii.py` routes to it.
- Rust: cosine similarity kernel only (src/similarity.rs)
- Build tool: maturin
- Package manager: uv

## File structure
omna/
├── src/lib.rs             # Rust entry point
├── src/similarity.rs      # Rust cosine similarity kernel
├── omna/__init__.py       # Registers df.omna namespace
├── omna/frame.py          # All df.omna.* public methods
├── omna/embedder.py       # FastEmbed wrapper
├── omna/index.py          # Save/load embeddings (Parquet)
├── omna/hybrid.py         # BM25 + RRF for hybrid search (pure Python/numpy)
├── omna/pii.py            # PII detection + masking via the omna_pii_mask engine
├── omna/understand.py     # Schema inference
└── omna/ask.py            # LLM query layer

## Two-repo structure
Omna is split across two directories:
- `~/Developer/Omna` — working repo. All development happens here. src/ is .gitignored.
- `~/Developer/Omna-engine` — private IP archive (github.com/gaurjin/Omna-engine). Rust source only.

The compiled .so ships in the pip wheel. The Rust source never appears in the public repo.

## Rust sync rule — ALWAYS DO THIS
Any change to src/lib.rs, src/similarity.rs, or Cargo.toml must be copied to Omna-engine after testing:
```
cp src/lib.rs src/similarity.rs ~/Developer/Omna-engine/src/
cp Cargo.toml Cargo.lock ~/Developer/Omna-engine/
cd ~/Developer/Omna-engine && git add -A && git commit -m "sync: <description>" && git push
```
Develop in Omna first. Sync to Omna-engine after every Rust change. Never edit Omna-engine directly.

## Coding rules
- Python first. Only write Rust in src/similarity.rs.
- Every public method must have a docstring.
- Never break the df.omna namespace interface.
- Always run `maturin develop --release` after any Rust change (never plain `maturin develop` — debug builds are 10x slower).
- Tests live in tests/ and use pytest.

## Repositories
- Public (Python layer): https://github.com/gaurjin/Omna
- Private (Rust engine): https://github.com/gaurjin/Omna-engine
- PyPI account: gaurjin — published! pypi.org/project/omna (v0.1.0, 2026-04-26)

## Current build status
[x] Day 0 — tools installed
[x] Day 1 — Foundation complete
[x] Day 2 — Rust kernel complete
[x] Day 3 — Embedder + index complete
[x] Day 4 — Search + filter complete
[x] Day 5 — PII guard complete
[x] Day 6 — Understand + ask complete
[x] Day 7 — Build complete (113 tests passing, README written, lazy loading done, release.yml created)
[x] Smoke test passed 7/7 on real data (scripts/smoke_test.py)
[x] ANTHROPIC_API_KEY permanently saved to ~/.zshrc
[x] Omna-engine private repo live: github.com/gaurjin/Omna-engine (commit 81cae3d)
[x] src/ excluded from public repo via .gitignore
[x] embedder.py memory fix: batch_size=32, chunk_size=2000, gc.collect() per chunk, CoreML warmup
[x] Rich output for all 6 methods (search, filter, pii_report, mask_pii, ask — consistent with understand())
[x] PII false-positive fix: _REAL_PII_TYPES allow-list, hit-rate threshold (>10%), unique-hit guard
[x] mask_pii() XXXX-skip fix: government pre-redacted tokens no longer double-redacted
[x] mask_pii() fast=True mode: regex-only, catches email/phone/SSN/URL
[x] mask_pii() full mode optimised: spaCy NER-only pipeline + dedup + single pool
[x] Demo dataset: Gretel PII Benchmark (acquired by NVIDIA) — 50,000 synthetic documents
[x] demo_shield.py recorded — PII audit + redaction story (assets/demo_shield.gif)
[x] demo_sword.py recorded — semantic search + filter + ask story (assets/demo_sword.gif)
[x] README rewritten — search-first story, Gretel/NVIDIA dataset, real GIFs embedded
[x] 113 tests passing

## Next steps
- [x] Phase C: Website live at omna.dev (2026-04-26)
- [x] Phase D: PyPI publish — v0.1.0 live at pypi.org/project/omna (2026-04-26)
- [x] Phase B: PyPI publish-readiness for v0.2.0 (2026-06-13) — see below
- Phase E: Announce — X/Twitter post, Hacker News, Python communities

## 2026-06-13 — Phase B: PyPI publish-readiness (v0.2.0) — DONE (uploaded 2026-06-14)
- [x] Bumped to **0.2.0** (0.1.0 is already on PyPI; re-upload fails). `__version__` + pyproject kept in lockstep, guarded by tests/test_packaging.py (which also queries live PyPI).
- [x] **Fixed a real bare-install bug**: `numpy` + `rich` are imported at `import omna` time but were undeclared core deps → `pip install omna` crashed on import. Now core deps = `polars, numpy, rich`. Guard test added. (Caught by the clean-venv smoke; existing import-speed test missed it.)
- [x] `[pii]` extra now depends on `omna-pii-mask>=0.2,<0.3` (was empty). pii.py error names `pip install "omna[pii]"`.
- [x] **omna-pii-mask engine prepared for PyPI**: dist version aligned to 0.2.2 (was 0.2.0/Cargo 0.2.2 skew), full metadata + README, self-contained abi3 wheel (no Requires-Dist, ORT statically linked). New `omna-workspace/.github/workflows/release-pypi.yml` (abi3 macOS+manylinux). 404 on PyPI = name free.
- [x] Both wheels built + `twine check` PASSED. Clean-venv smoke (scripts/clean_venv_smoke.py): bare import (no heavy deps), hybrid search (exact code #1), mask model=False AND model=True (L3 bare-name redaction) all pass. **156 tests passing.**
- [x] **Edge-case hardening (2026-06-14):** found + fixed a real `ask()` dead-end — it masks rows before the API by default (needs the engine) but `omna[ask]` didn't pull the engine and `df.omna.ask()` didn't expose the documented `mask_rows=False` escape (README documented a param that didn't exist). Fix: `[ask]` → engine; `df.omna.ask(..., mask_rows=…)`. Also verified across Python 3.10/3.11/3.12 (abi3 engine loads on all), `_score` stays cosine in hybrid mode, re-embed doesn't serve a stale index, secrets are never restorable (privacy invariant), and `omna[embed,pii,ask]==0.2.0` resolves end-to-end. New tests/test_edge_cases.py (+7).
- [x] Runbook: docs/RELEASING-PYPI.md (engine-first publish order, manual + CI paths, ENGINE_PAT/PYPI_API_TOKEN secrets).
- [x] **THE UPLOAD — DONE 2026-06-14.** Both packages live on PyPI: `omna` 0.2.2 and `omna-pii-mask` 0.2.2 (engine published first per the runbook). Website omna.dev live (HTTP 200). Verified via live PyPI JSON API.
- Superseded the old "v0.1.1" plan: the PII-engine upgrade already shipped in-tree (6-layer Rust engine, L3 in-process); this releases it as 0.2.0.

## 2026-06-10 — install-story + accuracy fixes (ship as v0.1.1)
- [x] README documents extras: `pip install "omna[all]"` (bare install was crashing the quick start — fastembed/presidio are optional extras the README never mentioned)
- [x] Friendly ImportErrors in embedder.py + pii.py naming the extra to install (mirrors ask.py); 2 new tests → **115 passing**
- [x] faker moved from runtime dependencies to the dev extra (only used by scripts/generate_demo_data.py)
- [x] README accuracy: Polars 0.20+ → 1.0+ (matches pyproject); "23 lines" → "under 70 lines"; PII blurb "all gone" → "redacted" + open benchmark link (own Gretel benchmark: core-PII recall 0.692 — see benchmarks.json)
- [x] First Rust kernel unit suite (7 tests in src/similarity.rs `#[cfg(test)]`), synced to Omna-engine per the sync rule
- [x] ~~Publish v0.1.1 to PyPI~~ → superseded by **v0.2.0** (the PII-engine upgrade — OpenAI privacy-filter ONNX as L3, in-process — has shipped in-tree; released as 0.2.0). See the 2026-06-13 Phase B section above.

## Demo dataset
- data/gretel_pii.csv — Gretel PII Benchmark, 50,000 synthetic documents
- Search index: .omna/text.parquet (built with df.omna.embed("text"))
- Demo scripts: scripts/demo_shield.py, scripts/demo_sword.py

## Git
- First commit: 6d5234a (main branch)

## Communication rules
- Spoon-feed every step. Never assume knowledge.
- Explain every terminal command in plain English.
- User is intermediate Python, no Rust experience.
