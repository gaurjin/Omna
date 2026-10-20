# Omna — CLAUDE.md

## What this project is
Omna is a hybrid Python + Rust library that adds semantic
search, PII masking, and schema understanding to Polars
DataFrames. It installs as a Polars namespace plugin so
developers can call df.omna.search(), df.omna.mask_pii(),
and omna.understand(df).

## Two-repo structure
Omna is split across two directories:
- `~/Developer/Omna Labs/Omna` — working repo. All development happens here. src/ is .gitignored.
- `~/Developer/Omna Labs/Omna-engine` — private IP archive (github.com/gaurjin/Omna-engine). Rust source only.

The compiled .so ships in the pip wheel. The Rust source never appears in the public repo.

## Rust sync rule — ALWAYS DO THIS
Any change to src/lib.rs, src/similarity.rs, or Cargo.toml must be copied to Omna-engine after testing:
```
cp src/lib.rs src/similarity.rs ~/Developer/Omna\ Labs/Omna-engine/src/
cp Cargo.toml Cargo.lock ~/Developer/Omna\ Labs/Omna-engine/
cd ~/Developer/Omna\ Labs/Omna-engine && git add -A && git commit -m "sync: <description>" && git push
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

## Next steps
- Phase E: Announce — X/Twitter post, Hacker News, Python communities

## 2026-07-07 — Community + Claude-for-OSS readiness (docs only, no code)
Applying for the Claude for OSS program (6 months Claude Max for OSS maintainers).
Repo-surface + community work to strengthen the application — no code/API change:
- [x] Filled the blank `LICENSE` copyright holder → "The Omna Authors".
- [x] Added `CONTRIBUTING.md` (bug reports, setup, tests, PR flow) — grows the
      external-contributor count (a program eligibility door).
- [x] Opened a community RFC: **issue #3** "scale local search toward a billion
      rows — simplest idea wins" (labels: help wanted / enhancement / question),
      to pull in outside contributors.
- [x] Added `docs/scaling-to-a-billion-rows.md` (the three walls: embed-time,
      ~3 TB storage, brute-force O(N) scan) + a README "Help wanted" section
      linking both.
- [x] Synced stale `uv.lock` to 0.2.2; committed the mask-rate benchmark script.
- Application draft (narrative + facts) kept in the session scratchpad, not the repo.
- NOTE: the `done` skill is wired to omna-workspace (MASTER.md); this repo tracks
  state in CLAUDE.md + CHANGELOG.md, so the wrap-up pattern was applied here instead.

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
