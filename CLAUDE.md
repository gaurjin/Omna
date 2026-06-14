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
- Phase E: Announce — X/Twitter post, Hacker News, Python communities

## 2026-06-10 — install-story + accuracy fixes (ship as v0.1.1)
- [x] README documents extras: `pip install "omna[all]"` (bare install was crashing the quick start — fastembed/presidio are optional extras the README never mentioned)
- [x] Friendly ImportErrors in embedder.py + pii.py naming the extra to install (mirrors ask.py); 2 new tests → **115 passing**
- [x] faker moved from runtime dependencies to the dev extra (only used by scripts/generate_demo_data.py)
- [x] README accuracy: Polars 0.20+ → 1.0+ (matches pyproject); "23 lines" → "under 70 lines"; PII blurb "all gone" → "redacted" + open benchmark link (own Gretel benchmark: core-PII recall 0.692 — see benchmarks.json)
- [x] First Rust kernel unit suite (7 tests in src/similarity.rs `#[cfg(test)]`), synced to Omna-engine per the sync rule
- [ ] Publish v0.1.1 to PyPI (bundle with the PII-engine upgrade: swap to OpenAI privacy-filter ONNX model, hybrid with regex fast path; acceptance gate = Gretel harness core-PII recall ≥ 0.95)

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
