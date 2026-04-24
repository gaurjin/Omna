# Omna — Architecture Reference

## What Omna is
Hybrid Python + Rust library. Polars DataFrame namespace plugin.
Adds semantic search, PII masking, and schema understanding.

## The full API
omna.understand(df)              # schema inference, column labelling
df.omna.embed("column")          # vectorize a column, save to disk
df.omna.search("query", on, k)   # semantic search — Rust kernel inside
df.omna.filter("concept", on)    # filter by concept, not exact string
df.omna.mask_pii()               # auto-detect and redact PII, audit log
df.omna.pii_report()             # show PII found without masking yet
df.omna.ask("question")          # natural language query via LLM

## Tech stack
- Polars namespace plugin (Python)
- FastEmbed — local embeddings, no API key
- Presidio — PII detection, Microsoft open source
- Rust src/similarity.rs — cosine similarity kernel only
- maturin — builds Rust-Python wheels
- uv — Python package manager

## Machine
MacBook Air M5
User: gaurav
Project path: ~/Developer/omna

## Repositories
- Public (Python layer): https://github.com/gaurjin/Omna
- Private (Rust engine): https://github.com/gaurjin/Omna-engine
- PyPI account: gaurjin (created, not yet published)
- First commit: 6d5234a

## Current status
All 7 build days complete. 112 tests passing.
README written. Lazy loading complete.
Multi-platform wheel workflow: .github/workflows/release.yml

**Launch gate — do NOT run `git tag v0.1.0` until:**
1. Website is live
2. Launch plan is ready

## 7-day build plan
Day 0 — Install Rust, uv, maturin, Claude Code ✓
Day 1 — maturin scaffold + Polars namespace ✓
Day 2 — Rust similarity kernel ✓
Day 3 — embedder.py + index.py ✓
Day 4 — search() + filter() ✓
Day 5 — mask_pii() + audit log ✓
Day 6 — understand() + ask() ✓
Day 7 — wheels, tests, README, lazy loading, release.yml ✓

## Differentiators
1. PII guard as first-class feature
2. Polars-native — no separate vector DB
3. Works offline — no API key needed
4. Index persistence — embed once reuse forever
5. Combines understand() with search and compliance

## Fundraising angle
Rust-core similarity engine. Polars-native.
Built for regulated industries that cannot feed
raw PII to cloud AI.

## Phase 2 (after 3,000 GitHub stars)
- df1.omna.join(df2, on="description") — semantic joins
- Match rows between two DataFrames by meaning not exact key
- Example: match transaction descriptions against regulatory categories
- Biggest missing feature across all 47 existing semantic libraries
- This is the second viral moment after launch

## Post-launch integration targets
- OpenBB (66K stars) — open source Bloomberg, full of financial DataFrames, no semantic layer
- FinGPT — financial LLMs, Omna feeds them clean PII-masked data
- Polars Discord — join before launch, post benchmark on Day 7
- Graphify — installed in omna/ folder, 71x fewer tokens per Claude Code session

## License model
- Python layer (omna/ package): MIT — public, https://github.com/gaurjin/Omna
- Rust engine (src/): Proprietary — private, https://github.com/gaurjin/Omna-engine
  - Contains: src/lib.rs, src/similarity.rs
  - Never published as source — ships as compiled binary inside pip wheel
  - Source lives in private repo only
  - Omna-engine repo is live and confirmed private
  - First commit 81cae3d pushed April 23 2026
  - src/ protected from public repo via .gitignore entry
