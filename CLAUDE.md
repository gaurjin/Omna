# Omna — CLAUDE.md

## What this project is
Omna is a hybrid Python + Rust library that adds semantic
search, PII masking, and schema understanding to Polars
DataFrames. It installs as a Polars namespace plugin so
developers can call df.omna.search(), df.omna.mask_pii(),
and omna.understand(df).

## Tech stack
- Python: Polars namespace, FastEmbed, Presidio,
  index persistence
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
├── omna/pii.py            # Presidio PII detection + audit
├── omna/understand.py     # Schema inference
└── omna/ask.py            # LLM query layer

## Coding rules
- Python first. Only write Rust in src/similarity.rs.
- Every public method must have a docstring.
- Never break the df.omna namespace interface.
- Always run maturin develop after any Rust change.
- Tests live in tests/ and use pytest.

## Repositories
- Public (Python layer): https://github.com/gaurjin/Omna
- Private (Rust engine): https://github.com/gaurjin/Omna-engine
- PyPI account: gaurjin (created, not yet published)

## Current build status
[x] Day 0 — tools installed
[x] Day 1 — Foundation complete
[x] Day 2 — Rust kernel complete
[x] Day 3 — Embedder + index complete
[x] Day 4 — Search + filter complete
[x] Day 5 — PII guard complete
[x] Day 6 — Understand + ask complete
[x] Day 7 — Build complete (112 tests passing, README written, lazy loading done, release.yml created)
[ ] PyPI publish — BLOCKED: do not run `git tag v0.1.0` until website is live and launch plan is ready

## Git
- First commit: 6d5234a (main branch)

## Communication rules
- Spoon-feed every step. Never assume knowledge.
- Explain every terminal command in plain English.
- User is intermediate Python, no Rust experience.
