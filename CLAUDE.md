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
[x] Smoke test passed 7/7 on real data (scripts/smoke_test.py)
[x] scripts/demo_data.py and scripts/smoke_test.py created
[x] ANTHROPIC_API_KEY permanently saved to ~/.zshrc
[x] Omna-engine private repo live: github.com/gaurjin/Omna-engine (commit 81cae3d)
[x] src/ excluded from public repo via .gitignore

## Next steps
- Phase B: record demo GIF
- Phase C: PyPI publish
- Do NOT run `git tag v0.1.0` until demo is recorded and website is live

## Git
- First commit: 6d5234a (main branch)

## Communication rules
- Spoon-feed every step. Never assume knowledge.
- Explain every terminal command in plain English.
- User is intermediate Python, no Rust experience.
