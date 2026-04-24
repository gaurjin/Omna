# Omna — Architecture Reference

## What Omna is
Hybrid Python + Rust library. Polars DataFrame namespace plugin.
Adds semantic search, PII masking, and schema understanding.

## The full API
```
omna.understand(df)              # schema inference, column labelling
df.omna.embed("column")          # vectorize a column, save to disk
df.omna.search("query", on, k)   # semantic search — Rust kernel inside
df.omna.filter("concept", on)    # filter by concept, not exact string
df.omna.mask_pii()               # auto-detect and redact PII, audit log
df.omna.pii_report()             # show PII found without masking yet
df.omna.ask("question")          # natural language query via LLM
```

## Tech stack
- Polars namespace plugin (Python)
- FastEmbed — local embeddings, no API key
- Presidio — PII detection, Microsoft open source
- Rust src/similarity.rs — cosine similarity kernel, rayon parallel
- maturin — builds Rust-Python wheels
- uv — Python package manager

---

## Two-repo structure

Omna is deliberately split across two directories and two GitHub repos.
Understanding why is essential before touching any Rust code.

### ~/Developer/Omna  (public repo: github.com/gaurjin/Omna)

This is the **working repo** — where all development happens.

**What it contains:**
- `omna/` — the full Python package (frame.py, embedder.py, index.py, pii.py, understand.py, ask.py, __init__.py)
- `src/` — Rust source (lib.rs, similarity.rs) — **excluded from git via .gitignore**
- `Cargo.toml` / `Cargo.lock` — Rust build config — **excluded from git**
- `tests/` — pytest suite
- `scripts/` — benchmark.py, smoke_test.py, generate_demo_data.py
- `pyproject.toml`, `uv.lock` — Python packaging

**What gets pushed to GitHub:**
Only the Python layer. The `.gitignore` excludes `src/`, `Cargo.toml`, `Cargo.lock`, and `target/`. A developer who clones this repo gets only the Python code, not the Rust source.

**Why src/ lives here:**
The Rust code must be present to build the wheel with maturin. `maturin develop --release` reads `src/lib.rs` and `Cargo.toml` to compile `_omna.cpython-311-darwin.so` and install it into `.venv/`. Every change to the Rust kernel is made and tested here first.

---

### ~/Developer/Omna-engine  (private repo: github.com/gaurjin/Omna-engine)

This is the **IP archive** — source of truth for the proprietary Rust code.

**What it contains:**
- `src/lib.rs` — PyO3 bindings
- `src/similarity.rs` — cosine similarity kernel
- `Cargo.toml` — Rust dependencies
- `Cargo.lock` — locked dependency tree
- `README.md` — internal notes

**What it does NOT contain:**
- Python code (lives in the public repo)
- Tests (live in the public repo)
- Wheels or compiled binaries

**Why this repo exists:**
The Rust source is the proprietary core of Omna. It must never appear in the public GitHub repo. Omna-engine is a private mirror that lets the Rust source be version-controlled separately, audited, and eventually used for commercial licensing or enterprise distribution. The compiled `.so` ships inside the pip wheel; the source stays in this private repo.

---

## Which files live where

| File | Omna (public) | Omna-engine (private) | Notes |
|---|---|---|---|
| `omna/__init__.py` | ✓ | — | Python only |
| `omna/frame.py` | ✓ | — | Python only |
| `omna/embedder.py` | ✓ | — | Python only |
| `omna/index.py` | ✓ | — | Python only |
| `omna/pii.py` | ✓ | — | Python only |
| `omna/understand.py` | ✓ | — | Python only |
| `omna/ask.py` | ✓ | — | Python only |
| `src/lib.rs` | ✓ (local only, .gitignored) | ✓ (committed) | Must stay in sync |
| `src/similarity.rs` | ✓ (local only, .gitignored) | ✓ (committed) | Must stay in sync |
| `Cargo.toml` | ✓ (local only, .gitignored) | ✓ (committed) | Must stay in sync |
| `Cargo.lock` | ✓ (local only, .gitignored) | ✓ (committed) | Must stay in sync |
| `pyproject.toml` | ✓ | — | Python packaging |
| `tests/` | ✓ | — | Python tests |
| `scripts/` | ✓ | — | Python scripts |

---

## The sync rule — CRITICAL

**Any Rust change must be made in Omna/src/ first, then copied to Omna-engine/src/.**

The Omna working directory is where you develop, compile, and test. After any change to Rust files, copy them to the engine repo and commit both.

Step-by-step after a Rust change:

```bash
# 1. Make the change in Omna/src/ and verify it works
cd ~/Developer/Omna
# ... edit src/lib.rs or src/similarity.rs ...
maturin develop --release
uv run pytest tests/ -q

# 2. Copy changed files to Omna-engine
cp src/lib.rs ~/Developer/Omna-engine/src/
cp src/similarity.rs ~/Developer/Omna-engine/src/
cp Cargo.toml ~/Developer/Omna-engine/Cargo.toml
cp Cargo.lock ~/Developer/Omna-engine/Cargo.lock

# 3. Commit to the private engine repo
cd ~/Developer/Omna-engine
git add src/ Cargo.toml Cargo.lock
git commit -m "sync: ..."
git push
```

Never edit files in Omna-engine directly. Always develop in Omna, then sync.

---

## Current sync status

As of the benchmark optimisation session (April 2026), Omna-engine is **out of sync**.
The following changes in Omna/src/ have NOT yet been copied to Omna-engine:

| Change | Omna/src/ | Omna-engine/src/ |
|---|---|---|
| `top_k_flat` PyO3 binding | ✓ | ✗ |
| `top_k_flat_np` (zero-copy numpy buffer) | ✓ | ✗ |
| `rayon` parallelism in `top_k_flat` | ✓ | ✗ |
| `select_nth_unstable_by` partial sort | ✓ | ✗ |
| `rayon = "1"` in Cargo.toml | ✓ | ✗ |

Run the copy commands above before the next release.

---

## Machine
MacBook Air M5
User: gaurav
Project path: ~/Developer/Omna

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
