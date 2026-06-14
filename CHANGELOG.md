# Changelog

All notable changes to Omna are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Omna aims for
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-06-13

These changes ship with the `omna-pii-mask` detection-engine wheel, now prepared
for PyPI alongside `omna` (`pip install "omna[pii]"` pulls it automatically). The
`omna` public API is backward-compatible.

Every change below is shown as **before → after** so you can see what's
different at a glance.

### Added

- **Hybrid search — keyword matching now runs alongside meaning-based search.**
  `df.omna.search()` used to match purely on *meaning*, so a rare exact term
  could rank low or be missed. Now it matches *keywords* too and merges both.

  | Your search | Before (semantic only) | After (hybrid, default) |
  |---|---|---|
  | `"claim denied"` (by meaning) | finds related docs ✅ | finds related docs ✅ |
  | `"XJ9000"` (an exact part code) | often buried or missed ⚠️ | **ranked #1** ✅ |
  | New dependency to install | — | none |
  | Re-index existing data? | — | no — existing indexes just work |

  On by default; `hybrid=False` restores pure-semantic. (Under the hood: BM25
  keyword scoring fused with semantic ranking via Reciprocal Rank Fusion;
  `_score` is still the cosine similarity.)

- **On-device AI for PII — `mask_pii(model=True)`.** A new optional AI layer (L3)
  catches contextual PII that regex cannot — like a bare name with no title.

  | Detecting… | Before (regex only) | After (`model=True`) |
  |---|---|---|
  | `SSN 123-45-6789`, emails, cards | caught ✅ | caught ✅ |
  | `"Contact Jane Doe about it"` (bare name) | missed ⚠️ | **caught** ✅ |
  | Setup | none | model auto-downloads once (~809 MB), or `download_model()` to pre-fetch / `OMNA_MODEL_DIR` to reuse the Mac app's |
  | Default (`model=False`) | regex, instant, offline | regex, instant, offline — unchanged |

### Changed

- **PII engine — Presidio wrapper → Omna's own six-layer Rust engine.** Same
  engine now runs in the Python library, the Mac app, and the browser extension
  (byte-identical output, parity-gated).

  | | Before (v0.1.0) | After |
  |---|---|---|
  | Detection engine | Microsoft Presidio + spaCy | Omna 6-layer Rust engine |
  | Core-PII recall (Gretel) | 0.69 | **0.84** (with `model=True`) |
  | Secret detection | — | **220+ rules** (AWS, GitHub, JWT…) + entropy |
  | ID validation | basic patterns | **checksum-validated** (Luhn, IBAN, Verhoeff, NHS, 30+ intl) |
  | Heavy Python ML deps | presidio-analyzer/-anonymizer, spaCy model | **none** |

  (All-types with the model: precision 0.84 / recall 0.79 / F1 0.815 on the same
  benchmark — see `docs/benchmark.md`. Secrets are always redacted irreversibly
  and never written to the reversible token map.)

- **Embedding model upgraded.**

  | | Before | After |
  |---|---|---|
  | Model | `BAAI/bge-small-en-v1.5` (384-dim) | `nomic-embed-text-v1.5` (768-dim) |
  | Queries with no shared keywords | weaker | better recall |
  | Consistency with the Mac app | different model | **same model** |

  **Action required:** rebuild any saved index — re-run `df.omna.embed(column)`
  (old 384-dim and new 768-dim vectors are not comparable).

### Removed

- **Microsoft Presidio + spaCy (and the `engine="presidio"` / `fast=` options).**

  | | Before | After |
  |---|---|---|
  | PII detection path | Presidio + spaCy NER (Python ML deps) | the unified Rust engine |
  | Capability lost? | — | **none** — Presidio's rules were ported into L1, its NER replaced by the L3 model |

### Internal

- **L3 model now runs in-process** (Python wheel and Mac app).

  | | Before | After |
  |---|---|---|
  | L3 model runs as | a separate helper process | **in-process**, on one shared ONNX Runtime |

### Fixed

- **Bare `pip install omna` no longer crashes on import.** `numpy` and `rich`
  are imported at `import omna` time but were undeclared as core dependencies.

  | | Before | After |
  |---|---|---|
  | `import omna` after bare `pip install omna` | `ModuleNotFoundError: numpy` | imports cleanly |
  | Core dependencies | `polars` only | `polars`, `numpy`, `rich` |
  | Heavy deps at bare import (fastembed/onnxruntime/engine) | none | none — still lazy |

- **`ask()` no longer dead-ends on a partial install.** `ask()` masks the
  sampled rows before sending them to the API by default, which needs the
  engine — but `omna[ask]` didn't install it, and `df.omna.ask()` didn't expose
  the documented `mask_rows=False` escape hatch.

  | | Before | After |
  |---|---|---|
  | `omna[ask]` install | `anthropic` only → default `ask()` errored, no engine | also pulls `omna-pii-mask` → masks out of the box |
  | `df.omna.ask(...)` | `(question, model)` — no way to skip masking | adds `mask_rows=False` (synthetic/public data only) |

### Packaging

- **`pip install "omna[pii]"` now installs the PII engine.** The `[pii]` extra
  was empty, so masking failed on a clean install.

  | | Before | After |
  |---|---|---|
  | `[pii]` extra | empty (`pii = []`) | depends on `omna-pii-mask>=0.2,<0.3` |
  | `[ask]` extra | `anthropic` only | also `omna-pii-mask` (ask masks by default) |
  | Engine availability on PyPI | manual wheel from `omna-workspace` | published wheel, auto-resolved |

## [0.1.0]

- Initial public release: semantic search, threshold filtering, schema
  understanding, PII audit (`pii_report`) and masking (`mask_pii`), and natural-
  language `ask()` — all on Polars DataFrames, running locally. PII masking in
  this release was Presidio-backed (see `docs/benchmark.md`).
