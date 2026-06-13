# Changelog

All notable changes to Omna are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Omna aims for
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

These changes ship with the `omna-core` detection-engine wheel (currently built
from `omna-workspace`; not yet on PyPI). The pure-Python `omna` package API is
unchanged and backward-compatible.

### Removed

- **Microsoft Presidio + spaCy are gone.** `presidio-analyzer`,
  `presidio-anonymizer`, the spaCy model download, and the `engine="presidio"`
  / `fast=` options were all removed. Their value was extracted first —
  Presidio's detection rules were ported into the engine's L1 layer and its
  spaCy NER was replaced by the L3 model — so there is no loss of capability,
  just a heavy redundant dependency deleted. `mask_pii()` / `pii_report()` now
  have no Python ML dependencies.

### Changed

- **PII detection engine rebuilt as Omna's own six-layer pipeline.** Masking is
  a self-contained Rust engine — no Presidio, no spaCy. The **same Rust
  engine** runs in the Python library, the Omna Mac
  app, and the browser extension — output is byte-for-byte identical across all
  three (verified by a parity gate on every change).
  - **Core-PII recall** (name / email / SSN / phone / card) is **0.77**,
    measured on the Gretel PII benchmark — above the previous Presidio-based
    path's 0.69 for the first time.
  - **Leak rate** (share of sensitive values that slip through) is **2–6%** on
    the Gretel benchmark.
  - Detection is checksum-validated (Luhn, IBAN mod-97, Verhoeff, NHS, and 30+
    international ID schemes) and includes **220+ secret-detection rules** (AWS
    keys, GitHub tokens, JWTs, …) with entropy checks. Secrets are always
    redacted irreversibly and never written to the reversible token map.
- **Semantic search model upgraded** from `AllMiniLML6V2` (384-dim) to
  `nomic-embed-text-v1.5` (768-dim). On an internal 60-document / 18-query
  retrieval benchmark, recall@10 improved from **0.93 → 1.00** — most of the
  gain is on queries that share no keywords with the matching text.

### Added

- **`mask_pii(model=True)`** — opt-in on-device AI layer (L3) that catches
  contextual PII regex cannot, such as a bare name with no title or label
  ("Contact Jane Doe …"). The model (~809 MB) downloads once on first use and
  is cached; `download_model()` pre-fetches it; set `OMNA_MODEL_DIR` to point at
  an existing copy (e.g. the Mac app's). Without the model, L1+L2 run instantly
  and fully offline.
- **`download_model()`** — pre-download the L3 model so the first masking call
  is instant (useful for CI / enterprise pre-staging).

### Internal

- The L3 model now links **in-process** into every product (Python wheel, Mac
  app) — the previous standalone helper process was removed after the embedding
  stack was upgraded to a single shared ONNX Runtime.

## [0.1.0]

- Initial public release: semantic search, threshold filtering, schema
  understanding, PII audit (`pii_report`) and masking (`mask_pii`), and natural-
  language `ask()` — all on Polars DataFrames, running locally. PII masking in
  this release was Presidio-backed (see `docs/benchmark.md`).
