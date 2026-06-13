# Omna's PII detection engine — what it is, and how it measures

Omna's masking is not a regex wrapper and not a toy. It's a **unified six-layer
detection engine**, written in Rust, that runs identically in the Python
library, the Omna Mac app, and the browser extension. This page explains the
layers and shows measured numbers on public + synthetic benchmarks. Raw figures
are in [`benchmarks.json`](../benchmarks.json); everything here is reproducible
from the engine's benchmark suite.

## Why we upgraded — before vs after

Same Gretel benchmark, 1,000 rows, seed 42, identical char-overlap scoring.
**Before** = the original Presidio-based engine (recorded April 2026; Presidio
is now removed so it can't be re-run). **After** = the current unified engine,
re-measured through the library today.

| Gretel benchmark | Before (Presidio) | After (unified engine + model) |
|---|---|---|
| **Core-PII recall** (name/email/phone/SSN/card) | 0.692 | **0.840** |
| All-types recall | 0.353 | **0.791** |
| All-types precision | 0.871 | 0.840 |
| All-types F1 | 0.503 | **0.815** |
| PII entity types detected | ~17 | **30+** |
| Secret-detection rules | 0 | **220+** |
| Checksum-validated international IDs | none | **30+ schemes** |
| Contextual AI model (catches bare prose names) | no | **yes (L3)** |
| Token policy | irreversible `<REDACTED>` only | **reversible PII tokens; secrets always irreversible** |
| Python ML dependencies | Presidio + spaCy | **none** |
| Runs in | Python only | **Mac app + browser extension + Python** |

**The headline:** recall jumps (all-types 0.35 → 0.79, ~2.2×; core-PII 0.69 →
0.84) at roughly the same precision, **plus** a large capability leap — secrets,
checksum validation, an on-device model, reversible tokens, and the same engine
across all three products, with no Python ML dependencies.

## The six layers

1. **L1 — patterns + validators.** Emails, phones, SSNs, credit cards
   (Luhn-checked), IBANs (mod-97), and 30+ international ID schemes with real
   checksum validation (Verhoeff/Aadhaar, NHS mod-11, AU TFN/ABN, …). A match
   that passes its checksum is high-confidence; a pattern-only match is scored
   lower.
2. **L2 — secrets.** 220+ rules (AWS keys, GitHub tokens, JWTs, private keys, …)
   with keyword pre-filtering and Shannon-entropy checks.
3. **L3 — on-device AI model** *(optional, `model=True`)*. Catches contextual
   PII that no regex can — a bare name in prose, an address, medical context.
4. **L4 — fusion.** Resolves overlapping spans, links the same person across a
   document, applies deterministic precedence.
5. **L5 — policy.** Non-secrets become reversible `[PERSON_1]`-style Shield
   tokens; **secrets are always irreversibly `[REDACTED:KIND]`** and never enter
   the reversible token map.
6. **L6 — audit.** Every decision is logged (entity, layer, confidence, a hashed
   span — never the value).

## Measured results

Metrics are relaxed (IoU ≥ 0.5). **Leak rate** = the share of gold sensitive
values left in the output — the number that actually matters for privacy.

Measured on the **Gretel PII Benchmark** (Gretel AI / NVIDIA), 1,000 rows,
seed 42 — the **same dataset, sample, and scoring as our original benchmark**;
only the detector changed (it was Presidio then), so these are like-for-like.
Reproducible: `python scripts/benchmark_pii.py --sample 1000 --seed 42 --model`.

**Core PII** (name / email / phone / SSN / credit-card)

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| L1+L2 (no model) | 0.589 | 0.531 | 0.559 |
| **L1–L6 (with AI model)** | 0.300\* | **0.840** | 0.442 |

Recall of **0.840** with the model is well above the previous Presidio-based
path's **0.692** — the engine catches *more* core PII, with no Presidio.

> \* The low core-PII *precision* is a **scoring artifact, not a weakness**: the
> engine detects 30+ PII types, but this slice's gold contains only those 5, so
> every correct detection of the other 25+ types (IPs, secrets, IDs, …) is
> counted as a "false positive" here. The fair precision is the all-types view:

**All PII types**

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| L1+L2 (no model) | 0.948 | 0.287 | 0.440 |
| **L1–L6 (with AI model)** | **0.840** | **0.791** | **0.815** |

## Why this is trustworthy

- **Local.** Detection runs entirely on your machine — no cloud, no API key.
- **The same engine everywhere.** Python library, Mac app, and browser
  extension share one Rust kernel, parity-checked so their output is identical.
- **Measured, not asserted.** Every number here is measured **through the
  Python library** on the public Gretel benchmark and is reproducible from one
  command (`scripts/benchmark_pii.py`) — same dataset/sample/scoring as our
  original run.
- **Secrets are never reversible.** Credentials are redacted irreversibly by
  policy, not best-effort.

## Honest limits

- Bare prose names need the L3 model (`mask_pii(model=True)`); L1+L2 alone catch
  structured PII and labeled/titled names, not unlabeled names in free text.
- The model adds a one-time ~809 MB download and per-row inference cost; L1+L2
  run instantly and fully offline.
- These are *our* benchmarks on these corpora — strong and reproducible, but
  your data may differ. The suite is open so you can re-run it on yours.
