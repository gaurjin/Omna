# Omna's PII detection engine — what it is, and how it measures

Omna's masking is not a regex wrapper and not a toy. It's a **unified six-layer
detection engine**, written in Rust, that runs identically in the Python
library, the Omna Mac app, and the browser extension. This page explains the
layers and shows measured numbers on public + synthetic benchmarks. Raw figures
are in [`benchmarks.json`](../benchmarks.json); everything here is reproducible
from the engine's benchmark suite.

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

**Synthetic corpus**

| Configuration | Precision | Recall | Leak rate |
|---|---|---|---|
| L1+L2 (no model) | 0.942 | 0.836 | 7.8% |
| **L1–L6 (with AI model)** | 0.880 | **0.938** | **1.8%** |

**Gretel PII Benchmark** (Gretel AI / NVIDIA)

| Slice | Precision | Recall | Leak rate |
|---|---|---|---|
| Full gold (3,183 spans) | 0.621 | 0.764 | 6.5% |
| Core PII (name/email/phone/SSN/card) | — | **0.765** | — |

Core-PII recall of **0.765** is, for the first time, **above the previous
Presidio-based path (0.69)** — with no Presidio and no spaCy.

**Per-type recall on Gretel (with the model):** IP_ADDRESS 0.98 · SSN 0.90 ·
BANK_ACCOUNT 0.89 · MEDICAL_RECORD_NUMBER 0.82 · ADDRESS 0.60.

## Why this is trustworthy

- **Local.** Detection runs entirely on your machine — no cloud, no API key.
- **The same engine everywhere.** Python library, Mac app, and browser
  extension share one Rust kernel, parity-checked so their output is identical.
- **Measured, not asserted.** Every number here comes from the engine's
  benchmark suite against public (Gretel) and seeded synthetic corpora, and is
  reproducible.
- **Secrets are never reversible.** Credentials are redacted irreversibly by
  policy, not best-effort.

## Honest limits

- Bare prose names need the L3 model (`mask_pii(model=True)`); L1+L2 alone catch
  structured PII and labeled/titled names, not unlabeled names in free text.
- The model adds a one-time ~809 MB download and per-row inference cost; L1+L2
  run instantly and fully offline.
- These are *our* benchmarks on these corpora — strong and reproducible, but
  your data may differ. The suite is open so you can re-run it on yours.
