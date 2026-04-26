# Omna

[![PyPI](https://img.shields.io/pypi/v/omna)](https://pypi.org/project/omna/)
[![Python](https://img.shields.io/pypi/pyversions/omna)](https://pypi.org/project/omna/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-113%20passing-brightgreen)](tests/)

**Semantic search, PII masking, and schema understanding — directly on your Polars DataFrames. No vector database. No API key. Data never leaves your machine.**

---

## The problem

```python
# Finding every insurance claim denial — painful
keywords = ["claim denied", "coverage rejected", "policy voided", ...]
pattern  = re.compile("|".join(keywords), re.IGNORECASE)
results  = df[df["text"].str.contains(pattern, na=False)]
# Still misses: "insurer refused to honour the policy"
# Still misses: "claim outcome: not payable"
# Still misses: medical claim rejections using clinical terminology
# ...50+ lines per task. Grows with every edge case. Still wrong.
```

```python
# With Omna
results = df.omna.search("insurance claim denied", on="text", k=5)
# Finds ALL of them — including docs that never say "denied" literally.
# 9ms. 50,000 documents. Zero cloud.

filtered = df.omna.filter("insurance claim denied", on="text", threshold=0.73)
# Every semantically matching document above the threshold.
# No keyword lists. No guesswork. Pure meaning.

answer = results.omna.ask("What personal data do these documents expose?")
# → "These insurance documents expose SSNs, medical record numbers,
#    dates of birth, health plan numbers, and claimant identifiers."
# Instant. One line.
```

---

```python
# Auditing for PII before the data ships — painful
for col in df.columns:
    for i, val in enumerate(df[col].to_list()):
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', str(val)):   # SSNs only
            print(f"row {i}, {col}: {str(val)[:60]}")
# Catches one pattern. Misses emails, phones, names, IBANs.
# No confidence score. No audit trail. No redaction.
```

```python
# With Omna
df.omna.pii_report()   # audit — find every leak, every column
df.omna.mask_pii()     # redact — one line, full audit log
# Names, SSNs, emails, phone numbers — all gone. Local. No cloud.
```

---

## Demo

**The Sword** — semantic search, filter, and ask across 50,000 documents:

![Omna Sword Demo](assets/demo_sword.gif)

**The Shield** — PII audit and redaction in one line:

![Omna Shield Demo](assets/demo_shield.gif)

Dataset: [Gretel PII Benchmark](https://gretel.ai) (acquired by NVIDIA) — 50,000 synthetic documents built to test data privacy tools.

---

## Install

```bash
pip install omna
python -m spacy download en_core_web_lg   # one-time, for PII detection
```

Requires Python 3.10+. No API key needed for search, filter, embed, pii_report, mask_pii, or understand. Only `ask()` requires `ANTHROPIC_API_KEY`.

---

## Quick start

```python
import polars as pl
import omna

df = pl.read_csv("documents.csv")

# 1 — explore the schema
omna.understand_df(df)

# 2 — audit for PII before anything touches the data
df.omna.pii_report()

# 3 — redact
clean = df.omna.mask_pii()

# 4 — build a search index once
clean.omna.embed("text")

# 5 — search by meaning
results = clean.omna.search("insurance claim denied", on="text", k=5)

# 6 — filter everything above a threshold
flagged = clean.omna.filter("insurance claim denied", on="text", threshold=0.73)

# 7 — ask a question in plain English
results.omna.ask("What personal data do these documents expose?")
```

---

## What Omna does

| Method | What it does |
|---|---|
| `omna.understand_df(df)` | Schema inference — labels, null rates, samples. No LLM. |
| `df.omna.embed(column)` | Vectorize a text column once; reuse across sessions |
| `df.omna.search(query, on, k)` | Top-k results by semantic meaning |
| `df.omna.filter(query, on, threshold)` | Every row above a similarity threshold |
| `df.omna.pii_report()` | Audit every string column for PII |
| `df.omna.mask_pii()` | Redact PII, auto-save audit log |
| `df.omna.ask(question)` | Natural language queries over your DataFrame |

---

## API reference

<details>
<summary><b>omna.understand_df(df)</b> — explore before you do anything</summary>

No LLM. No API call. Analyzes column names, dtypes, null rates, and sample values.

```python
omna.understand_df(df)
```

```
 column                dtype    null_pct   label     sample
 uid                   String     0.0%     category  24bb757...
 domain                String     0.0%     category  insurance, healthcare...
 document_type         String     0.0%     category  Invoice, ClaimForm...
 document_description  String     0.0%     text      An insurance claim...
 text                  String     0.0%     text      **Claim ID: 285-14...
```

Labels: `email` `phone` `name` `id` `date` `text` `numeric` `boolean` `category` `unknown`

</details>

<details>
<summary><b>df.omna.embed(column)</b> — vectorize once, search forever</summary>

Converts text to 384-dimensional vectors using FastEmbed (local ONNX, no API key). Saves to `.omna/{column}.parquet`. Run once — `search()` and `filter()` load it automatically on every subsequent call.

```python
df.omna.embed("text")
# → .omna/text.parquet
```

Model: `BAAI/bge-small-en-v1.5` (~130 MB, downloaded once). Embed is a one-time cost.

| Hardware | 50k rows |
|---|---|
| MacBook Air M5 | ~45 min |
| MacBook Pro M4 Max | ~15 min |
| AWS GPU instance | ~2 min |

</details>

<details>
<summary><b>df.omna.search(query, on, k)</b> — semantic search</summary>

> Requires `df.omna.embed("column")` first.

```python
results = df.omna.search("insurance claim denied", on="text", k=5)
```

```
 uid            document_type         domain      text                               _score
 67fccc1e207…   ClaimSummary          insurance   **Claim ID: 285-14-1755, Policy…   0.762
 b8ae088cd21…   ClaimSummary          insurance   **Claim Summary**…                 0.749
 de5bba0a2cc…   Insurance Claim Form  healthcare  **Insurance Claim Form**…          0.748
 ebccdde3b42…   Insurance Claim       healthcare  Insurance Claim for MED74974358…   0.747
 aebb0eb55fb…   ClaimForm             healthcare  **Claim Form** - Patient ID…       0.747
```

`_score` is cosine similarity (0–1). None of these documents contain the phrase "insurance claim denied" — Omna finds them by meaning.

</details>

<details>
<summary><b>df.omna.filter(query, on, threshold)</b> — semantic filter</summary>

> Requires `df.omna.embed("column")` first.

```python
filtered = df.omna.filter("insurance claim denied", on="text", threshold=0.73)
# → N documents matched — all semantically related to claim denials
```

Returns every row above the threshold. Default: `0.3`. Raise for precision, lower for recall.

Use `search()` for the top k. Use `filter()` for everything above a threshold.

</details>

<details>
<summary><b>df.omna.pii_report()</b> — audit before you redact</summary>

```python
df.omna.pii_report()
```

```
 column    detected types                                    hit rate   flagged
 entities  CREDIT_CARD, EMAIL_ADDRESS, PERSON, PHONE_NUMBER   85.4%    ✓ YES
 text      CREDIT_CARD, EMAIL_ADDRESS, PERSON, PHONE_NUMBER   78.1%    ✓ YES
```

Scans every string column. Returns hit rates, PII types, and confidence scores. Nothing is modified.

</details>

<details>
<summary><b>df.omna.mask_pii()</b> — redact in one line</summary>

```python
clean = df.omna.mask_pii()
# → <REDACTED> replaces every detected entity
# → audit log saved to .omna/pii_audit.parquet automatically

# Fast mode — regex only, ~10x faster, catches email/phone/SSN/URL
clean = df.omna.mask_pii(fast=True)
```

Detects: `PERSON` `EMAIL_ADDRESS` `PHONE_NUMBER` `CREDIT_CARD` `US_SSN` `US_PASSPORT` `IP_ADDRESS` `IBAN_CODE` `URL` and more.

</details>

<details>
<summary><b>df.omna.ask(question)</b> — natural language queries</summary>

Sends schema + up to 20 sample rows to Claude. Requires `ANTHROPIC_API_KEY`.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
results.omna.ask("What personal data do these documents expose?")
# → "These insurance documents expose SSNs, medical record numbers,
#    dates of birth, health plan numbers, and claimant identifiers."

# Override model
results.omna.ask("Summarise the key themes", model="claude-sonnet-4-6")
```

Default model: `claude-haiku-4-5-20251001`.

</details>

---

## How it works

```
df.omna.search("insurance claim denied", on="text", k=5)
         │
         ▼
   embedder.py       FastEmbed — BAAI/bge-small-en-v1.5, local ONNX
                     query → [0.12, -0.34, 0.87, ...]  384-dim vector
         │
         ▼
   index.py          loads .omna/text.parquet → Arrow memory, zero-copy
                     50,000 stored vectors in Polars' own allocation
         │
         ▼
   similarity.rs     Rust kernel — cosine similarity over all vectors
                     returns top-k sorted descending, no Python loop
         │
         ▼
   frame.py          slices result rows, attaches _score → pl.DataFrame
```

The Rust kernel is 23 lines. Dot products and norms in machine code, no intermediate allocations. 500,000 × 384-dim in under 10ms on a single core.

---

## Performance

| | 50k rows | 500k rows |
|---|---|---|
| **Omna search** | **9ms** | **27ms** |
| **Omna filter** | **9ms** | **27ms** |
| Pandas + FAISS | ~25ms + index build | ~25ms + index build |
| Polars keyword regex | 1ms — exact match only | 1ms — exact match only |

Benchmarked on MacBook Air M5, `BAAI/bge-small-en-v1.5` (384-dim), 10-query median, warm index.

Omna inherits Polars' Arrow columnar memory. The Rust similarity kernel operates on the same memory — no copy into NumPy, no copy into a C buffer.

---

## FAQ

<details>
<summary><b>Does Omna send my data to the cloud?</b></summary>

No. Embedding, search, filter, PII detection, and masking all run locally. The only method that makes a network call is `ask()`, which sends schema metadata and sample rows to Claude via the Anthropic API — and only when you explicitly call it.

</details>

<details>
<summary><b>Do I need a GPU?</b></summary>

No. FastEmbed uses ONNX and runs on CPU. On Apple Silicon, it uses CoreML automatically. Embedding 50,000 documents takes ~45 minutes on a MacBook Air M5 — a one-time cost. After that, `search()` and `filter()` run in milliseconds from the saved index.

</details>

<details>
<summary><b>Why not FAISS / ChromaDB / Pinecone?</b></summary>

Those are vector databases. Omna is a Polars plugin. If your data already lives in a DataFrame, Omna adds semantic search with zero infrastructure — no separate process, no index server, no network hop. It's the difference between `df.omna.search(...)` and spinning up a separate service just to query your own data.

</details>

<details>
<summary><b>What PII types does Omna detect?</b></summary>

`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `US_SSN`, `US_PASSPORT`, `IP_ADDRESS`, `IBAN_CODE`, `URL`, `DATE_TIME`, `LOCATION`, and more. Detection uses Microsoft Presidio + spaCy NER, running fully local.

</details>

<details>
<summary><b>Which Polars versions are supported?</b></summary>

Omna is tested on Polars 0.20+. It installs as a namespace plugin via `df.omna.*` — no import needed after `import omna`.

</details>

<details>
<summary><b>The embed step took 45 minutes. Do I have to redo it every time?</b></summary>

No. `embed()` saves the index to `.omna/{column}.parquet`. Every subsequent `search()` or `filter()` call loads it in ~300ms. You only re-run `embed()` if your data changes.

</details>

---

## Roadmap

```python
# Coming in v0.2
matched = transactions.omna.join(regulatory_categories, on="description")
# Match rows between two DataFrames by meaning, not exact key.
```

Star the repo to follow progress.

---

## License

| Layer | License |
|---|---|
| Python package (`omna/`) | MIT |
| Rust engine (`src/`) | Proprietary — ships as a compiled binary in the pip wheel |

---

**[omna.dev](https://omna.dev) · [PyPI](https://pypi.org/project/omna/) · [GitHub](https://github.com/gaurjin/Omna)**
