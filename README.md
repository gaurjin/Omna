# Omna

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

With Omna:

```python
results  = df.omna.search("insurance claim denied", on="text", k=5)
# Finds ALL of them — including docs that never say "denied" literally.
# 95ms. 50,000 documents. Zero cloud.

filtered = df.omna.filter("insurance claim denied", on="text", threshold=0.73)
# Every semantically matching document above the threshold.
# No keyword lists. No guesswork. Pure meaning.

answer   = results.omna.ask("What personal data do these documents expose?")
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

With Omna:

```python
df.omna.pii_report()   # audit — find every leak, every column
df.omna.mask_pii()     # redact — one line, full audit log
# Names, SSNs, emails, phone numbers — all gone. Local. No cloud.
```

---

**Semantic search, PII masking, and schema understanding — directly on your Polars DataFrames. No vector database. No API key. Data never leaves your machine.**

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
export ANTHROPIC_API_KEY=sk-ant-...       # only needed for ask()
```

Requires Python 3.10+.

---

## Quick start

```python
import polars as pl
import omna

df = pl.read_csv("documents.csv")

# 1 — explore the schema
omna.understand_df(df)

# 2 — audit for PII
df.omna.pii_report()

# 3 — redact before anything else touches the data
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

## API

### `omna.understand_df(df)` — explore before you do anything else

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
 entities              String     0.0%     text      [{'entity': 'John'...
 text                  String     0.0%     text      **Claim ID: 285-14...
```

Labels: `email` `phone` `name` `id` `date` `text` `numeric` `boolean` `category` `unknown`

---

### `df.omna.search("query", on, k)` — semantic search

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

---

### `df.omna.filter("concept", on, threshold)` — semantic filter

> Requires `df.omna.embed("column")` first.

```python
filtered = df.omna.filter("insurance claim denied", on="text", threshold=0.73)
# → N documents matched — all semantically related to claim denials
```

Use `search()` for the top k. Use `filter()` for everything above a threshold.

Default threshold: `0.3`. Raise it for precision, lower it for recall.

---

### `df.omna.pii_report()` — audit before you redact

```python
df.omna.pii_report()
```

```
 column    detected types                                    hit rate   flagged
 entities  CREDIT_CARD, EMAIL_ADDRESS, PERSON, PHONE_NUMBER   85.4%    ✓ YES
 text      CREDIT_CARD, EMAIL_ADDRESS, PERSON, PHONE_NUMBER   78.1%    ✓ YES
```

Scans every string column. Returns hit rates, PII types, and confidence scores. Nothing is modified.

---

### `df.omna.mask_pii()` — redact in one line

```python
clean = df.omna.mask_pii()
# → <REDACTED> replaces every detected entity
# → audit log saved to .omna/pii_audit.parquet automatically

# Fast mode — regex only, ~10x faster, catches email/phone/SSN/URL
clean = df.omna.mask_pii(fast=True)
```

Detects: `PERSON` `EMAIL_ADDRESS` `PHONE_NUMBER` `CREDIT_CARD` `US_SSN` `US_PASSPORT` `IP_ADDRESS` `IBAN_CODE` `URL` and more.

---

### `df.omna.embed("column")` — vectorize once, search forever

Converts text to 384-dimensional vectors using FastEmbed (local ONNX, no API key). Saves to `.omna/{column}.parquet`. Run once — `search()` and `filter()` read the saved file.

```python
df.omna.embed("text")
# → .omna/text.parquet
```

Model: `BAAI/bge-small-en-v1.5` (~130 MB, downloaded once). Embed is a one-time cost. After that, `search()` runs in under 100ms regardless of dataset size.

| Hardware | 50k rows |
|---|---|
| MacBook Air M5 | ~45 min |
| MacBook Pro M4 Max | ~15 min |
| AWS GPU instance | ~2 min |

---

### `df.omna.ask("question")` — natural language queries

Sends schema + up to 20 sample rows to Claude. Returns the answer as a string. Requires `ANTHROPIC_API_KEY`.

```python
results.omna.ask("What personal data do these documents expose?")
# → "These insurance documents expose SSNs, medical record numbers,
#    dates of birth, health plan numbers, and claimant identifiers."

# Override model
results.omna.ask("Summarise the key themes", model="claude-sonnet-4-6")
```

Default model: `claude-haiku-4-5-20251001`.

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
   index.py          loads index → Arrow memory, zero-copy
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

| | 50k rows · 384-dim |
|---|---|
| **Omna search** | **~95ms** |
| **Omna filter** | **~95ms** |
| Pandas + FAISS | ~25ms setup + index build |
| Pandas + keyword regex | misses semantic matches |

Omna inherits Polars' Arrow columnar memory. The Rust similarity kernel operates on the same memory — no copy into NumPy, no copy into a C buffer.

---

## Phase 2 — semantic joins

```python
# Coming soon
matched = transactions.omna.join(regulatory_categories, on="description")
```

Match rows between two DataFrames by meaning, not exact key. Star the repo to follow progress.

---

## License

| Layer | License |
|---|---|
| Python package (`omna/`) | MIT |
| Rust engine (`src/`) | Proprietary — ships as a compiled binary in the pip wheel |
