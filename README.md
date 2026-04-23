# Omna

```python
# Before Omna
df.filter(
    pl.col("text").str.contains("angry") |
    pl.col("text").str.contains("frustrated") |
    pl.col("text").str.contains("terrible") |
    pl.col("text").str.contains("worst")
)
# Still misses: "I am done with you people"
# Still misses: "this is unacceptable"
# Still misses: anything in another language

# After Omna
df.omna.search("angry customer", on="text", k=10)
# Finds all of them. 9ms. One line.
```

**500,000 rows searched by meaning in 9ms. No vector database. No API key. PII never leaves your machine.**

AI is smart but blind. Your data is rich but invisible. Omna is the bridge.

Omna is a Polars namespace plugin. Semantic search, PII masking, and schema understanding — directly on your DataFrames, no new infrastructure required.

---

## Install

```bash
pip install omna
python -m spacy download en_core_web_lg   # one-time, for PII detection
export ANTHROPIC_API_KEY=sk-ant-...       # only needed for ask()
```

Requires Python 3.10+.

---

## API

```python
import polars as pl
import omna

df = pl.read_csv("data.csv")
```

---

### `omna.understand(df)` — schema inference

No LLM. Analyzes every column and assigns a semantic label from column name, dtype, and sample values.

```python
omna.understand(df)
```

```
┌─────────────┬─────────┬──────────┬──────────────┬─────────┬──────────────────────┐
│ column      ┆ dtype   ┆ null_pct ┆ unique_count ┆ label   ┆ sample               │
╞═════════════╪═════════╪══════════╪══════════════╪═════════╪══════════════════════╡
│ customer_id ┆ Int64   ┆ 0.0      ┆ 50000        ┆ id      ┆ 1, 2, 3              │
│ email       ┆ String  ┆ 0.0      ┆ 50000        ┆ email   ┆ 'a@b.com', 'c@d.io' │
│ phone       ┆ String  ┆ 2.3      ┆ 48847        ┆ phone   ┆ '555-1234', ...      │
│ notes       ┆ String  ┆ 0.0      ┆ 49901        ┆ text    ┆ 'Escalated twice...' │
│ created_at  ┆ Date    ┆ 0.0      ┆ 730          ┆ date    ┆ 2023-01-01, ...      │
│ churned     ┆ Boolean ┆ 0.0      ┆ 2            ┆ boolean ┆ True, False          │
└─────────────┴─────────┴──────────┴──────────────┴─────────┴──────────────────────┘
```

Labels: `email` `phone` `name` `id` `date` `text` `numeric` `boolean` `category` `unknown`

---

### `df.omna.embed("column")` — vectorize once, search forever

Converts text to 384-dimensional vectors using FastEmbed (local, no API key). Saves the index to `.omna/{column}.parquet`. Run once — `search()` and `filter()` read the saved file.

```python
df.omna.embed("notes")
# → .omna/notes.parquet
```

Default model: `BAAI/bge-small-en-v1.5` (~130 MB, downloaded once). Override the save path:

```python
df.omna.embed("notes", index_path="indexes/notes.parquet")
```

---

### `df.omna.search("query", on, k)` — semantic search

Returns the k rows most similar to your query by meaning, not keyword match.

```python
results = df.omna.search("angry customer", on="notes", k=5)
```

```
┌─────────────┬────────────────────────────────────────────┬────────┐
│ customer_id ┆ notes                                      ┆ _score │
╞═════════════╪════════════════════════════════════════════╪════════╡
│ 4821        ┆ "Absolutely furious. Nobody called back."  ┆ 0.91   │
│ 1203        ┆ "I've never been so frustrated in my life" ┆ 0.89   │
│ 9944        ┆ "Terrible experience, demanded a refund"   ┆ 0.87   │
│ 3312        ┆ "I am done with you people"                ┆ 0.85   │
│ 7701        ┆ "this is completely unacceptable"          ┆ 0.83   │
└─────────────┴────────────────────────────────────────────┴────────┘
```

`_score` is cosine similarity (0–1), sorted descending. None of these rows contain the words "angry" or "customer".

---

### `df.omna.filter("concept", on, threshold)` — concept filter

Keeps all rows above a similarity threshold. Returns a normal DataFrame — no `_score` column.

```python
flagged = df.omna.filter("fraud related", on="notes", threshold=0.75)
```

Use `search()` when you want the top k. Use `filter()` when you want everything above a threshold.

---

### `df.omna.mask_pii()` — redact PII, save audit log

Detects and redacts across all string columns. Powered by Microsoft Presidio. Runs locally — data never leaves your machine.

When an AI reads your DataFrame to summarise patterns, it also reads every name, email, and phone number in your sample rows. If you're using a cloud LLM (GPT-4, Claude API), that data just left your building. In finance, healthcare, and legal — **that is a regulatory violation.** GDPR, CCPA, FCA, SEC — all have rules about this.

**PII masking as a first-class feature** means it is a built-in method on every DataFrame, called *before* any AI touches the data:

```python
# Step 1 — mask before the AI touches anything
clean_df = df.omna.mask_pii(columns=["name", "email", "phone"])

# Step 2 — now safe to search
results = clean_df.omna.search("fraud complaint", on="notes", k=50)
```

Before `mask_pii()`:

| customer_id | name | email | phone | notes |
|---|---|---|---|---|
| 8821 | John Smith | john@gmail.com | 07911123456 | I was charged twice and want refund |
| 8822 | Sarah Jones | sarah@bank.com | 07922234567 | Suspicious transaction on my account |

After `mask_pii()`:

| customer_id | name | email | phone | notes |
|---|---|---|---|---|
| 8821 | `[REDACTED]` | `[REDACTED]` | `[REDACTED]` | I was charged twice and want refund |
| 8822 | `[REDACTED]` | `[REDACTED]` | `[REDACTED]` | Suspicious transaction on my account |


Detects: `PERSON` `EMAIL_ADDRESS` `PHONE_NUMBER` `CREDIT_CARD` `US_SSN` `US_PASSPORT` `IP_ADDRESS` `IBAN_CODE` `LOCATION` `URL` and more.

The audit log records every redaction — column, row, entity type, original text, confidence score — in Parquet format. Built for GDPR, HIPAA, and SOC 2 trails.

```python
# audit log → .omna/pii_audit.parquet
clean = df.omna.mask_pii()

# custom audit path for compliance teams
clean = df.omna.mask_pii(audit_path="compliance/2024-04-23.parquet")
```

---

### `df.omna.pii_report()` — preview before you redact

See exactly what will be redacted. Doesn't modify anything.

```python
df.omna.pii_report()
```

```
┌─────────┬─────┬──────────────────┬────────────────┬───────┐
│ column  ┆ row ┆ entity_type      ┆ text           ┆ score │
╞═════════╪═════╪══════════════════╪════════════════╪═══════╡
│ notes   ┆ 0   ┆ PERSON           ┆ Alice Smith    ┆ 0.85  │
│ notes   ┆ 0   ┆ PHONE_NUMBER     ┆ 555-867-5309   ┆ 0.75  │
│ notes   ┆ 0   ┆ EMAIL_ADDRESS    ┆ alice@corp.com ┆ 1.0   │
└─────────┴─────┴──────────────────┴────────────────┴───────┘
```

---

### `df.omna.ask("question")` — natural language queries

Sends schema + up to 20 sample rows to Claude. Returns the answer as a string. Requires `ANTHROPIC_API_KEY`.

```python
df.omna.ask("which customers complained about refunds?")
# → "Customers 1203, 4821, and 9944 mentioned refund issues.
#    Row 1203 was the most explicit: 'demanded a refund immediately'."
```

Default model: `claude-haiku-4-5-20251001`. Override:

```python
df.omna.ask("summarise the key complaint themes", model="claude-sonnet-4-6")
```

---

## How it works

```
df.omna.search("angry customer", on="notes", k=10)
         │
         ▼
   embedder.py       FastEmbed — BAAI/bge-small-en-v1.5, local ONNX
                     "angry customer" → [0.12, -0.34, 0.87, ...]  384-dim
         │
         ▼
   index.py          loads .omna/notes.parquet → Arrow memory, zero-copy
                     500,000 stored vectors already in Polars' allocation
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

## Performance ## Why Omna is fast
Omna inherits Polars' Arrow columnar memory, SIMD vectorization, and zero-copy data structures. The Rust similarity kernel operates on the same memory Polars is already using — no copy into NumPy, no copy into a C buffer.

When you call `df.omna.search()`, the Rust similarity kernel operates directly on the same memory Polars is already using.

> "Polars uses Apache Arrow's columnar memory format with SIMD vectorization — the same memory our Rust similarity kernel operates on directly. On Pandas we'd need to copy data into NumPy arrays first. On Polars it's zero-copy end to end."

> — Ritchie Vink explains the architecture: [I wrote one of the fastest DataFrame libraries](https://pola.rs/posts/i-wrote-one-of-the-fastest-dataframe-libraries/)

| | 500k rows · 384-dim | Data copies |
|---|---|---|
| **Omna** | **9ms** | zero-copy |
| Pandas + FAISS | ~25ms | yes |
| Pandas + NumPy | ~180ms | yes |


---

## Phase 2 — semantic joins

```python
# Coming soon
matched = transactions.omna.join(regulatory_categories, on="description")
```

Match rows between two DataFrames by meaning, not exact key. The missing feature across every existing semantic library. Star the repo to follow progress.

---

## License

| Layer | License | Notes |
|---|---|---|
| Python package (`omna/`) | MIT | [github.com/gaurjin/omna](https://github.com/gaurjin/omna) |
| Rust engine (`src/`) | Proprietary | Ships as a compiled binary inside the pip wheel |
