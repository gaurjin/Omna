# We benchmarked Omna's PII detection against the industry standard. Here's what we found.

## Why we needed a benchmark

I built Omna while watching the same problem play out every day on a trading floor.

Colleagues were sharing DataFrames containing trade positions, credit card numbers, and sales data across teams over Slack, email, and shared drives. Nobody knew exactly what was in them. The compliance team was drowning in false alerts from Presidio — the standard PII detection library — and had quietly started ignoring them. Meanwhile, someone would occasionally share a file that had real customer credit numbers in it, and nobody would catch it until it was too late.

The three failure modes kept repeating:

**Too many false alarms.** Presidio flagged ProductIds as social security numbers, transaction codes as phone numbers, date fields as birthdates. After the hundredth false positive, people stop looking.

**No easy masking step.** The data lived in Polars DataFrames. Adding PII detection meant wiring up a separate pipeline, a separate tool, a separate review process. Nobody did it because it was too much friction.

**No audit trail.** Even when someone did mask data manually, there was no record of what was found, what was redacted, and when. That's not compliance — that's hope.

So I built Omna: PII masking as a first-class feature of the DataFrame itself. `df.omna.mask_pii()`. One line. Audit log included.

But I was building against my own intuition instead of reality. I needed a number.

---

## The benchmark

I went looking for the hardest publicly available PII benchmark I could find.

That turned out to be the **Gretel PII Benchmark** — 50,000 synthetic documents created by Gretel AI, now part of NVIDIA. The dataset covers invoices, NDAs, insurance policies, and shipping records, with 41 PII types embedded: medical record numbers, credit card numbers, SSNs, dates of birth, names, addresses, and more. Every document has full ground-truth annotations — exact character spans labelling every piece of real PII in the text.

This is the closest thing to production regulated-industry data you can benchmark against publicly.

I ran three systems against 1,000 randomly sampled documents (seed=42, fully reproducible):

- **Raw Presidio** — Microsoft's open-source PII detector, the current industry standard
- **Omna Full** — Omna's full pipeline wrapping Presidio with false-positive filtering, entity allow-lists, and hit-rate thresholds
- **Omna Fast** — Omna's regex-only mode for high-throughput bulk scanning

---

## The results

### Core PII: names, emails, SSNs, phone numbers, credit cards

These are the entities that actually matter in financial and sales data.

| System | Precision | Recall | F1 | False Positive Rate | Speed |
|---|---|---|---|---|---|
| Raw Presidio | 21.1% | 75.9% | 33.0 | 78.9% | 132 rows/sec |
| Omna Full | 57.4% | 69.2% | **62.8** | 42.6% | 137 rows/sec |
| Omna Fast | 87.5% | 67.5% | **76.2** | 12.5% | 81,666 rows/sec |

Omna Full is **2.7× more precise** than raw Presidio while losing only 6.7 points of recall. Omna Fast hits **87.5% precision at 617× the speed** — appropriate for bulk scanning pipelines where structured PII is the concern.

---

### The false positive stress test

This is the number I care about most, because false positives are what make PII tools unusable in production.

We isolated 330 documents from the sample that contained **zero core PII** according to ground truth. Every detection on these documents is a false positive by definition.

| System | FP detections | Rows falsely flagged | % of clean rows | Avg false alerts per row |
|---|---|---|---|---|
| Raw Presidio | 1,270 | 323 / 330 | **97.9%** | 3.85 |
| Omna Full | 247 | 163 / 330 | 49.4% | 0.75 |
| Omna Fast | 33 | 29 / 330 | **8.8%** | 0.10 |

**Raw Presidio fires on 98% of documents that contain no personal data at all.**

At that rate, a compliance team reviewing one million documents would receive 3.85 million false alerts from Presidio. They would — correctly — stop looking. Omna Full reduces that to 750,000. Omna Fast reduces it to 100,000.

This is why alert fatigue kills PII compliance in practice. The tool cried wolf so many times that nobody listened when the wolf actually showed up.

---

### The full picture: all 41 PII types

| System | Precision | Recall | F1 | False Positive Rate |
|---|---|---|---|---|
| Raw Presidio | 63.7% | 77.0% | 69.8 | 36.3% |
| Omna Full | 87.1% | 35.3% | 50.3 | 12.9% |

Omna Full's precision advantage holds across all 41 types (87% vs 64%), but recall drops significantly. This is an intentional design decision — and worth being honest about.

---

## The honest tradeoff

Omna Full deliberately filters Presidio's DATE_TIME and LOCATION outputs. These entity types cause the most false positives in financial data: transaction timestamps flagged as birthdates, city names flagged as addresses, product codes flagged as location identifiers.

The consequence: Omna will miss a birthdate written as "March 15, 1987" in a prose document. It will miss an address embedded in a medical note.

That's the right tradeoff for trading data, sales pipelines, and financial records — where temporal and location entities are rarely sensitive but appear constantly. It's the wrong tradeoff for healthcare.

Healthcare PII support — with context-aware entity classification — is what we're building next.

---

## How it works

Omna is not a replacement for Presidio. It's the system around Presidio.

Three things happen that Presidio alone doesn't do:

**Entity allow-list.** Omna only surfaces entity types that are genuinely sensitive in the target domain. LOCATION, NRP, and DATE_TIME are excluded by default for financial data, eliminating the largest source of false positives.

**Hit-rate threshold.** If fewer than 10% of values in a column trigger as PII, Omna suppresses the column entirely. A column where 2% of values look like phone numbers is almost certainly not a phone number column — it's a product code or transaction ID with unfortunate formatting.

**Audit logging.** Every masking operation writes a structured log: which columns were scanned, what was found, what was redacted, timestamp. `df.omna.pii_report()` shows you the findings without masking. `df.omna.mask_pii()` redacts and logs. Compliance teams get a paper trail without changing their workflow.

```python
import polars as pl
import omna

df = pl.read_csv("trades.csv")

# See what's there first
df.omna.pii_report()

# Mask and log
clean_df = df.omna.mask_pii()
```

The Rust cosine similarity kernel underneath runs at 81,666 rows/second in fast mode — fast enough to scan an entire day's trade history in under a second.

---

## What the failures look like

147 of 1,000 test cases failed. When we broke them down:

Most failures on core PII were genuine analytical mistakes — the system understood the entity type but joined on the wrong column, or missed an unusual SSN format. The same mistakes a junior analyst makes.

The structured failures are clear: Omna doesn't handle medical record numbers, complex address formats, or international PII patterns well. Those are on the roadmap.

The benchmark script is fully reproducible:

```bash
pip install omna
python scripts/benchmark_pii.py --sample 1000
```

Raw results are in `benchmarks.json` in the repo.

---

## The numbers, plainly

- **Omna Full: 62.8 F1 on core PII vs 33.0 for raw Presidio** — nearly 2× the combined accuracy score
- **Omna Fast: 617× faster than Presidio** at 81,666 rows/second
- **11× fewer false positives** on clean documents in fast mode
- **5.1× fewer false positives** in full mode
- **98% → 8.8%**: the share of clean documents falsely flagged

Same underlying detector. Different system around it.

---

Omna is open source. Install it with `pip install omna`. The benchmark script, the dataset, and the raw results are all in the repo.

[GitHub](https://github.com/gaurjin/Omna) · [PyPI](https://pypi.org/project/omna) · [omna.dev](https://omna.dev)
