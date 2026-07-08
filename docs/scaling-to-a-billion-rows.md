# Scaling local semantic search toward a billion rows

> **This is an open challenge.** Discussion and ideas live in
> [issue #3](https://github.com/gaurjin/Omna/issues/3). The simplest workable
> idea wins — you don't need to know the internals to contribute one.

Omna's promise is semantic search that runs **entirely on your machine — no
cloud, no vector database**. Today that promise is fast and real at **50k–500k
rows** (9–27 ms per search). This doc is an honest look at what stands between
that and a **billion** rows, so contributors know exactly where to push.

## Where Omna is great today

| Rows | Search latency | Where it runs |
|---|---|---|
| 50,000 | ~9 ms | fully local |
| 500,000 | ~27 ms | fully local |

Search is an **exact brute-force cosine scan** — a small Rust kernel over Polars'
Arrow memory, no separate index server. Simple, exact, and fast at these sizes.

## The three walls at a billion rows

**Wall 1 — Index build is a huge one-time job.**
Embedding is ~18 rows/sec on a MacBook Air M5 (50k ≈ 45 min). A billion rows is
**months** on a laptop, ~weeks on a cloud GPU. Inherent to any embedding tool —
a model must read every row once. Not "real-time."

**Wall 2 — Storage is ~3 TB.**
768 dims × 4 bytes ≈ **3 KB/row** → **~3 TB** for a billion vectors, before the
original data. Beyond a laptop SSD.

**Wall 3 — Brute-force search stops being real-time.**
The exact scan grows linearly: a billion rows ≈ **~50 s/query** and wants every
vector in RAM. Real-time at that scale needs an *approximate* index instead.

## Idea directions (contributions welcome)

Each of these is a plausible `good first issue` — a benchmark, a prototype behind
a flag, or a design note all count.

- **Approximate nearest-neighbour** (HNSW / IVF / usearch): check thousands of
  candidates, not billions. The biggest lever for Wall 3.
- **Vector compression** (int8 / binary quantization) or a **smaller embedding
  model**: shrink the ~3 TB and speed up scanning.
- **Disk-backed / memory-mapped index**: search without holding all vectors in RAM.
- **Sharding**: split the index; search only the promising shards.
- **Incremental / streaming indexing**: embed new rows as they arrive.
- **Honest scale tiers**: define and document the real local ceilings
  (e.g. laptop ≈ up to ~10M rows; workstation ≈ ~100M+), so users pick the right
  tool with eyes open.

## The one rule

Whatever the approach: **it must stay local.** No cloud round-trip, no managed
vector service. Keeping that constraint at scale is the entire point of the
challenge.
