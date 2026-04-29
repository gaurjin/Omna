#!/usr/bin/env python3
"""
benchmark_pii.py
================
Publication-quality evaluation of PII detection:
  Raw Presidio  vs  Omna Full (filtered)  vs  Omna Fast (regex)

Dataset : Gretel PII Benchmark — 50,000 synthetic documents
          (Gretel AI / NVIDIA, https://huggingface.co/datasets/gretel-ai/gretel-pii-masking-en)
Ground truth: `entities` column contains gold-standard PII spans with text + type labels.

Matching: a detected character span is a TP when it overlaps (by even one character)
with a ground-truth entity span located in the same text.  Each GT span can be
claimed by at most one detected span; each detected span matches at most one GT span.
This prevents double-counting when a GT entity appears multiple times.

Usage:
    python3 scripts/benchmark_pii.py             # 1 000-row sample
    python3 scripts/benchmark_pii.py --sample 500
    python3 scripts/benchmark_pii.py --sample 5000  # slower but tighter CIs
"""

import ast
import re
import sys
import time
import random
import argparse
from pathlib import Path
from typing import NamedTuple

import polars as pl

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "data" / "gretel_pii.csv"

# ─── Ground-truth type taxonomy ──────────────────────────────────────────────
# Gretel types that standard NER (Presidio/spaCy) is designed to detect.
# These are used for the "Core PII" precision/recall evaluation.
CORE_GT_TYPES = frozenset({
    "name", "first_name", "last_name",  # → PERSON
    "email",                             # → EMAIL_ADDRESS
    "ssn",                               # → US_SSN
    "phone_number",                      # → PHONE_NUMBER
    "credit_card_number",                # → CREDIT_CARD
})

# Gretel types catchable by regex alone (Omna fast=True mode)
REGEX_GT_TYPES = frozenset({
    "email", "phone_number", "ssn", "credit_card_number",
})

# ─── Omna internal constants (mirrors omna/pii.py exactly) ──────────────────
_REAL_PII_TYPES = frozenset({
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN",
    "IBAN_CODE", "MEDICAL_LICENSE", "US_PASSPORT", "US_DRIVER_LICENSE",
    "UK_NHS", "SG_NRIC_FIN", "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
    "IN_PAN", "IN_AADHAAR",
})
_MIN_SCORE = 0.35

_FAST_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
    re.compile(r'(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)'),
    re.compile(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b'),
    re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    re.compile(r'https?://[^\s]+'),
]

# ─── Ground-truth parsing ────────────────────────────────────────────────────

def _parse_entities(raw: str) -> list[dict]:
    """Parse the `entities` column string into a list of dicts."""
    try:
        return ast.literal_eval(raw) or []
    except Exception:
        return []


def _locate_gt_spans(
    text: str,
    entities: list[dict],
    type_filter: frozenset[str] | None = None,
) -> list[tuple[int, int]]:
    """
    Find character-level spans for ground-truth entities in `text`.
    Returns a deduplicated list of (start, end) tuples.
    When an entity text occurs multiple times we record each occurrence.
    """
    if type_filter is not None:
        entities = [
            e for e in entities
            if any(t in type_filter for t in e.get("types", []))
        ]

    spans: list[tuple[int, int]] = []
    text_lower = text.lower()
    for ent in entities:
        et = str(ent.get("entity", "")).strip()
        if not et:
            continue
        et_lower = et.lower()
        pos = 0
        while True:
            idx = text_lower.find(et_lower, pos)
            if idx == -1:
                break
            spans.append((idx, idx + len(et_lower)))
            pos = idx + 1
    return spans


# ─── Detection helpers ───────────────────────────────────────────────────────

def _fast_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for pat in _FAST_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _presidio_spans(
    text: str,
    analyzer,
    omna_filter: bool = False,
) -> list[tuple[int, int]]:
    results = analyzer.analyze(text=text, language="en")
    if omna_filter:
        results = [
            r for r in results
            if r.entity_type in _REAL_PII_TYPES and r.score >= _MIN_SCORE
        ]
    return [(r.start, r.end) for r in results]


# ─── Greedy bipartite matching ───────────────────────────────────────────────

def _match(
    det_spans: list[tuple[int, int]],
    gt_spans: list[tuple[int, int]],
) -> tuple[int, int, int]:
    """
    Greedy character-overlap matching between detected and GT spans.
    Each side can be claimed at most once.
    Returns (tp, fp, fn).
    """
    gt_matched = [False] * len(gt_spans)
    tp = fp = 0

    for d_start, d_end in det_spans:
        hit = False
        for i, (g_start, g_end) in enumerate(gt_spans):
            if not gt_matched[i] and d_start < g_end and d_end > g_start:
                gt_matched[i] = True
                hit = True
                break
        if hit:
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


# ─── Result container ────────────────────────────────────────────────────────

class EvalResult(NamedTuple):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fp_rate: float       # FP / total detections  (1 - precision)
    elapsed_sec: float
    rows_per_sec: float


def _compute_metrics(tp: int, fp: int, fn: int, elapsed: float, n: int) -> EvalResult:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fp_rate   = fp / (tp + fp) if (tp + fp) else 0.0
    return EvalResult(tp, fp, fn, precision, recall, f1, fp_rate, elapsed, n / elapsed)


# ─── Per-system evaluation ───────────────────────────────────────────────────

def evaluate(
    rows: list[dict],
    mode: str,
    analyzer=None,
    gt_type_filter: frozenset[str] | None = None,
) -> EvalResult:
    """
    Parameters
    ----------
    rows           : list of {"text": str, "entities": list[dict]}
    mode           : "raw_presidio" | "omna_full" | "omna_fast"
    analyzer       : Presidio AnalyzerEngine (ignored for omna_fast)
    gt_type_filter : if set, only count GT entities of these Gretel types
    """
    total_tp = total_fp = total_fn = 0
    t0 = time.perf_counter()

    for row in rows:
        text     = row["text"]
        entities = row["entities"]

        gt_spans = _locate_gt_spans(text, entities, type_filter=gt_type_filter)

        if mode == "omna_fast":
            det_spans = _fast_spans(text)
        elif mode == "omna_full":
            det_spans = _presidio_spans(text, analyzer, omna_filter=True)
        else:  # raw_presidio
            det_spans = _presidio_spans(text, analyzer, omna_filter=False)

        tp, fp, fn = _match(det_spans, gt_spans)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    elapsed = time.perf_counter() - t0
    return _compute_metrics(total_tp, total_fp, total_fn, elapsed, len(rows))


# ─── Reporting ───────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def _print_table(results: dict[str, EvalResult]) -> None:
    header = f"{'System':<22} {'Precision':>9} {'Recall':>8} {'F1':>7} {'FP rate':>8} {'Speed (r/s)':>11}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        print(
            f"{name:<22} "
            f"{_pct(r.precision):>9} "
            f"{_pct(r.recall):>8} "
            f"{_pct(r.f1):>7} "
            f"{_pct(r.fp_rate):>8} "
            f"{r.rows_per_sec:>10.0f}"
        )
    print(sep)


def _print_counts(results: dict[str, EvalResult]) -> None:
    header = f"{'System':<22} {'TP':>7} {'FP':>7} {'FN':>7}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        print(f"{name:<22} {r.tp:>7,} {r.fp:>7,} {r.fn:>7,}")
    print(sep)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Omna vs Presidio PII benchmark")
    parser.add_argument("--sample", type=int, default=1000,
                        help="Number of rows to sample (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nOmna PII Benchmark  —  Gretel PII Benchmark dataset (N={args.sample:,})")
    print("=" * 64)

    # ── Load & sample ──────────────────────────────────────────────
    print(f"\nLoading {DATASET_PATH.name} …")
    df = pl.read_csv(DATASET_PATH)
    random.seed(args.seed)
    indices = random.sample(range(len(df)), min(args.sample, len(df)))
    sample_df = df[indices]
    print(f"Sampled {len(sample_df):,} rows (seed={args.seed})")

    rows = [
        {
            "text":     sample_df["text"][i],
            "entities": _parse_entities(sample_df["entities"][i]),
        }
        for i in range(len(sample_df))
    ]

    # Quick sanity check: GT entity count
    gt_total = sum(len(r["entities"]) for r in rows)
    core_total = sum(
        len(_locate_gt_spans(r["text"], r["entities"], CORE_GT_TYPES))
        for r in rows
    )
    regex_total = sum(
        len(_locate_gt_spans(r["text"], r["entities"], REGEX_GT_TYPES))
        for r in rows
    )
    print(f"GT entities in sample  : {gt_total:,} total  |  "
          f"{core_total:,} core (name/email/SSN/phone/CC)  |  "
          f"{regex_total:,} regex-catchable (email/phone/SSN/CC)")

    # ── Build Presidio analyzer once ──────────────────────────────
    print("\nInitialising Presidio AnalyzerEngine …", end="", flush=True)
    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    # Warm-up to pay spaCy model-load cost before timing
    _ = analyzer.analyze(text="John Smith, john@example.com, 555-867-5309", language="en")
    print(" done.")

    # ─────────────────────────────────────────────────────────────
    # EVALUATION 1: Core PII types (name / email / SSN / phone / CC)
    # These are the types all three systems are designed to catch.
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("EVALUATION 1 — Core PII (name · email · SSN · phone · credit card)")
    print("Ground truth filtered to: name/first_name/last_name, email,")
    print("                          ssn, phone_number, credit_card_number")
    print("=" * 64)

    core_results: dict[str, EvalResult] = {}

    print("\n  Running Raw Presidio …", end="", flush=True)
    core_results["Raw Presidio"] = evaluate(
        rows, "raw_presidio", analyzer, gt_type_filter=CORE_GT_TYPES
    )
    print(f" {core_results['Raw Presidio'].elapsed_sec:.1f}s")

    print("  Running Omna Full  …", end="", flush=True)
    core_results["Omna Full"] = evaluate(
        rows, "omna_full", analyzer, gt_type_filter=CORE_GT_TYPES
    )
    print(f" {core_results['Omna Full'].elapsed_sec:.1f}s")

    print("  Running Omna Fast  …", end="", flush=True)
    core_results["Omna Fast (email/phone/SSN/CC)"] = evaluate(
        rows, "omna_fast", analyzer, gt_type_filter=REGEX_GT_TYPES
    )
    print(f" {core_results['Omna Fast (email/phone/SSN/CC)'].elapsed_sec:.3f}s")

    print()
    _print_table(core_results)
    print()
    _print_counts(core_results)

    # ─────────────────────────────────────────────────────────────
    # EVALUATION 2: False-positive stress test
    # Rows where the ground truth contains ZERO core PII entities.
    # Every detection is a false positive by definition.
    # ─────────────────────────────────────────────────────────────
    no_core_rows = [
        r for r in rows
        if not _locate_gt_spans(r["text"], r["entities"], CORE_GT_TYPES)
    ]
    print("\n" + "=" * 64)
    print(f"EVALUATION 2 — False-Positive Stress Test (N={len(no_core_rows):,} rows)")
    print("Rows with ZERO core PII in ground truth.")
    print("Every detection here is a false positive.")
    print("=" * 64)

    if len(no_core_rows) < 10:
        print("  (Too few rows without core PII — skipping this section.)")
    else:
        fp_results: dict[str, dict] = {}
        for name, mode in [
            ("Raw Presidio", "raw_presidio"),
            ("Omna Full",    "omna_full"),
            ("Omna Fast",    "omna_fast"),
        ]:
            t0 = time.perf_counter()
            total_fp_here = 0
            rows_with_any_fp = 0
            for row in no_core_rows:
                text = row["text"]
                if mode == "omna_fast":
                    det_spans = _fast_spans(text)
                elif mode == "omna_full":
                    det_spans = _presidio_spans(text, analyzer, omna_filter=True)
                else:
                    det_spans = _presidio_spans(text, analyzer, omna_filter=False)
                total_fp_here += len(det_spans)
                if det_spans:
                    rows_with_any_fp += 1
            elapsed = time.perf_counter() - t0
            fp_results[name] = {
                "total_fp":   total_fp_here,
                "rows_flagged": rows_with_any_fp,
                "pct_rows":   rows_with_any_fp / len(no_core_rows),
                "avg_fp_per_row": total_fp_here / len(no_core_rows),
                "elapsed": elapsed,
            }

        hdr = f"{'System':<22} {'Total FP':>9} {'Rows flagged':>13} {'% rows':>8} {'Avg FP/row':>11}"
        sep = "-" * len(hdr)
        print(f"\n{sep}")
        print(hdr)
        print(sep)
        for name, d in fp_results.items():
            print(
                f"{name:<22} "
                f"{d['total_fp']:>9,} "
                f"{d['rows_flagged']:>13,} "
                f"{_pct(d['pct_rows']):>8} "
                f"{d['avg_fp_per_row']:>11.2f}"
            )
        print(sep)

    # ─────────────────────────────────────────────────────────────
    # EVALUATION 3: All GT types (complete picture)
    # Measures recall across the full Gretel taxonomy —
    # including types Presidio can partially or rarely detect.
    # Lower recall here is expected and acceptable; what matters
    # is that Omna Full is not dramatically worse than raw Presidio.
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("EVALUATION 3 — All Ground-Truth PII Types (full taxonomy)")
    print("Includes medical record #, dates, addresses, IDs, etc.")
    print("Lower recall is expected — shows the complete picture.")
    print("=" * 64)

    all_results: dict[str, EvalResult] = {}

    print("\n  Running Raw Presidio …", end="", flush=True)
    all_results["Raw Presidio"] = evaluate(
        rows, "raw_presidio", analyzer, gt_type_filter=None
    )
    print(f" {all_results['Raw Presidio'].elapsed_sec:.1f}s")

    print("  Running Omna Full  …", end="", flush=True)
    all_results["Omna Full"] = evaluate(
        rows, "omna_full", analyzer, gt_type_filter=None
    )
    print(f" {all_results['Omna Full'].elapsed_sec:.1f}s")

    print()
    _print_table(all_results)
    print()
    _print_counts(all_results)

    # ─────────────────────────────────────────────────────────────
    # Speed summary (all three modes head-to-head)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SPEED SUMMARY (single-threaded, no batching)")
    print("=" * 64)
    speeds = {
        "Raw Presidio":  core_results["Raw Presidio"].rows_per_sec,
        "Omna Full":     core_results["Omna Full"].rows_per_sec,
        "Omna Fast":     core_results["Omna Fast (email/phone/SSN/CC)"].rows_per_sec,
    }
    fastest = max(speeds.values())
    hdr = f"{'System':<22} {'Rows/sec':>10} {'vs Presidio':>12}"
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    presidio_speed = speeds["Raw Presidio"]
    for name, speed in speeds.items():
        ratio = speed / presidio_speed
        print(f"{name:<22} {speed:>10.0f} {ratio:>10.1f}x")
    print(sep)

    print(f"\nDataset : Gretel PII Benchmark (NVIDIA / Gretel AI)")
    print(f"Sample  : {args.sample:,} rows, seed={args.seed}")
    print(f"Note    : Omna Full is single-threaded here; production uses")
    print(f"          ProcessPoolExecutor (N_CPU workers) for ~{8}x additional throughput.")
    print()


if __name__ == "__main__":
    main()
