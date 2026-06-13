"""
Omna PII benchmark — the CURRENT unified engine, measured through the library.

  Dataset : Gretel PII Benchmark (Gretel AI / NVIDIA)
  Detector: omna_core (the same engine df.omna.mask_pii / pii_report use)
  Configs : Omna L1+L2 (mask_pii)  and  Omna L1-L6 (mask_pii(model=True))

Methodology is IDENTICAL to the original 2026-04-28 benchmark (recoverable in
git history): ground truth comes from the `entities` column; a detected
character span is a TP when it overlaps a gold span (>=1 char) in the same text;
greedy one-to-one matching; sample N rows at a fixed seed. Only the detector
changed (the old run measured Presidio; this measures the shipped engine), so
the numbers are like-for-like with the original core-PII figures.

Run:  python scripts/benchmark_pii.py --sample 1000 --seed 42 [--model]
"""
from __future__ import annotations

import argparse
import ast
import random
import time
from pathlib import Path
from typing import NamedTuple

import polars as pl

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "data" / "gretel_pii.csv"

# Gretel gold types that make up the "Core PII" evaluation (same set the
# original benchmark used, so the headline number is comparable).
CORE_GT_TYPES = frozenset({
    "name", "first_name", "last_name",  # PERSON
    "email",                            # EMAIL
    "ssn",                             # SSN
    "phone_number",                    # PHONE
    "credit_card_number",              # CREDIT_CARD
})


def _parse_entities(raw: str) -> list[dict]:
    try:
        return ast.literal_eval(raw) or []
    except Exception:
        return []


def _locate_gt_spans(text: str, entities: list[dict],
                     type_filter: frozenset[str] | None = None) -> list[tuple[int, int]]:
    if type_filter is not None:
        entities = [e for e in entities
                    if any(t in type_filter for t in e.get("types", []))]
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


def _core_spans(text: str, model: bool) -> list[tuple[int, int]]:
    """Detected character spans from the current omna_core engine."""
    import omna_core
    return [(s["start"], s["end"]) for s in omna_core.detect(text, model=model)]


def _match(det_spans, gt_spans) -> tuple[int, int, int]:
    gt_matched = [False] * len(gt_spans)
    tp = fp = 0
    for d_start, d_end in det_spans:
        hit = False
        for i, (g_start, g_end) in enumerate(gt_spans):
            if not gt_matched[i] and d_start < g_end and d_end > g_start:
                gt_matched[i] = True
                hit = True
                break
        tp += 1 if hit else 0
        fp += 0 if hit else 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


class EvalResult(NamedTuple):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def _metrics(tp: int, fp: int, fn: int) -> EvalResult:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return EvalResult(tp, fp, fn, p, r, f)


def evaluate(rows, model: bool) -> tuple[EvalResult, EvalResult]:
    """One detection pass per doc; scored against core-only and all gold."""
    ctp = cfp = cfn = 0
    atp = afp = afn = 0
    for row in rows:
        det = _core_spans(row["text"], model)  # detect ONCE
        core_gt = _locate_gt_spans(row["text"], row["entities"], type_filter=CORE_GT_TYPES)
        all_gt = _locate_gt_spans(row["text"], row["entities"], type_filter=None)
        a, b, c = _match(det, core_gt); ctp += a; cfp += b; cfn += c
        a, b, c = _match(det, all_gt); atp += a; afp += b; afn += c
    return _metrics(ctp, cfp, cfn), _metrics(atp, afp, afn)


def main() -> None:
    ap = argparse.ArgumentParser(description="Omna PII benchmark (current engine)")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", action="store_true", help="also run L1-L6 (AI model)")
    args = ap.parse_args()

    df = pl.read_csv(DATASET_PATH)
    random.seed(args.seed)
    idx = random.sample(range(len(df)), min(args.sample, len(df)))
    sample_df = df[idx]
    rows = [{"text": sample_df["text"][i], "entities": _parse_entities(sample_df["entities"][i])}
            for i in range(len(sample_df))]
    print(f"Omna PII benchmark — Gretel, N={len(rows):,}, seed={args.seed}\n")

    configs = [("Omna L1+L2", False)] + ([("Omna L1-L6 (model)", True)] if args.model else [])
    for name, model in configs:
        t0 = time.perf_counter()
        core, allt = evaluate(rows, model)
        dt = time.perf_counter() - t0
        print(f"{name}  ({dt:.0f}s)")
        print(f"  core-PII : precision {core.precision:.3f}  recall {core.recall:.3f}  F1 {core.f1:.3f}")
        print(f"  all-types: precision {allt.precision:.3f}  recall {allt.recall:.3f}  F1 {allt.f1:.3f}\n")


if __name__ == "__main__":
    main()
