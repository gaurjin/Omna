"""Manual measurement: how fast does df.omna.mask_pii run, in rows/sec?
Fast path (L1+L2, model=False) on a large slice, smart path (L3, model=True)
on a small slice. Real numbers for pipeline.md — no estimates.

Run:  ~/Developer/Omna/.venv/bin/python scripts/measure_mask_rate.py

NOTE: the main-guard is REQUIRED — mask_pii uses a ProcessPoolExecutor, and on
macOS (spawn) every worker re-imports this module; without the guard each worker
re-runs the whole benchmark and the pool melts down (BrokenProcessPool).
"""
import time
import polars as pl
from omna.pii import mask_pii


def main():
    csv = "data/gretel_pii.csv"
    df = pl.read_csv(csv)
    text_col = "text" if "text" in df.columns else df.columns[0]
    df = df.select(text_col)
    print(f"loaded {df.height} rows from {csv}, column '{text_col}'")

    # ---- FAST (L1+L2), model=False ----
    n_fast = min(5000, df.height)
    sub = df.head(n_fast)
    t = time.perf_counter()
    _ = mask_pii(sub, columns=[text_col], model=False)
    fast_s = time.perf_counter() - t
    print(f"FAST  (L1+L2, model=False): {n_fast} rows | {fast_s:.2f}s = {n_fast/fast_s:,.0f} rows/s")

    # ---- SMART (L3), model=True, small slice (single-process) ----
    n_smart = min(300, df.height)
    sub2 = df.head(n_smart)
    t = time.perf_counter()
    _ = mask_pii(sub2, columns=[text_col], model=True)
    smart_s = time.perf_counter() - t
    print(f"SMART (L3,    model=True ): {n_smart} rows | {smart_s:.2f}s = {n_smart/smart_s:,.1f} rows/s")


if __name__ == "__main__":
    main()
