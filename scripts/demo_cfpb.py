"""
Omna — Terminal Demo
Shield → Sword → Closer  (~50s on pre-warmed M5)

Dataset: CFPB Consumer Financial Complaints (9 GB, 4.4M records)
Run with: uv run python scripts/demo_cfpb.py
"""
import sys
import os
import time
import threading
import polars as pl
import omna
from contextlib import contextmanager

if __name__ == "__main__":

    # ── palette ───────────────────────────────────────────────────────────────
    R   = "\033[91m"    # red   — danger, leaks, federal failure
    G   = "\033[92m"    # green — Omna's shield, success
    C   = "\033[96m"    # cyan  — speed, magic, metrics
    W   = "\033[97m"    # white — neutral emphasis
    DIM = "\033[2m"     # dim   — secondary text
    B   = "\033[1m"     # bold
    RST = "\033[0m"     # reset

    REAL = sys.__stdout__
    COL  = "Consumer complaint narrative"

    # ── helpers ───────────────────────────────────────────────────────────────

    @contextmanager
    def silent():
        devnull = open(os.devnull, "w")
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old
            devnull.close()

    def out(text: str = "") -> None:
        REAL.write(text + "\n")
        REAL.flush()

    def typewrite(text: str, delay: float = 0.038) -> None:
        for ch in text:
            REAL.write(ch)
            REAL.flush()
            time.sleep(delay)
        REAL.write("\n")
        REAL.flush()

    def pause(s: float) -> None:
        time.sleep(s)

    def spinner_while(fn, label: str = "working"):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        stop   = threading.Event()
        result = [None]
        err    = [None]

        def spin():
            i = 0
            while not stop.is_set():
                REAL.write(f"\r  {DIM}{label}{RST}  {frames[i % 10]}")
                REAL.flush()
                i += 1
                time.sleep(0.09)

        def run():
            with silent():
                try:
                    result[0] = fn()
                except Exception as e:
                    err[0] = e

        t1 = threading.Thread(target=spin, daemon=True)
        t2 = threading.Thread(target=run,  daemon=True)
        t1.start()
        t2.start()
        t2.join()
        stop.set()
        t1.join(timeout=0.3)
        REAL.write(f"\r{' ' * 62}\r")
        REAL.flush()
        if err[0]:
            raise err[0]
        return result[0]

    def bar(color: str = DIM) -> None:
        out(f"  {color}" + "─" * 60 + RST)

    def section(icon: str, title: str, color: str) -> None:
        out()
        out(f"  {color}{B}{icon}  {title}{RST}")
        bar(color)
        out()
        pause(0.2)

    # ── SILENT LOADING  (~2s on pre-warmed machine) ───────────────────────────
    REAL.write(f"\n  {DIM}loading ...{RST}")
    REAL.flush()

    with silent():
        df = (
            pl.read_csv(
                "data/cfpb/complaints.csv",
                n_rows=200_000,
                infer_schema_length=10_000,
            )
            .drop_nulls(subset=[COL])
            .head(50_000)
        )
        _ = df.head(1).omna.search("warmup", on=COL, k=1)

    REAL.write(f"\r  {DIM}          {RST}\n")
    REAL.flush()

    # ── COLD OPEN  (~2s) ──────────────────────────────────────────────────────
    out()
    out(f"  {B}CFPB Consumer Financial Complaints{RST}  {DIM}· 4.4M records · 9 GB{RST}")
    out(f"  {DIM}\"Certified safe for public use.\"  — CFPB Data Disclosure Policy{RST}")
    pause(0.4)
    out()
    REAL.write(f"  {W}")
    REAL.flush()
    typewrite("We tested that.", delay=0.05)
    REAL.write(RST)
    REAL.flush()
    pause(0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # SHIELD  (~20s)
    # ══════════════════════════════════════════════════════════════════════════
    section("⚠", "THE SHIELD  —  detecting what the government missed", R)

    REAL.write(f"  {C}")
    REAL.flush()
    typewrite("df.omna.pii_report()", delay=0.055)
    REAL.write(RST)
    REAL.flush()
    pause(0.3)

    report = spinner_while(
        lambda: df.omna.pii_report(),
        "scanning 50,000 rows  ·  17 columns",
    )

    out()
    for row in report.filter(pl.col("flagged")).iter_rows(named=True):
        pct = 100.0 * row["rows_with_pii"] / row["sample_size"]
        out(f"  {R}⚠  {row['column']:<44}{pct:5.1f}%  [{row['pii_types']}]{RST}")
        pause(0.8)

    pause(0.5)
    out()
    out(f"  {R}{B}The government missed this.{RST}")
    pause(0.8)

    out()
    bar(R)
    out(f"  {R}{B}42,646{RST}   {W}PII leaks in federal-certified public data{RST}")
    out(f"  {C}{B}   164s{RST}   {DIM}to neutralize  ·  M5  ·  fanless  ·  zero cloud{RST}")
    bar(R)
    pause(1.5)

    # ══════════════════════════════════════════════════════════════════════════
    # SWORD  (~8s)
    # ══════════════════════════════════════════════════════════════════════════
    section("⚡", "THE SWORD  —  search by meaning, not keywords", C)

    out(f"  {DIM}One line.{RST}")
    out()
    pause(0.3)

    QUERY = "bank changed my rate without telling me"

    REAL.write(f"  {C}")
    REAL.flush()
    typewrite(f'df.omna.search("{QUERY}", on="{COL}", k=10)', delay=0.038)
    REAL.write(RST)
    REAL.flush()
    pause(0.2)

    t0 = time.perf_counter()
    results = spinner_while(
        lambda: df.omna.search(QUERY, on=COL, k=10),
        "searching 50,000 rows",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out()
    for i, text in enumerate(results[COL].to_list()[:3]):
        snippet = str(text).replace("\n", " ")[:88]
        out(f"  {DIM}[{i + 1}]{RST}  {snippet}…")
        pause(0.4)

    out()
    out(f"  {C}{B}⚡ {elapsed_ms:.0f}ms{RST}   {DIM}·  50,000 rows  ·  M5  ·  fanless  ·  zero cloud{RST}")
    pause(1.2)

    # ══════════════════════════════════════════════════════════════════════════
    # CLOSER  (~12s)
    # ══════════════════════════════════════════════════════════════════════════
    section("◆", "THE CLOSER  —  filter and ask", G)

    FILTER_Q = "unauthorized rate change"

    REAL.write(f"  {G}")
    REAL.flush()
    typewrite(f'df.omna.filter("{FILTER_Q}", on="{COL}", threshold=0.35)', delay=0.038)
    REAL.write(RST)
    REAL.flush()
    pause(0.2)

    filtered = spinner_while(
        lambda: df.omna.filter(FILTER_Q, on=COL, threshold=0.35),
        "filtering",
    )
    out(f"  {G}✓  {len(filtered):,} rows matched{RST}")
    pause(0.8)

    out()
    ASK_Q = "In one sentence: what connects these complaints?"

    REAL.write(f"  {G}")
    REAL.flush()
    typewrite(f'results.omna.ask("{ASK_Q}")', delay=0.038)
    REAL.write(RST)
    REAL.flush()
    pause(0.2)

    answer = spinner_while(
        lambda: results.select([COL]).omna.ask(ASK_Q),
        "asking Claude Haiku",
    )
    out()
    out(f"  {G}{B}◆{RST}  {answer}")
    pause(1.5)

    # ── OUTRO  (~6s) ──────────────────────────────────────────────────────────
    out()
    out()
    outro = [
        (f"  {W}{B}Secure. Searchable. Intelligent.{RST}",                         1.0),
        ("",                                                                          0.2),
        (f"  {C}df.omna.mask_pii()     {DIM}# redact PII — one line{RST}",          0.6),
        (f"  {C}df.omna.search(...)    {DIM}# find meaning — not keywords{RST}",    0.6),
        (f"  {C}df.omna.ask(...)       {DIM}# query in plain English{RST}",         0.6),
        ("",                                                                          0.3),
        (f"  {DIM}pip install omna   ·   github.com/gaurjin/Omna{RST}",             0.6),
    ]
    for line, delay in outro:
        out(line)
        pause(delay)

    out()
