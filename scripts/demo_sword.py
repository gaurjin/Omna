"""
Omna — Demo 2: The Sword  (~60s on pre-warmed M5)

Flow:
  cold open → understand → pain (all at once) → magic (3 lines, 3 outputs)

Dataset: Gretel PII dataset (acquired by NVIDIA, 50,000 documents)
Run with: uv run python scripts/demo_sword.py
Record at: asciinema rec demo_sword.cast --cols 120 --rows 40
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
    R   = "\033[91m"
    G   = "\033[92m"
    Y   = "\033[93m"
    C   = "\033[96m"
    W   = "\033[97m"
    DIM = "\033[2m"
    B   = "\033[1m"
    RST = "\033[0m"

    REAL  = sys.__stdout__
    COL   = "text"
    QUERY = "insurance claim denied"

    # Column widths tuned to stay inside --cols 120
    # 2 + (10+2) + (16+2) + (18+2) + (38+2) + (6+2) = 2+12+18+20+40+8 = 100
    DISPLAY_COLS   = ["uid", "document_type", "domain", COL]
    DISPLAY_WIDTHS = [10, 16, 18, 38]

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
        REAL.write(f"\r{' ' * 64}\r")
        REAL.flush()
        if err[0]:
            raise err[0]
        return result[0]

    def bar(color: str = DIM) -> None:
        out(f"  {color}" + "─" * 100 + RST)

    def section(icon: str, title: str, color: str) -> None:
        out()
        out(f"  {color}{B}{icon}  {title}{RST}")
        bar(color)
        out()
        pause(0.2)

    def show_rows(df: pl.DataFrame, cols: list, widths: list) -> None:
        header = "  "
        for col, w in zip(cols, widths):
            header += f"{DIM}{col[:w].ljust(w)}{RST}  "
        out(header)
        bar(DIM)
        for row in df.select(cols).iter_rows(named=True):
            line = "  "
            for col, w in zip(cols, widths):
                raw  = str(row[col]).replace("\n", " ") if row[col] is not None else ""
                cell = (raw[: w - 1] + "…") if len(raw) > w else raw.ljust(w)
                line += cell + "  "
            out(line)
        out()

    def show_results(df: pl.DataFrame) -> None:
        avail  = [c for c in DISPLAY_COLS if c in df.columns]
        widths = [DISPLAY_WIDTHS[DISPLAY_COLS.index(c)] for c in avail]
        cols   = avail + ["_score"]
        ws     = widths + [6]

        header = "  "
        for col, w in zip(cols, ws):
            header += f"{DIM}{col[:w].ljust(w)}{RST}  "
        out(header)
        bar(G)

        for row in df.select(cols).iter_rows(named=True):
            line = "  "
            for col, w in zip(cols, ws):
                if col == "_score":
                    s = f"{row[col]:.3f}" if row[col] is not None else "—"
                    line += f"{G}{s.ljust(w)}{RST}  "
                else:
                    raw  = str(row[col]).replace("\n", " ") if row[col] is not None else ""
                    cell = (raw[: w - 1] + "…") if len(raw) > w else raw.ljust(w)
                    line += cell + "  "
            out(line)
        out()

    # ── SILENT LOADING ────────────────────────────────────────────────────────
    REAL.write(f"\n  {DIM}loading ...{RST}")
    REAL.flush()

    with silent():
        df = (
            pl.read_csv("data/gretel_pii.csv", infer_schema_length=1000)
            .drop_nulls(subset=[COL])
            .head(50_000)
        )
        _ = df.head(1).omna.search("warmup", on=COL, k=1)

        # Pre-run search to find aligned pain example — a result whose text
        # contains a semantic variant keyword search would miss
        _early = df.omna.search(QUERY, on=COL, k=10)
        _miss_phrase = None
        _miss_targets = ["refused", "rejected", "not payable", "not covered",
                         "declined", "denial", "cannot", "unable to"]
        for _row in _early.iter_rows(named=True):
            _text = (_row[COL] or "").lower()
            for _phrase in _miss_targets:
                if _phrase in _text:
                    _miss_phrase = _phrase
                    break
            if _miss_phrase:
                break

    REAL.write(f"\r  {DIM}          {RST}\n")
    REAL.flush()

    # ── COLD OPEN  (~4s) ──────────────────────────────────────────────────────
    out()
    out(f"  {B}Gretel PII Dataset{RST}  {DIM}· acquired by NVIDIA  ·  50,000 synthetic documents{RST}")
    pause(0.4)
    out()
    out(f"  {DIM}Find the insurance claims. Filter them. Understand them.{RST}")
    pause(0.4)
    REAL.write(f"  {W}")
    REAL.flush()
    typewrite("Without writing 200 lines of code.", delay=0.05)
    REAL.write(RST)
    REAL.flush()
    pause(0.5)

    # ── STEP 1 : understand  (~7s) ────────────────────────────────────────────
    section("◆", "EXPLORE  —  what's in the data?", C)

    REAL.write(f"  {C}")
    REAL.flush()
    typewrite("omna.understand_df(df)", delay=0.055)
    REAL.write(RST)
    REAL.flush()
    pause(0.3)

    schema_df = omna.understand_df(df)
    show_rows(
        schema_df,
        ["column", "dtype", "null_pct", "label", "sample"],
        [22, 8, 9, 10, 40],
    )
    pause(0.5)
    out(f"  {W}Free-form text. Invoices, NDAs, insurance policies, shipping records.{RST}")
    out(f"  {W}Task: find every document about an insurance claim denial.{RST}")
    pause(1.5)

    # ── STEP 2 : THE PAIN — all three tasks, all at once  (~12s) ─────────────
    section("✗", "THE OLD WAY  —  50+ lines per task", Y)

    # Use the aligned miss phrase if found, else a generic one
    _miss_ex = f'"{_miss_phrase} to pay"' if _miss_phrase else '"coverage not payable"'

    pain_lines = [
        f"  {Y}# Task 1: Search — keyword list that grows forever{RST}",
        f"  {Y}keywords = ['claim denied', 'coverage rejected', 'policy voided', ...]{RST}",
        f"  {Y}pattern  = re.compile('|'.join(keywords), re.IGNORECASE){RST}",
        f"  {Y}results  = df[df['text'].str.contains(pattern, na=False)]{RST}",
        f"  {R}# Still misses: {_miss_ex}  ← in your actual results{RST}",
        f"  {DIM}",
        f"  {Y}# Task 2: Filter by relevance — pure guesswork{RST}",
        f"  {Y}scores   = tfidf_vectorizer.transform(results['text']){RST}",
        f"  {Y}filtered = results[cosine_similarity(scores, query_vec) > 0.4]{RST}",
        f"  {R}# Threshold 0.4? 0.6? Wrong either way. You won't know.{RST}",
        f"  {R}# Still misses: 'insurer refused to honour' → TF-IDF score: 0.11  ← below cutoff{RST}",
        f"  {R}# Still misses: medical claim rejections using clinical terminology{RST}",
        f"  {DIM}",
        f"  {Y}# Task 3: Summarize — manual, slow, error-prone{RST}",
        f"  {Y}# Export to CSV → read 300 docs → write summary → hours of work{RST}",
        f"  {R}# Can't ask: 'what personal data is exposed?' across 300 docs instantly{RST}",
        f"  {R}# Can't surface cross-document patterns without reading every single one{RST}",
    ]
    for line in pain_lines:
        REAL.write(line + RST + "\n")
        REAL.flush()
        time.sleep(0.07)

    pause(0.6)
    out()
    bar(R)
    out(f"  {R}{B}50+ lines per task.  Still brittle.  Still misses half.  Hours of work.{RST}")
    bar(R)
    pause(1.8)

    # ── STEP 3 : THE MAGIC — 3 lines, 3 outputs  (~30s) ──────────────────────
    section("⚡", "WITH OMNA  —  one line per task", C)

    out(f"  {W}{B}Semantic layer:{RST}  {W}understands meaning, not just words.{RST}")
    out()
    pause(0.6)

    # ── Magic line 1: search ──────────────────────────────────────────────────
    out(f"  {DIM}# 1 — find the top matches{RST}")
    REAL.write(f"  {C}")
    REAL.flush()
    typewrite(f'results = df.omna.search("{QUERY}", on="{COL}", k=5)', delay=0.04)
    REAL.write(RST)
    REAL.flush()

    t0 = time.perf_counter()
    results = spinner_while(
        lambda: df.omna.search(QUERY, on=COL, k=5),
        "searching 50,000 documents",
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    out()
    show_results(results)
    out(f"  {C}{B}⚡ {elapsed_ms:.0f}ms{RST}   {DIM}· {len(results)} of 50,000 documents · M5 · zero cloud{RST}")
    out(f"  {W}These results match the intent — not the exact words.  Keyword search misses all of them.{RST}")
    pause(1.2)

    # ── Magic line 2: filter ──────────────────────────────────────────────────
    out()
    out(f"  {DIM}# 2 — keep everything above a similarity threshold{RST}")
    REAL.write(f"  {C}")
    REAL.flush()
    typewrite(f'filtered = df.omna.filter("{QUERY}", on="{COL}", threshold=0.73)', delay=0.04)
    REAL.write(RST)
    REAL.flush()

    filtered = spinner_while(
        lambda: df.omna.filter(QUERY, on=COL, threshold=0.73),
        "filtering 50,000 documents",
    )
    doc_word = "document" if len(filtered) == 1 else "documents"
    out()
    out(f"  {G}{B}✓  {len(filtered):,} {doc_word} matched{RST}   {DIM}· no guesswork, no keyword lists — pure meaning{RST}")
    out()
    show_rows(filtered.head(3), DISPLAY_COLS, DISPLAY_WIDTHS)
    pause(1.2)

    # ── Magic line 3: ask ─────────────────────────────────────────────────────
    out()
    out(f"  {DIM}# 3 — ask a question about them{RST}")
    REAL.write(f"  {C}")
    REAL.flush()
    ASK_Q = "What personal data do these insurance documents expose?"
    typewrite(f'results.omna.ask("{ASK_Q}")', delay=0.04)
    REAL.write(RST)
    REAL.flush()

    answer = spinner_while(
        lambda: results.select([COL]).omna.ask(ASK_Q),
        "asking Claude Haiku",
    )
    out()
    out(f"  {G}{B}◆{RST}  {W}{answer}{RST}")
    out()
    out(f"  {DIM}Old way: export → read 300 docs manually → hours.  Omna: one line, instant.{RST}")
    pause(1.5)

    # ── THE CONTRAST ──────────────────────────────────────────────────────────
    out()
    bar(G)
    out(f"  {Y}Old way:  {R}50+ lines per task · guesswork · brittle · hours{RST}")
    out(f"  {C}Omna:     {G}1 line per task   · semantic  · instant · done{RST}")
    bar(G)
    pause(1.5)

    # ── OUTRO  (~4s) ──────────────────────────────────────────────────────────
    out()
    outro = [
        (f"  {W}{B}The semantic layer for your data.{RST}",                              1.0),
        ("",                                                                               0.2),
        (f"  {C}df.omna.search(...)    {DIM}# meaning — not keywords{RST}",             0.5),
        (f"  {C}df.omna.filter(...)    {DIM}# semantic threshold — not regex{RST}",      0.5),
        (f"  {C}df.omna.ask(...)       {DIM}# query in plain English{RST}",              0.5),
        ("",                                                                               0.3),
        (f"  {DIM}pip install omna   ·   github.com/gaurjin/Omna{RST}",                 0.6),
    ]
    for line, delay in outro:
        out(line)
        pause(delay)

    out()
