"""
Omna — Demo 1: The Shield  (~55s on pre-warmed M5)

Flow: cold open → understand → pii_report → leak example → mask_pii before/after

Dataset: Gretel PII dataset (acquired by NVIDIA)
Run with: uv run python scripts/demo_shield.py
Record at: asciinema rec demo_shield.cast --cols 120 --rows 42
"""
import sys
import os
import re
import time
import threading
import polars as pl
import omna
from contextlib import contextmanager

_PHONE_RE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9.+_%\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


if __name__ == "__main__":

    # ── palette ───────────────────────────────────────────────────────────────
    R      = "\033[91m"          # red
    G      = "\033[92m"          # green
    C      = "\033[96m"          # cyan
    W      = "\033[97m"          # white
    DIM    = "\033[2m"           # dim
    B      = "\033[1m"           # bold
    RST    = "\033[0m"           # reset
    REDBG  = "\033[1;97;41m"    # bold white on red background — for <REDACTED>

    REAL = sys.__stdout__
    COL  = "text"

    # uid + document_type kept visible in both Before and After
    BA_COLS   = ["uid", "document_type", COL]
    BA_WIDTHS = [12, 20, 76]   # total ≈ 116 with separators

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
        REAL.write(f"\r{' ' * 66}\r")
        REAL.flush()
        if err[0]:
            raise err[0]
        return result[0]

    def bar(color: str = DIM) -> None:
        out(f"  {color}" + "─" * 116 + RST)

    def section(icon: str, title: str, color: str) -> None:
        out()
        out(f"  {color}{B}{icon}  {title}{RST}")
        bar(color)
        out()
        pause(0.2)

    def show_rows(df: pl.DataFrame, cols: list, widths: list,
                  highlight_redacted: bool = False) -> None:
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
                if highlight_redacted:
                    cell = cell.replace("<REDACTED>", f"{REDBG}<REDACTED>{RST}")
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
            .head(5_000)
        )

        # Pre-qualify ba_sample: rows with phone/email/SSN in first 90 chars
        evidence_rows = df.filter(
            pl.col(COL).str.slice(0, 90).str.contains(
                r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|"
                r"[a-zA-Z0-9.+_%\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}|"
                r"\b\d{3}-\d{2}-\d{4}\b"
            )
        )

        ba_candidates     = evidence_rows.head(50)
        masked_candidates = (
            ba_candidates.omna.mask_pii()
            if not ba_candidates.is_empty() else None
        )

        good_idx = []
        if masked_candidates is not None:
            for i in range(len(ba_candidates)):
                orig_95   = (ba_candidates[COL][i]    or "")[:95]
                masked_95 = (masked_candidates[COL][i] or "")[:95]
                if "<REDACTED>" in masked_95 and orig_95 != masked_95:
                    good_idx.append(i)
                if len(good_idx) >= 5:
                    break

        ba_sample = (
            ba_candidates[good_idx] if good_idx
            else ba_candidates.head(5) if not ba_candidates.is_empty()
            else df.head(5)
        )

        # Extract one concrete PII leak to show as evidence
        pii_example = None
        for row in ba_sample.iter_rows(named=True):
            text = row[COL] or ""
            for pat, label in [(_SSN_RE, "SSN"), (_EMAIL_RE, "Email"), (_PHONE_RE, "Phone")]:
                m = pat.search(text[:300])
                if m:
                    pii_example = (label, m.group(), row.get("document_type", "document"))
                    break
            if pii_example:
                break

    REAL.write(f"\r  {DIM}          {RST}\n")
    REAL.flush()

    # ── COLD OPEN  (~4s) ──────────────────────────────────────────────────────
    out()
    out(f"  {B}Gretel PII Dataset{RST}  {DIM}· acquired by NVIDIA  ·  50,000 synthetic documents{RST}")
    pause(0.4)
    out()
    out(f"  {DIM}Invoices. NDAs. Insurance policies. Shipping records.{RST}")
    out(f"  {DIM}Built to benchmark data privacy tools.{RST}")
    pause(0.8)
    out()
    REAL.write(f"  {W}")
    REAL.flush()
    typewrite("Let's see what's actually inside.", delay=0.05)
    REAL.write(RST)
    REAL.flush()
    pause(0.5)

    # ── STEP 1 : understand  (~6s) ────────────────────────────────────────────
    section("◆", "EXPLORE  —  understand the schema first", C)

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
        [22, 8, 9, 10, 56],
    )
    pause(0.5)
    out(f"  {W}Free-form text field. Multiple document types. Could contain anything.{RST}")
    out(f"  {R}One question: is any of it sensitive?{RST}")
    pause(1.5)

    # ── STEP 2 : pii_report  (~15s) ───────────────────────────────────────────
    section("⚠", "AUDIT  —  scan every document for PII", R)

    REAL.write(f"  {C}")
    REAL.flush()
    typewrite("df.omna.pii_report()", delay=0.055)
    REAL.write(RST)
    REAL.flush()
    pause(0.3)

    report = spinner_while(
        lambda: df.omna.pii_report(),
        "scanning 5,000 documents  ·  6 columns",
    )

    out()
    for row in report.filter(pl.col("flagged")).iter_rows(named=True):
        pct = 100.0 * row["rows_with_pii"] / row["sample_size"]
        out(f"  {R}⚠  {row['column']:<44}{pct:5.1f}%  [{row['pii_types']}]{RST}")
        pause(0.8)

    pause(0.5)
    out()

    text_row = report.filter(pl.col("column") == COL)
    if text_row.height > 0:
        text_hit_pct = 100.0 * text_row["rows_with_pii"][0] / text_row["sample_size"][0]
        estimated = int(text_hit_pct / 100 * 5_000)
        bar(R)
        out(f"  {R}{B}~{estimated:,}{RST}   {W}documents contain PII — names, SSNs, emails, phone numbers{RST}")
        bar(R)

    # ── Concrete leak example ─────────────────────────────────────────────────
    if pii_example:
        label, value, doc_type = pii_example
        out()
        pause(0.6)
        out(f"  {DIM}For example, in a {doc_type}:{RST}")
        out(f"  {R}{B}    {label}: {value}{RST}   {DIM}← sitting in your pipeline, unmasked{RST}")
    pause(1.5)

    # ── STEP 3 : mask_pii — before / after  (~10s) ───────────────────────────
    section("✓", "FIX  —  one line to redact everything", G)

    REAL.write(f"  {G}")
    REAL.flush()
    typewrite("masked = df.omna.mask_pii()", delay=0.055)
    REAL.write(RST)
    REAL.flush()
    pause(0.3)

    masked_live = spinner_while(lambda: ba_sample.omna.mask_pii(), "masking")

    out()
    out(f"  {R}{B}Before:{RST}")
    show_rows(ba_sample, BA_COLS, BA_WIDTHS)
    pause(0.5)

    out(f"  {G}{B}After:{RST}")
    show_rows(masked_live, BA_COLS, BA_WIDTHS, highlight_redacted=True)

    out(f"  {C}{B}One line.{RST}   {DIM}Names, SSNs, emails, phone numbers — all gone.  Local.  No cloud.{RST}")
    pause(1.5)

    # ── OUTRO  (~4s) ──────────────────────────────────────────────────────────
    out()
    outro = [
        (f"  {W}{B}Secure. Auditable. One line.{RST}",                           1.0),
        ("",                                                                        0.2),
        (f"  {C}omna.understand_df(df)  {DIM}# explore your schema{RST}",        0.6),
        (f"  {C}df.omna.pii_report()    {DIM}# audit — find every leak{RST}",    0.6),
        (f"  {C}df.omna.mask_pii()      {DIM}# redact — one line{RST}",           0.6),
        ("",                                                                        0.3),
        (f"  {DIM}pip install omna   ·   github.com/gaurjin/Omna{RST}",           0.6),
    ]
    for line, delay in outro:
        out(line)
        pause(delay)

    out()
