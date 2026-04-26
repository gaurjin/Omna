"""Public df.omna namespace — all user-facing DataFrame methods live here."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from omna import index
from _omna import top_k_flat, top_k_flat_np  # noqa: F401

_AUDIT_PATH = Path(".omna") / "pii_audit.parquet"


def _default_index_path(column: str) -> Path:
    return Path(".omna") / f"{column}.parquet"


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------

def _print_search(result: pl.DataFrame, query: str, on: str, k: int) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    n = len(result)
    header = f"  Omna — Semantic Search  │  \"{query}\"  │  on: {on}  │  top {n}"
    console.print(Panel(header, box=box.SQUARE, expand=False))
    console.print()

    if n == 0:
        console.print("  [dim]No results found.[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=True,
        padding=(0, 1),
    )
    non_score_cols = [c for c in result.columns if c != "_score"]
    for col in non_score_cols:
        table.add_column(col)
    table.add_column("score", justify="right")

    for row in result.iter_rows(named=True):
        score = row.get("_score", 0.0) or 0.0
        if score >= 0.85:
            score_str = f"[bold green]{score:.3f}[/bold green]"
        elif score >= 0.65:
            score_str = f"[yellow]{score:.3f}[/yellow]"
        else:
            score_str = f"[dim]{score:.3f}[/dim]"

        cells = []
        for col in non_score_cols:
            val = str(row[col]) if row[col] is not None else ""
            cells.append(val[:45] + "…" if len(val) > 45 else val)
        cells.append(score_str)
        table.add_row(*cells)

    console.print(table)


def _print_filter(result: pl.DataFrame, concept: str, on: str, threshold: float) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    n = len(result)
    match_word = "match" if n == 1 else "matches"
    header = (
        f"  Omna — Filter  │  \"{concept}\"  ·  on: {on}"
        f"  ·  ≥{threshold:.2f}  ·  {n} {match_word}"
    )
    console.print(Panel(header, box=box.SQUARE, expand=False))
    console.print()

    if n == 0:
        console.print("  [dim]No rows matched the threshold.[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=True,
        padding=(0, 1),
    )
    for col in result.columns:
        table.add_column(col)

    for row in result.iter_rows(named=True):
        cells = []
        for col in result.columns:
            val = str(row[col]) if row[col] is not None else ""
            cells.append(val[:50] + "…" if len(val) > 50 else val)
        table.add_row(*cells)

    console.print(table)


def _print_pii_report(result: pl.DataFrame) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    n_cols = len(result)
    flagged = result.filter(pl.col("flagged")).height if n_cols > 0 else 0

    col_word = "column" if n_cols == 1 else "columns"
    header = f"  Omna — PII Report  │  {n_cols} {col_word} scanned  │  {flagged} flagged"
    console.print(Panel(header, box=box.SQUARE, expand=False))
    console.print()

    if n_cols == 0:
        console.print("  [dim]No string columns to scan.[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=True,
        padding=(0, 1),
    )
    table.add_column("column")
    table.add_column("detected types")
    table.add_column("sample", justify="right")
    table.add_column("hits", justify="right")
    table.add_column("hit rate", justify="right")
    table.add_column("avg conf", justify="right")
    table.add_column("flagged", justify="center")

    for row in result.iter_rows(named=True):
        is_flagged = row["flagged"]

        # ── flagged badge ──────────────────────────────────────────────────
        flag_str = "[bold red]✓ YES[/bold red]" if is_flagged else "[green]— no[/green]"

        # ── PII types: one per line so nothing is truncated ────────────────
        types_raw = row["pii_types"]
        if types_raw:
            types_display = "\n".join(t.strip() for t in types_raw.split(","))
        else:
            types_display = "[dim]—[/dim]"

        # ── hits ───────────────────────────────────────────────────────────
        hits_str = (
            f"[bold red]{row['rows_with_pii']}[/bold red]"
            if is_flagged else f"[dim]{row['rows_with_pii']}[/dim]"
        )

        # ── hit rate %, color-coded ────────────────────────────────────────
        sample_n = row["sample_size"] or 1
        hit_pct = 100.0 * row["rows_with_pii"] / sample_n
        hit_pct_str = f"{hit_pct:.1f}%"
        if hit_pct > 30:
            hit_rate_display = f"[bold red]{hit_pct_str}[/bold red]"
        elif hit_pct > 10:
            hit_rate_display = f"[yellow]{hit_pct_str}[/yellow]"
        else:
            hit_rate_display = f"[green]{hit_pct_str}[/green]"

        # ── avg confidence ─────────────────────────────────────────────────
        avg_conf = row.get("avg_confidence") or 0.0
        if avg_conf == 0.0:
            conf_display = "[dim]—[/dim]"
        elif avg_conf >= 0.85:
            conf_display = f"[bold green]{avg_conf:.2f}[/bold green]"
        elif avg_conf >= 0.7:
            conf_display = f"[yellow]{avg_conf:.2f}[/yellow]"
        else:
            conf_display = f"[dim]{avg_conf:.2f}[/dim]"

        table.add_row(
            row["column"],
            types_display,
            str(row["sample_size"]),
            hits_str,
            hit_rate_display,
            conf_display,
            flag_str,
        )

    console.print(table)


def _print_mask_pii(original: pl.DataFrame, masked: pl.DataFrame) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    audit_rows = []
    for col in original.columns:
        if original[col].dtype != pl.Utf8:
            continue
        orig_vals = original[col].to_list()
        new_vals = masked[col].to_list()
        changed = sum(1 for a, b in zip(orig_vals, new_vals) if a != b)
        if changed > 0:
            audit_rows.append({
                "column": col,
                "rows_scanned": len(orig_vals),
                "rows_masked": changed,
            })

    total_changed = sum(r["rows_masked"] for r in audit_rows)
    n_cols = len(audit_rows)
    col_word = "column" if n_cols == 1 else "columns"
    header = (
        f"  Omna — PII Mask  │  {n_cols} {col_word} redacted"
        f"  │  {total_changed:,} cells changed"
    )
    console.print(Panel(header, box=box.SQUARE, expand=False))
    console.print()

    if not audit_rows:
        console.print("  [green]No PII detected — DataFrame unchanged.[/green]")
        return

    # ── Audit summary table ────────────────────────────────────────────────────
    summary = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=True,
        padding=(0, 1),
    )
    summary.add_column("column")
    summary.add_column("rows scanned", justify="right")
    summary.add_column("cells redacted", justify="right")
    summary.add_column("rate", justify="right")

    for r in audit_rows:
        rate = 100.0 * r["rows_masked"] / r["rows_scanned"] if r["rows_scanned"] else 0.0
        rate_fmt = f"{rate:.1f}%"
        if rate > 50:
            rate_str = f"[bold red]{rate_fmt}[/bold red]"
        elif rate > 10:
            rate_str = f"[yellow]{rate_fmt}[/yellow]"
        else:
            rate_str = rate_fmt
        summary.add_row(
            r["column"],
            f"{r['rows_scanned']:,}",
            f"[bold red]{r['rows_masked']:,}[/bold red]",
            rate_str,
        )

    console.print(summary)

    # ── Colored preview of masked columns ─────────────────────────────────────
    redacted_cols = [r["column"] for r in audit_rows]
    n_preview = min(len(masked), 10)
    console.print()
    console.print(f"  [bold]Preview[/bold] — [bold red]<REDACTED>[/bold red] marks replaced cells  ({n_preview} rows shown)")
    console.print()

    preview = Table(
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=True,
        padding=(0, 1),
    )
    for col in redacted_cols:
        preview.add_column(col)

    for row in masked.head(n_preview).iter_rows(named=True):
        cells = []
        for col in redacted_cols:
            val = row[col]
            if val is None:
                cells.append("[dim]null[/dim]")
                continue
            val = str(val)
            # Truncate before applying markup so we never cut inside a tag
            truncated = val[:55] + "…" if len(val) > 55 else val
            colored = truncated.replace(
                "<REDACTED>", "[bold red on dark_red]<REDACTED>[/bold red on dark_red]"
            )
            cells.append(colored)
        preview.add_row(*cells)

    console.print(preview)


def _print_ask(question: str, answer: str, model: str) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    header = f"  Omna — Ask  │  {model}"
    console.print(Panel(header, box=box.SQUARE, expand=False))
    console.print()

    q_text = Text()
    q_text.append("  Q  ", style="bold white on blue")
    q_text.append(f"  {question}")
    console.print(q_text)
    console.print()

    console.print(Panel(answer, box=box.ROUNDED, expand=False, border_style="dim cyan"))


# ---------------------------------------------------------------------------
# OmnaFrame namespace
# ---------------------------------------------------------------------------

@pl.api.register_dataframe_namespace("omna")
class OmnaFrame:
    """Omna namespace attached to every Polars DataFrame as df.omna."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def embed(self, column: str, index_path: str | Path | None = None) -> pl.DataFrame:
        """Vectorize *column* and persist the embedding index to disk.

        The index is saved as a Parquet file at .omna/{column}.parquet (or
        *index_path* if given). Re-running overwrites the existing index.
        Call this once; search() and filter() read the saved file.

        Args:
            column: Name of the string column to embed.
            index_path: Override the default save location.

        Returns:
            The original DataFrame unchanged.
        """
        from omna import embedder
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        texts = self._df[column].cast(pl.String).to_list()
        vectors = embedder.embed_texts(texts)
        path = Path(index_path) if index_path else _default_index_path(column)
        index.save(self._df, vectors, path)
        return self._df

    def search(self, query: str, on: str, k: int = 10,
               index_path: str | Path | None = None) -> pl.DataFrame:
        """Return the *k* rows most semantically similar to *query*.

        Requires df.omna.embed(on) to have been called first.

        Args:
            query: Natural-language search string.
            on: Column name that was previously embedded.
            k: Number of results (default 10).
            index_path: Override the default index location.

        Returns:
            DataFrame of the top k matching rows plus a '_score' column (0-1).
        """
        from omna import embedder
        path = Path(index_path) if index_path else _default_index_path(on)
        if not path.exists():
            raise FileNotFoundError(
                f"No index for column '{on}'. Run df.omna.embed('{on}') first."
            )
        df, embeddings = index.load(path)
        query_vec = np.array(embedder.embed([query])[0], dtype=np.float32)
        dim = embeddings.shape[1]
        flat_emb = np.ascontiguousarray(embeddings)
        hits = top_k_flat_np(query_vec, flat_emb, dim, k)
        if not hits:
            result = df.clear()
            _print_search(result, query, on, k)
            return result
        result = df[list(h[0] for h in hits)].with_columns(
            pl.Series("_score", [h[1] for h in hits], dtype=pl.Float32)
        )
        _print_search(result, query, on, k)
        return result

    def filter(self, concept: str, on: str, threshold: float = 0.3,
               index_path: str | Path | None = None) -> pl.DataFrame:
        """Keep rows whose *on* column semantically matches *concept*.

        Requires df.omna.embed(on) to have been called first.

        Args:
            concept: Concept or phrase to match against.
            on: Column name that was previously embedded.
            threshold: Minimum cosine similarity score to keep a row (default 0.3).
            index_path: Override the default index location.

        Returns:
            Filtered DataFrame sorted by similarity, no score column.
        """
        from omna import embedder
        path = Path(index_path) if index_path else _default_index_path(on)
        if not path.exists():
            raise FileNotFoundError(
                f"No index for column '{on}'. Run df.omna.embed('{on}') first."
            )
        df, embeddings = index.load(path)
        concept_vec = np.array(embedder.embed([concept])[0], dtype=np.float32)
        dim = embeddings.shape[1]
        flat_emb = np.ascontiguousarray(embeddings)
        hits = top_k_flat_np(concept_vec, flat_emb, dim, len(df))
        above = [h for h in hits if h[1] >= threshold]
        if not above:
            result = df.clear()
            _print_filter(result, concept, on, threshold)
            return result
        result = df[[h[0] for h in above]]
        _print_filter(result, concept, on, threshold)
        return result

    def mask_pii(
        self,
        audit_path: str | Path | None = None,
        fast: bool = False,
    ) -> pl.DataFrame:
        """Redact PII in all string columns and save an audit log to disk.

        Args:
            audit_path: Override the default audit log location.
            fast: If True, use regex-only masking (no spaCy NER). Catches
                email, phone, SSN, credit card, URL — misses person names in
                prose. Typically 10-50x faster on long-text columns.
                Default False (full Presidio + spaCy, maximum recall).

        Returns:
            New DataFrame with PII redacted. The original is not modified.
        """
        from omna import pii
        masked_df = pii.mask(self._df, audit_path=audit_path, fast=fast)
        _print_mask_pii(self._df, masked_df)
        return masked_df

    def pii_report(self) -> pl.DataFrame:
        """Scan all string columns and return a PII findings report.

        Returns:
            DataFrame of PII findings. Empty (same schema) when none found.
        """
        from omna import pii
        result = pii.report(self._df)
        _print_pii_report(result)
        return result

    def ask(self, question: str, model: str | None = None) -> str:
        """Answer a natural-language question about this DataFrame using Claude.

        Args:
            question: Any natural-language question about the data.
            model: Claude model ID. Defaults to claude-haiku-4-5-20251001.

        Returns:
            Claude's answer as a string.
        """
        from omna import ask as ask_mod
        kwargs = {"model": model} if model else {}
        answer = ask_mod.query(self._df, question, **kwargs)
        _print_ask(question, answer, model or ask_mod.DEFAULT_MODEL)
        return answer
