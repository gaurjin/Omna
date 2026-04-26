"""
PII Diagnostic — runs mask_pii on 500 rows and saves a side-by-side CSV.

Use this to verify:
  1. mask_pii() is actually redacting real PII (not false-positives)
  2. How many rows genuinely changed
  3. What the before/after looks like

Run with: uv run python scripts/pii_diagnostic.py
Output:   data/pii_diagnostic.csv
"""
import polars as pl
import omna

COL = "Consumer complaint narrative"

if __name__ == "__main__":

    print("Loading data...")
    df = (
        pl.read_csv(
            "data/cfpb/complaints.csv",
            n_rows=10_000,
            infer_schema_length=10_000,
        )
        .drop_nulls(subset=[COL])
        .head(500)
    )
    print(f"  {len(df)} rows loaded.")

    print("\nRunning pii_report()...")
    report = df.omna.pii_report()
    print(report.select(["column", "pii_types", "rows_with_pii", "flagged"]))

    print("\nRunning mask_pii()...")
    masked = df.omna.mask_pii()

    orig_vals   = df[COL].to_list()
    masked_vals = masked[COL].to_list()
    ids         = (
        df["Complaint ID"].to_list()
        if "Complaint ID" in df.columns
        else list(range(len(df)))
    )

    changed_rows = [
        {
            "Complaint ID":    ids[i],
            "original":        orig_vals[i],
            "masked":          masked_vals[i],
            "redaction_count": (masked_vals[i] or "").count("<REDACTED>"),
        }
        for i in range(len(orig_vals))
        if orig_vals[i] != masked_vals[i]
    ]

    unchanged_count = len(orig_vals) - len(changed_rows)

    print(f"\n{'='*60}")
    print(f"Total rows:       {len(df)}")
    print(f"Rows changed:     {len(changed_rows)}  ({100*len(changed_rows)/len(df):.1f}%)")
    print(f"Rows unchanged:   {unchanged_count}  ({100*unchanged_count/len(df):.1f}%)")
    if changed_rows:
        total = sum(r["redaction_count"] for r in changed_rows)
        print(f"Total <REDACTED>: {total}")

    print(f"\n{'='*60}")
    print("First 10 changed rows:\n")
    for r in changed_rows[:10]:
        print(f"  ID:     {r['Complaint ID']}")
        print(f"  Before: {r['original'][:120]}")
        print(f"  After:  {r['masked'][:120]}")
        print()

    # Save full comparison
    comparison_df = pl.DataFrame({
        "Complaint ID":       ids,
        "changed":            [o != m for o, m in zip(orig_vals, masked_vals)],
        "redaction_count":    [(m or "").count("<REDACTED>") for m in masked_vals],
        "original_narrative": orig_vals,
        "masked_narrative":   masked_vals,
    })
    out_path = "data/pii_diagnostic.csv"
    comparison_df.write_csv(out_path)
    print(f"Full comparison saved to: {out_path}")
