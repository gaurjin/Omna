"""
Omna smoke test on real(ish) data.
Run with: python scripts/smoke_test.py

This script exercises every public Omna method end-to-end.
If it all prints without errors, you're ready to record the demo.
"""
import omna
import polars as pl

print("=" * 60)
print("OMNA SMOKE TEST")
print("=" * 60)

# ── Load the dataset we generated with demo_data.py ──────────
df = pl.read_parquet("scripts/employees.parquet")
print(f"\n📋 Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── 1. understand() ──────────────────────────────────────────
print("── 1. omna.understand(df) ──────────────────────────────")
schema = omna.understand(df)
print(schema)

# ── 2. embed() ───────────────────────────────────────────────
print("\n── 2. df.omna.embed('bio') ──────────────────────────────")
df.omna.embed("bio")
print("✅ Embeddings saved to disk")

# ── 3. search() ──────────────────────────────────────────────
print("\n── 3. df.omna.search('machine learning', on='bio', k=3) ─")
results = df.omna.search("machine learning", on="bio", k=3)
print(results.select(["name", "role", "bio"]))

# ── 4. filter() ──────────────────────────────────────────────
print("\n── 4. df.omna.filter('cloud infrastructure', on='bio') ──")
infra = df.omna.filter("cloud infrastructure", on="bio")
print(infra.select(["name", "role", "bio"]))

# ── 5. pii_report() ──────────────────────────────────────────
print("\n── 5. df.omna.pii_report() ──────────────────────────────")
report = df.omna.pii_report()
print(report)

# ── 6. mask_pii() ────────────────────────────────────────────
print("\n── 6. df.omna.mask_pii() ────────────────────────────────")
masked = df.omna.mask_pii()
print(masked.select(["name", "email", "phone"]).head(5))

# ── 7. ask() ─────────────────────────────────────────────────
print("\n── 7. df.omna.ask('Who earns over 130k?') ───────────────")
answer = df.omna.ask("Who earns over 130k?")
print(answer)

print("\n" + "=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
