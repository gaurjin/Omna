# Omna — CLAUDE.md

## Two-repo structure
Omna is split across two directories:
- `~/Developer/Omna Labs/Omna` — working repo. All development happens here. src/ is .gitignored.
- `~/Developer/Omna Labs/Omna-engine` — private IP archive (github.com/gaurjin/Omna-engine). Rust source only.

The compiled .so ships in the pip wheel. The Rust source never appears in the public repo.

## Rust sync rule — ALWAYS DO THIS
Any change to src/lib.rs, src/similarity.rs, or Cargo.toml must be copied to Omna-engine after testing:
```
cp src/lib.rs src/similarity.rs ~/Developer/Omna\ Labs/Omna-engine/src/
cp Cargo.toml Cargo.lock ~/Developer/Omna\ Labs/Omna-engine/
cd ~/Developer/Omna\ Labs/Omna-engine && git add -A && git commit -m "sync: <description>" && git push
```
Develop in Omna first. Sync to Omna-engine after every Rust change. Never edit Omna-engine directly.

## Coding rules
- Python first. Only write Rust in src/similarity.rs.
- Every public method must have a docstring.
- Never break the df.omna namespace interface.
- Always run `maturin develop --release` after any Rust change (never plain `maturin develop` — debug builds are 10x slower).

## Repositories
- Public (Python layer): https://github.com/gaurjin/Omna
- Private (Rust engine): https://github.com/gaurjin/Omna-engine
- PyPI account: gaurjin — published! pypi.org/project/omna (v0.1.0, 2026-04-26)

## Communication rules
- Spoon-feed every step. Never assume knowledge.
- Explain every terminal command in plain English.
- User is intermediate Python, no Rust experience.
