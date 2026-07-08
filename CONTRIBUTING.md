# Contributing to Omna

Thanks for your interest in Omna — a local-first semantic search + PII-masking
plugin for Polars DataFrames. Contributions of every size are welcome: bug
reports, docs, tests, and code.

## Ways to help

- **Report a bug** — open an issue with a minimal `polars` DataFrame that
  reproduces it, your OS + Python version, and `pip show omna` output.
- **Improve the docs** — the README and `docs/` are always open to clarifications.
- **Pick up a `good first issue`** — issues tagged this way are scoped to be
  approachable without deep knowledge of the internals.
- **Add a test** — anything in `tests/` that pins down current behavior helps.

## Project layout (what you can work on)

Omna is a Python package with a compiled core:

- `omna/` — the Python layer (search, filter, PII routing, schema understanding,
  hybrid BM25/RRF, LLM `ask`). **This is where most contributions live.**
- The Rust similarity kernel and PII engine ship as **prebuilt wheels**
  (`omna`, `omna-pii-mask`) — you do **not** need a Rust toolchain to work on
  the Python layer.

## Development setup

```bash
# 1. Fork and clone
git clone https://github.com/<you>/Omna.git
cd Omna

# 2. Install with all extras (pulls the compiled engine wheels)
pip install -e ".[all]"

# 3. Run the test suite
pytest tests/ -q
```

- Supported: Python 3.10–3.12 on macOS (Apple Silicon) or Linux.
- The on-device PII model (`mask_pii(model=True)`) downloads once on first use.

## Pull requests

1. Branch off `main`.
2. Keep the change focused; add or update a test in `tests/`.
3. Every public method keeps its docstring.
4. Never break the `df.omna.*` namespace API.
5. Run `pytest tests/ -q` and make sure it's green before opening the PR.

## Code of conduct

Be kind and constructive. We want Omna to be a welcoming project.

## License

By contributing, you agree that your contributions to the Python layer are
licensed under the MIT License (see `LICENSE`).
