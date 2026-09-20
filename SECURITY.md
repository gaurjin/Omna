# Security

`omna` is a local data library. Your DataFrames, and anything masked from them, stay on your
machine. If that is ever not true, we want to know before your users do.

## Reporting a vulnerability

**Email: gaurjin@gmail.com** with `[SECURITY]` in the subject line, or use GitHub's
[private vulnerability reporting](https://github.com/gaurjin/Omna/security/advisories/new).

Please do **not** open a public issue for a security problem.

What helps:

- What you found, and what an attacker could do with it.
- The smallest set of steps that shows it.
- The versions: `python -c "import omna; print(omna.__version__)"` and
  `python -c "import omna_pii_mask; print(omna_pii_mask.version())"`.
- Your OS and Python version.

**Please do not include real personal data in a report.** A made-up value that reproduces the
problem is worth more to us than a real one.

### What to expect

| | |
|---|---|
| First reply | within 3 working days |
| Assessment | within 10 working days |
| Fix for a confirmed high-severity issue | as fast as we can, and we will tell you the date |
| Credit | your name in the release notes, unless you'd rather stay anonymous |

Honest targets from a small team, not a contractual SLA.

## What counts as a vulnerability here

**Yes, please report:**

- Any way to make the library send data anywhere. It should make no outbound connections at all
  except downloading the detection model on first use of `model=True`.
- Any path that writes a real value to disk that the caller did not ask for.
- Any way to make `restore()` return a value the caller never masked.
- Anything that would let a swapped detection model go unnoticed.
- Path traversal, arbitrary code execution, or unsafe deserialisation in any file the library reads.

**Known and documented, not a vulnerability:**

- **Detection misses.** No PII detector catches everything. Please report these as normal issues,
  with a fake example — they are quality bugs, not security holes.
- **Anything already running as you.** A process with your user rights can read your files and your
  process memory. No local library can prevent that.
- **`mask()` output is not anonymised data.** It is masked data. Re-identification from context,
  from what was *not* detected, or from combining columns is possible. Treat the output as
  "reduced risk", never as "safe to publish".

## Scope

In scope: this repository and the `omna` package on PyPI.

Out of scope: `omna-pii-mask` (the compiled engine — report those here anyway and we will route
them), Polars, NumPy and other dependencies. Report those upstream too, but tell us so we can pin.

## Supply chain

- Releases are published to PyPI from CI (`release-pypi.yml`), never from a laptop.
- The masking engine `omna-pii-mask` is a compiled wheel built from a private kernel. It is
  binary-only by design; the detection logic is not open source.
- The detection model is downloaded from Hugging Face on first use and verified against a
  SHA-256 pinned inside the engine before it is loaded.
