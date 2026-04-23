# Publishing Omna to PyPI

This document walks through the one-time setup and the exact commands to
release a new version. After setup, a release is two commands.

---

## One-time setup

### Step 1 — Create a PyPI account

Go to https://pypi.org/account/register/ and create a free account.
Verify your email address before continuing.

### Step 2 — Get an API token

1. Log in to PyPI.
2. Click your username (top right) → **Account settings**.
3. Scroll down to **API tokens** → **Add API token**.
4. Give it a name (e.g. `omna-github-actions`).
5. Set **Scope** to **Entire account** for the first publish.
   After the first successful release, you can create a project-scoped token
   with **Project: omna** and delete the account-scoped one.
6. Click **Create token**.
7. Copy the token. It starts with `pypi-`. You will not see it again — store it
   somewhere safe (a password manager, not a file in this repo).

### Step 3 — Add the token as a GitHub secret

1. Open your GitHub repository: `github.com/gaurjin/omna`
2. Click **Settings** → **Secrets and variables** → **Actions**.
3. Click **New repository secret**.
4. Name: `PYPI_API_TOKEN`
5. Value: paste the `pypi-...` token you copied in Step 2.
6. Click **Add secret**.

That's the setup done. You never need to repeat it.

---

## Releasing a new version

### Before you tag

1. Update the version number in **both** files (they must match):

   **`pyproject.toml`**
   ```toml
   [project]
   version = "0.1.0"   # ← change this
   ```

   **`Cargo.toml`**
   ```toml
   [package]
   version = "0.1.0"   # ← change this to the same value
   ```

2. Commit the version bump:
   ```bash
   git add pyproject.toml Cargo.toml
   git commit -m "Bump version to 0.1.0"
   git push
   ```

### Tag and trigger the release

```bash
git tag v0.1.0
git push origin v0.1.0
```

That's it. Pushing the tag triggers the GitHub Actions workflow automatically.

---

## What happens after you push the tag

The workflow runs these jobs in order:

```
build-macos (6 jobs)      build-linux (2 jobs)      build-sdist (1 job)
aarch64 · py3.10          x86_64 linux              source tarball
aarch64 · py3.11          aarch64 linux (QEMU)
aarch64 · py3.12
x86_64  · py3.10
x86_64  · py3.11
x86_64  · py3.12
         │                        │                        │
         └────────────────────────┴────────────────────────┘
                                  │
                            publish (1 job)
                   only runs if ALL builds pass
                   uploads 12 wheels + 1 sdist to PyPI
```

To watch it run: go to **github.com/gaurjin/omna** → **Actions** tab → click
the workflow named after your tag.

---

## Verifying the release

After the publish job completes (usually 15–20 minutes total):

```bash
pip install omna==0.1.0
python -c "import omna; print('ok')"
```

The package page will be live at https://pypi.org/project/omna/

---

## If something goes wrong

**Build fails on one platform** — the publish job will not run. Check the
failing job's logs in the Actions tab. Fix the issue, delete the tag, and
re-push:

```bash
git tag -d v0.1.0            # delete local tag
git push origin :v0.1.0      # delete remote tag
# fix the issue, commit, then re-tag
git tag v0.1.0
git push origin v0.1.0
```

**PyPI rejects the upload** — you cannot overwrite an existing version on PyPI.
If you need to fix a bad release, bump to a patch version (e.g. `v0.1.1`) and
publish again.

**Wrong API token** — double-check the secret name in GitHub is exactly
`PYPI_API_TOKEN` (no spaces, correct case). Tokens expire if you rotate them
on PyPI — update the secret if you generate a new one.
