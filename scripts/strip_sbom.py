"""Strip the auto-generated SBOM from a built wheel before publishing.

maturin bundles a CycloneDX SBOM at `<dist-info>/sboms/*.json`. For a binary-
only, source-closed distribution that publicly lists every internal crate name
and dependency — more than we want to advertise. This removes those entries and
keeps RECORD consistent. Everything else in the wheel is copied byte-for-byte.

Usage:  python strip_sbom.py <wheel-or-glob> [<wheel> ...]
Idempotent: a wheel with no SBOM is left unchanged.
"""
import sys
import zipfile
from glob import glob
from pathlib import Path


def strip(wheel: Path) -> bool:
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        sbom = [n for n in names if "/sboms/" in n and ".dist-info/" in n]
        if not sbom:
            return False
        record_name = next(n for n in names if n.endswith(".dist-info/RECORD"))
        data = {n: z.read(n) for n in names if n not in sbom and n != record_name}
        # Rewrite RECORD: drop lines pointing at the removed SBOM files.
        kept = [
            line for line in z.read(record_name).decode().splitlines()
            if line.split(",")[0] not in sbom
        ]
        data[record_name] = ("\n".join(kept) + "\n").encode()

    tmp = wheel.with_suffix(".whl.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, payload in data.items():
            z.writestr(name, payload)
    tmp.replace(wheel)
    print(f"stripped {len(sbom)} SBOM file(s) from {wheel.name}")
    return True


if __name__ == "__main__":
    targets = [Path(p) for arg in sys.argv[1:] for p in glob(arg)]
    if not targets:
        sys.exit("no wheels matched")
    for w in targets:
        if not strip(w):
            print(f"no SBOM in {w.name} (unchanged)")
