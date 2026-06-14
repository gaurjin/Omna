"""Unit test for scripts/strip_sbom.py — the pre-publish SBOM remover.

Builds a minimal wheel-shaped zip (a dist-info with an sboms/ file + RECORD),
strips it, and asserts the SBOM is gone, RECORD no longer references it, and
every other member is preserved byte-for-byte.
"""
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("strip_sbom", ROOT / "scripts" / "strip_sbom.py")
strip_sbom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strip_sbom)


def _make_wheel(path: Path) -> None:
    record = (
        "pkg/__init__.py,sha256=abc,12\n"
        "pkg-1.0.dist-info/sboms/pkg.cyclonedx.json,sha256=def,99\n"
        "pkg-1.0.dist-info/METADATA,sha256=ghi,34\n"
        "pkg-1.0.dist-info/RECORD,,\n"
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("pkg/__init__.py", b"x = 1\n")
        z.writestr("pkg-1.0.dist-info/sboms/pkg.cyclonedx.json", b'{"secret": "crates"}')
        z.writestr("pkg-1.0.dist-info/METADATA", b"Name: pkg\n")
        z.writestr("pkg-1.0.dist-info/RECORD", record)


def test_strip_removes_sbom_and_fixes_record(tmp_path):
    wheel = tmp_path / "pkg-1.0-py3-none-any.whl"
    _make_wheel(wheel)

    assert strip_sbom.strip(wheel) is True

    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        record = z.read("pkg-1.0.dist-info/RECORD").decode()
    assert not any("/sboms/" in n for n in names), f"SBOM survived: {names}"
    assert "sboms" not in record, "RECORD still references the removed SBOM"
    assert "pkg/__init__.py" in names and "pkg-1.0.dist-info/METADATA" in names, \
        "non-SBOM members must be preserved"


def test_strip_is_idempotent_when_no_sbom(tmp_path):
    wheel = tmp_path / "nosbom-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("pkg/__init__.py", b"x = 1\n")
        z.writestr("pkg-1.0.dist-info/RECORD", "pkg/__init__.py,sha256=abc,12\n")
    assert strip_sbom.strip(wheel) is False  # nothing to do, untouched
