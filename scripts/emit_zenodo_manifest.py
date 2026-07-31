"""Compute SHA-256 + size for every artifact in a directory and emit the
ARTIFACTS dict to paste into src/wamcast/artifacts.py.

Usage:
    python scripts/emit_zenodo_manifest.py \\
        --artifacts-dir /path/to/staged/artifacts \\
        --zenodo-record <ZENODO_RECORD_NUMBER>

Output: prints a Python snippet to stdout. Redirect to a file or paste directly
into the ARTIFACTS block in artifacts.py.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True, type=Path)
    ap.add_argument("--zenodo-record", required=True,
                    help="Zenodo record number (from the DOI, after the last '.').")
    args = ap.parse_args()

    if not args.artifacts_dir.is_dir():
        raise SystemExit(f"artifacts-dir does not exist: {args.artifacts_dir}")

    files = sorted(p for p in args.artifacts_dir.iterdir() if p.is_file())
    if not files:
        raise SystemExit(f"no files in {args.artifacts_dir}")

    print(f'_ZENODO_DOI = "10.5281/zenodo.{args.zenodo_record}"')
    print(f'_ZENODO_BASE = "https://zenodo.org/record/{args.zenodo_record}/files"')
    print()
    print("ARTIFACTS: dict[str, Artifact] = {")
    for p in files:
        key = p.stem
        print(f'    "{key}": Artifact(')
        print(f'        name="{p.name}",')
        print(f'        url=f"{{_ZENODO_BASE}}/{p.name}",')
        print(f'        sha256="{sha256(p)}",')
        print(f'        size_bytes={p.stat().st_size},')
        print(f'    ),')
    print("}")


if __name__ == "__main__":
    main()
