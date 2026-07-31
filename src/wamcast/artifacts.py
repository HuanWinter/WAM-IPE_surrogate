"""Artifact registry + downloader for Zenodo-hosted checkpoints and demo data.

Populated at release time (see docs/zenodo.md). Users get artifacts via:

    from wamcast.artifacts import ARTIFACTS, download
    ckpt = download(ARTIFACTS["ensemble_t16_member_00"])
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    sha256: str
    size_bytes: int


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "wamcast"


def download(artifact: Artifact,
             cache_dir: Path | None = None,
             chunk_size: int = 1024 * 1024) -> Path:
    """Download an artifact to cache_dir, verifying SHA-256. If the target
    file already exists with the correct hash, return the cached path."""
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / artifact.name

    if dest.exists() and _sha256(dest) == artifact.sha256:
        return dest

    tmp = dest.with_suffix(dest.suffix + ".partial")
    with requests.get(artifact.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f, tqdm(
            total=artifact.size_bytes, unit="B", unit_scale=True, desc=artifact.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    got = _sha256(tmp)
    if got != artifact.sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"{artifact.name}: hash mismatch (expected {artifact.sha256[:12]}..., "
            f"got {got[:12]}...)"
        )
    tmp.rename(dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Registry — populated at release time (Task 15). Placeholder DOI means the
# constants won't be usable until then.
# ---------------------------------------------------------------------------
_ZENODO_DOI = "10.5281/zenodo.PLACEHOLDER"
_ZENODO_BASE = "https://zenodo.org/record/PLACEHOLDER/files"

ARTIFACTS: dict[str, Artifact] = {
    # Populated by scripts/emit_zenodo_manifest.py at release time.
    # Example:
    # "ensemble_t16_member_00": Artifact(
    #     name="ensemble_t16_member_00.ckpt",
    #     url=f"{_ZENODO_BASE}/ensemble_t16_member_00.ckpt",
    #     sha256="...", size_bytes=...,
    # ),
}
