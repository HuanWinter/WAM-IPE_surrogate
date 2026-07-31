"""Tests for artifact downloader (uses a mocked HTTP server)."""
from __future__ import annotations

import hashlib
import http.server
import os
import pathlib
import socketserver
import threading

import pytest

from wamcast.artifacts import Artifact, download


@pytest.fixture
def local_server(tmp_path):
    """Serve tmp_path over HTTP so download() can pull a fake artifact."""
    original_dir = pathlib.Path.cwd()
    os.chdir(tmp_path)
    handler = http.server.SimpleHTTPRequestHandler
    # Bind to 0 for an ephemeral port
    with socketserver.TCPServer(("localhost", 0), handler) as srv:
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        port = srv.server_address[1]
        try:
            yield tmp_path, f"http://localhost:{port}"
        finally:
            srv.shutdown()
            os.chdir(original_dir)


def test_download_verifies_sha256(local_server, tmp_path):
    tmp, base_url = local_server
    payload = b"fake checkpoint bytes"
    (tmp / "member_00.ckpt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    a = Artifact(name="member_00.ckpt",
                 url=f"{base_url}/member_00.ckpt",
                 sha256=sha, size_bytes=len(payload))
    dest = tmp_path / "cache"
    got = download(a, cache_dir=dest)
    assert got.read_bytes() == payload
    assert got == dest / "member_00.ckpt"


def test_download_raises_on_hash_mismatch(local_server, tmp_path):
    tmp, base_url = local_server
    (tmp / "bad.ckpt").write_bytes(b"corrupted")
    a = Artifact(name="bad.ckpt",
                 url=f"{base_url}/bad.ckpt",
                 sha256="0" * 64, size_bytes=9)
    with pytest.raises(ValueError, match="hash mismatch"):
        download(a, cache_dir=tmp_path / "cache")


def test_download_uses_cache_when_present(local_server, tmp_path):
    tmp, base_url = local_server
    payload = b"cached"
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "cached.ckpt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    a = Artifact(name="cached.ckpt",
                 url=f"{base_url}/does-not-exist.ckpt",
                 sha256=sha, size_bytes=len(payload))
    # Server has no such URL; cache-hit path returns without hitting server.
    got = download(a, cache_dir=dest)
    assert got.read_bytes() == payload


def test_artifacts_registry_is_declared_but_empty():
    """The registry is populated at Task 15 (Zenodo mint). At Task 8 it should
    exist as an empty dict so downstream code can import it without fabricating.
    """
    from wamcast.artifacts import ARTIFACTS
    assert isinstance(ARTIFACTS, dict)
    # Deliberately empty until Zenodo release cut
