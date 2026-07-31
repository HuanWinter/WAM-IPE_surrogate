"""CLI tests via click.testing.CliRunner."""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip(
    "torch_harmonics",
    reason="torch_harmonics C extension unavailable (typical on CI without a matching torch ABI)",
)

import numpy as np  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from wamcast.cli import main  # noqa: E402


def test_cli_help_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for sub in ("forecast", "calibrate", "download-artifacts"):
        assert sub in result.output


def test_cli_version_prints_package_version():
    from wamcast import __version__
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_calibrate_end_to_end(tmp_path):
    """calibrate subcommand: given cal-pred NPZs, produce a Calibrator NPZ."""
    rng = np.random.default_rng(0)
    cal_paths = []
    for i in range(2):
        p = tmp_path / f"cal_{i}.npz"
        np.savez(
            p,
            mu=rng.standard_normal((3, 41, 89, 90)).astype(np.float32),
            sigma=np.ones((3, 41, 89, 90), dtype=np.float32),
            truth=rng.standard_normal((3, 41, 89, 90)).astype(np.float32),
            meta=dict(peak_kp=6.0 + i),
        )
        cal_paths.append(str(p))
    out = tmp_path / "calibrator.npz"
    runner = CliRunner()
    result = runner.invoke(main, [
        "calibrate",
        *sum([["--cal-preds", p] for p in cal_paths], []),
        "--alpha", "0.05",
        "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()

    # Verify the saved calibrator loads and has the expected shape
    from wamcast.conformal import Calibrator
    c = Calibrator.load(out)
    assert c.alpha == 0.05
    assert len(c.q_alpha) > 0


def test_cli_download_artifacts_empty_registry_errors():
    """With no ARTIFACTS registered (Task 15 pending), download should error clearly."""
    runner = CliRunner()
    result = runner.invoke(main, ["download-artifacts"])
    assert result.exit_code != 0
    assert "no artifacts" in result.output.lower() or "zenodo" in result.output.lower()
