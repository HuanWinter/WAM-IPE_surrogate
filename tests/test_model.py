"""Tests for the inference-only WAMCast model extract."""
from __future__ import annotations

import pathlib

import pytest
import torch

from wamcast.model import WAMCastModel


def test_model_instantiates_with_default_hparams():
    m = WAMCastModel()
    assert m.hparams.rho_ch == 41
    assert m.hparams.n_lags == 2
    assert m.hparams.aux_ch == 10
    # buffers registered so ckpt loading won't complain
    assert hasattr(m, "mean_u")
    assert hasattr(m, "std_u")
    assert hasattr(m, "ssim_kernel")


def test_step_forward_pass_shape():
    m = WAMCastModel().eval()
    B, C, H, W = 2, 41, 89, 90
    rho_cur = torch.randn(B, C, H, W)
    rho_lags = torch.randn(B, 2, C, H, W)  # (B, n_lags, C, H, W)
    aux = torch.randn(B, 10, H, W)
    with torch.no_grad():
        out = m.step(rho_cur, rho_lags, aux)
    assert out.shape == (B, C, H, W)
    assert torch.isfinite(out).all()


def test_step_is_residual():
    """When the head output is forced to zero, step() must return rho_cur exactly.

    This is the residual-formulation invariant: step(x, ...) = x + head_output.
    A non-residual (or raw-delta) implementation would return zero, not rho_cur.
    """
    m = WAMCastModel().eval()
    # Zero the head's final layer (Conv2d(hidden_ch, rho_ch, 1)) so it outputs
    # exactly zero regardless of input. head is a Sequential of 3 modules;
    # the last one is the Conv2d that produces delta.
    final_conv = m.head[-1]
    with torch.no_grad():
        final_conv.weight.zero_()
        final_conv.bias.zero_()
    rho_cur = torch.randn(1, 41, 89, 90)
    rho_lags = torch.randn(1, 2, 41, 89, 90)  # non-zero to exercise the pipeline
    aux = torch.randn(1, 10, 89, 90)
    with torch.no_grad():
        out = m.step(rho_cur, rho_lags, aux)
    assert torch.allclose(out, rho_cur, atol=1e-6), \
        f"step() must be residual; max diff = {(out - rho_cur).abs().max():.6f}"


def test_no_training_hooks_overridden():
    """The extract must not override Lightning's training-hook stubs.

    pl.LightningModule defines training_step, validation_step,
    configure_optimizers as base stubs. Any real trainer subclass overrides
    them; an inference-only extract should leave them as inherited stubs.
    """
    import pytorch_lightning as pl
    for name in ("training_step", "validation_step", "configure_optimizers"):
        assert getattr(WAMCastModel, name) is getattr(pl.LightningModule, name), \
            f"WAMCastModel.{name} overrides the base stub — training baggage leaked in"


def test_no_loss_methods_present():
    """The loss / rollout methods from CAMNetSFNOSpectral must all be stripped."""
    for name in (
        "_spectral_loss", "_ssim_loss", "_gradient_loss",
        "_multiscale_loss", "_combined_loss", "_rollout_loss",
        "_rollout_loss_with_preds",
        "_single_step_loss", "_denorm",
    ):
        assert not hasattr(WAMCastModel, name), f"loss method leaked: {name}"


REAL_CKPT = pathlib.Path("/media/faraday/andong/Workspace/WAM-IPE/Res/uq/ensemble_t16/member_00/best.ckpt")


@pytest.mark.skipif(not REAL_CKPT.exists(), reason="real ckpt not present in CI")
@pytest.mark.xfail(
    reason=(
        "Root-caused via control experiment: the research repo's own "
        "CAMNetSFNOSpectral.load_from_checkpoint(...) fails on this same "
        "ckpt in this same environment with the identical error "
        "(_pickle.UnpicklingError: Weights only load failed ... "
        "Unsupported global: numpy._core.multiarray.scalar). This is a "
        "torch>=2.6 weights_only=True default vs. a ckpt pickled with "
        "numpy scalars issue, not a WAMCastModel extraction bug. The "
        "manuscript's headline numbers were produced under torch<2.6; the "
        "Zenodo release (Task 15) will need either a torch.load shim/"
        "safe_globals allowlist or re-minted checkpoints."
    ),
    strict=True,
)
def test_load_real_training_checkpoint():
    m = WAMCastModel.load_from_checkpoint(str(REAL_CKPT), map_location="cpu")
    assert m.hparams.rho_ch == 41
    # mean_u/std_u should have been overwritten from the ckpt state_dict
    assert not torch.allclose(m.mean_u, torch.zeros_like(m.mean_u))
