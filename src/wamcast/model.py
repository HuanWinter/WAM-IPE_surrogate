"""Inference-only WAMCast model.

Extracted from train_camnet_spectral.py:51-108 (CAMNetSFNOSpectral) and
train_camnet_style.py:44-89 (SphericalConv, SFNOBlock). Loss functions,
Lightning training hooks, and the sat-loss module import are stripped;
only what the forward path needs at inference time is retained.

The class still inherits from pl.LightningModule so that Lightning-serialized
checkpoints (which carry the full hparams dict) round-trip via
.load_from_checkpoint(...).
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_harmonics import InverseRealSHT, RealSHT

RHO_CH_DEFAULT = 41   # matches LEV_HI - LEV_LO = 51 - 10 in train_camnet_upper.py


class SphericalConv(nn.Module):
    """Spectral conv in spherical-harmonic space. Ported unchanged from
    train_camnet_style.py:44-70."""

    def __init__(self, channels: int, modes_l: int, modes_m: int,
                 nlat: int, nlon: int) -> None:
        super().__init__()
        self.channels = channels
        self.modes_l = modes_l
        self.modes_m = modes_m
        self.sht = RealSHT(nlat, nlon, grid="equiangular")
        self.isht = InverseRealSHT(nlat, nlon, grid="equiangular")
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes_l, modes_m, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.sht(x)
        L, M = x_hat.shape[-2], x_hat.shape[-1]
        lm = min(self.modes_l, L)
        mm = min(self.modes_m, M)
        out_hat = torch.zeros_like(x_hat)
        w = torch.complex(self.weight[..., 0], self.weight[..., 1])
        out_hat[:, :, :lm, :mm] = torch.einsum(
            "bilm,iolm->bolm",
            x_hat[:, :, :lm, :mm],
            w[:, :, :lm, :mm],
        )
        return self.isht(out_hat)


class SFNOBlock(nn.Module):
    """Ported unchanged from train_camnet_style.py:73-89."""

    def __init__(self, channels: int, modes_l: int, modes_m: int,
                 nlat: int, nlon: int) -> None:
        super().__init__()
        self.spec = SphericalConv(channels, modes_l, modes_m, nlat, nlon)
        self.skip = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spec(x) + self.skip(x)
        y = F.gelu(y)
        y = y + self.mlp(self.norm(y))
        return y


class WAMCastModel(pl.LightningModule):
    """Inference-only extract of CAMNetSFNOSpectral.

    Constructor signature matches train_camnet_spectral.py:51-63 exactly so
    that .load_from_checkpoint(...) on a training-produced ckpt works. All
    loss-side hparams are kept in the signature (Lightning re-instantiates
    the class from `hparams`), but the corresponding methods are removed
    since they are never called on the forward path.
    """

    def __init__(
        self,
        rho_ch: int = RHO_CH_DEFAULT,
        n_lags: int = 2,
        aux_ch: int = 10,
        hidden_ch: int = 192,
        n_blocks: int = 6,
        modes_l: int = 16,
        modes_m: int = 16,
        nlat: int = 89,
        nlon: int = 90,
        # Loss / training hparams — retained for ckpt-load compatibility only.
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 120,
        warmup_epochs: int = 20,
        rollout_steps: int = 4,
        rollout_weight_decay: float = 0.9,
        spectral_weight: float = 0.1,
        spectral_freq_exp: float = 0.0,
        gradient_weight: float = 0.0,
        ssim_weight: float = 0.0,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
        multiscale_weight: float = 0.0,
        multiscale_scales: tuple[int, ...] = (2, 4),
        sat_loss_weight: float = 0.0,
        delay: int = 18,
        sat_loss_per_step: bool = False,
        sat_loss_k_decay: float = 0.9,
        loss_type: str = "z_mse",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        in_ch = rho_ch * (1 + n_lags)
        self.embed = nn.Conv2d(in_ch, hidden_ch, 1)
        self.blocks = nn.ModuleList([
            SFNOBlock(hidden_ch, modes_l, modes_m, nlat, nlon)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch + aux_ch, hidden_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, rho_ch, 1),
        )

        # SSIM kernel buffer — retained so ckpt load doesn't warn.
        coords = torch.arange(ssim_window, dtype=torch.float32) - (ssim_window - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * ssim_sigma ** 2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :])[None, None, :, :].expand(
            rho_ch, 1, ssim_window, ssim_window).contiguous()
        self.register_buffer("ssim_kernel", kernel)

        # Denormalization stats — populated at ckpt load if the trainer wrote them.
        self.register_buffer("mean_u", torch.zeros(rho_ch))
        self.register_buffer("std_u", torch.ones(rho_ch))

    def step(self, rho_cur: torch.Tensor, rho_lags: torch.Tensor,
             aux: torch.Tensor) -> torch.Tensor:
        """One residual forward step.

        rho_cur:  (B, C, H, W)             current z-scored density state
        rho_lags: (B, n_lags, C, H, W)     stacked historical lag snapshots
        aux:      (B, aux_ch, H, W)        static + driver aux channels
        Returns:  (B, C, H, W)             predicted z-scored density at t+delay
        """
        B, L, C, H, W = rho_lags.shape
        flat_lags = rho_lags.reshape(B, L * C, H, W)
        x = torch.cat([rho_cur, flat_lags], dim=1)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        delta = self.head(torch.cat([h, aux], dim=1))
        return rho_cur + delta

    # Note: training_step / validation_step / configure_optimizers are
    # intentionally NOT defined here. pl.LightningModule provides them as
    # base-class stubs; this inference-only extract inherits those stubs
    # unmodified rather than overriding or masking them. Overriding them
    # (e.g. with AttributeError-raising properties) would risk breaking
    # Lightning's internal code paths (including .load_from_checkpoint)
    # that may probe these attributes, and would misrepresent the class
    # shape to anyone introspecting it. The correct invariant to test is
    # "we didn't override the stubs," not "the attribute is absent" — see
    # test_no_training_hooks_overridden in tests/test_model.py.
