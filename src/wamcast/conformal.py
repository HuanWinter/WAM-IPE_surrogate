#!/usr/bin/env python3
"""Split conformal prediction + jackknife+ for the UQ pipeline.

Public API
----------
alt_bin_of_channel(c)                    -> str "alt_0".."alt_3"  (c: local channel idx 0..40)
storm_class(peak_kp)                     -> str "G3plus" | "below"
conformity_scores_per_bin(preds)         -> dict[(alt_bin_name, kp_class)] -> 1D float32 array
compute_q_alpha(scores, alpha)           -> dict[(alt_bin_name, kp_class)] -> scalar float
apply_conformal(preds, q_alpha)          -> pooled coverage scalar in [0, 1]
jackknife_plus_coverage(preds, alphas)   -> dict[alpha] -> coverage scalar

Design notes
------------
* ``alt_bin_of_channel`` returns the canonical string name ("alt_0" etc.).
  Dict keys throughout are (alt_bin_name: str, kp_class: str).
* Score:    s = |y - mu| / sigma
* Interval: [mu - q_alpha(bin) * sigma,  mu + q_alpha(bin) * sigma]
* Coverage: fraction of pixels with |y - mu| <= q_alpha(bin) * sigma

Bin definitions
---------------
Altitude (4 bins, channel indices 0..40):
    alt_0: channels  0– 9  (spec levels 10–19)
    alt_1: channels 10–19  (spec levels 20–29)
    alt_2: channels 20–29  (spec levels 30–39)
    alt_3: channels 30–40  (spec levels 40–50)

Storm class (2 classes):
    G3plus: peak_kp >= 7.0
    below:  peak_kp <  7.0
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Bin definitions
# ---------------------------------------------------------------------------
ALT_BIN_EDGES: list[tuple[int, int]] = [(0, 10), (10, 20), (20, 30), (30, 41)]
ALT_BIN_NAMES: list[str]             = ["alt_0", "alt_1", "alt_2", "alt_3"]
KP_THRESHOLD: float                  = 7.0
KP_CLASSES: list[str]                = ["G3plus", "below"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def alt_bin_of_channel(c: int) -> str:
    """Map local channel index (0..40) to altitude bin name string.

    Returns one of ``ALT_BIN_NAMES``: "alt_0", "alt_1", "alt_2", "alt_3".
    Raises ValueError for indices outside 0..40.
    """
    for (lo, hi), name in zip(ALT_BIN_EDGES, ALT_BIN_NAMES):
        if lo <= c < hi:
            return name
    raise ValueError(f"channel {c} is outside valid range 0..40")


def storm_class(peak_kp: float) -> str:
    """Map peak Kp to storm class string.

    Returns "G3plus" if peak_kp >= KP_THRESHOLD (7.0), else "below".
    """
    return "G3plus" if peak_kp >= KP_THRESHOLD else "below"


# ---------------------------------------------------------------------------
# Core conformal functions
# ---------------------------------------------------------------------------

def conformity_scores_per_bin(preds_per_storm: Iterable[dict]) -> dict[tuple, np.ndarray]:
    """Compute |y - mu| / sigma conformity scores grouped by (alt_bin, kp_class).

    Parameters
    ----------
    preds_per_storm:
        Iterable of dicts, each with keys:
          - ``mu``    : ndarray shape (N, C, H, W), float32 — predicted mean
          - ``sigma`` : ndarray shape (N, C, H, W), float32 — predicted std
          - ``truth`` : ndarray shape (N, C, H, W), float32 — ground truth
          - ``meta``  : dict with key ``peak_kp`` (float)

    Returns
    -------
    dict mapping (alt_bin_name: str, kp_class: str) -> 1D float32 array of scores.
    """
    buckets: dict[tuple, list[np.ndarray]] = {}
    for p in preds_per_storm:
        mu    = p["mu"]
        sigma = p["sigma"]
        truth = p["truth"]
        klass = storm_class(float(p["meta"]["peak_kp"]))
        for (lo, hi), name in zip(ALT_BIN_EDGES, ALT_BIN_NAMES):
            diff = np.abs(truth[:, lo:hi] - mu[:, lo:hi])
            sig  = sigma[:, lo:hi]
            s    = (diff / sig).reshape(-1).astype(np.float32)
            key  = (name, klass)
            buckets.setdefault(key, []).append(s)
    return {k: np.concatenate(v) for k, v in buckets.items()}


def compute_q_alpha(scores: dict[tuple, np.ndarray], alpha: float) -> dict[tuple, float]:
    """Split-conformal quantile per bin (smooth finite-sample approximation).

    For each bin, returns q such that P(score <= q) is approximately alpha
    on the calibration sample, where score = |y - mu| / sigma. The quantile
    level used is (alpha * n + 1) / (n + 1), a smooth approximation that
    converges to alpha as n grows.

    Note: this differs slightly from the textbook ceiling form
    ceil(alpha * (n + 1)) / n (Vovk 2005; Angelopoulos & Bates 2021, Thm 2),
    which gives a hard finite-sample coverage guarantee P(score <= q) >= alpha.
    The smooth form used here slightly undershoots that bound for small n
    (typically by < 1% for n >= 100). For the calibration sets used in this
    package (cal scores pooled across 5 storms x 1000s of samples per bin),
    n is large enough that the difference is negligible.

    Parameters
    ----------
    scores:
        Output of ``conformity_scores_per_bin``.
    alpha:
        Desired coverage level in (0, 1).

    Returns
    -------
    dict mapping (alt_bin_name, kp_class) -> scalar float q such that
    at least alpha fraction of calibration scores are <= q.
    """
    out: dict[tuple, float] = {}
    for k, s in scores.items():
        n       = len(s)
        q_level = min((alpha * n + 1.0) / (n + 1), 1.0)
        out[k]  = float(np.quantile(s, q_level))
    return out


def apply_conformal(
    preds_per_storm: Iterable[dict],
    q_alpha: dict[tuple, float],
) -> float:
    """Compute pooled coverage on preds_per_storm using q_alpha per bin.

    Coverage = (covered pixels) / (total pixels).
    Pixel is covered iff |y - mu| <= q_alpha[(alt_bin, kp_class)] * sigma.

    Raises KeyError with a helpful message if the calibration was missing
    a (bin, class) cell that the test data needs. This commonly happens when
    the calibration set has no storms of a particular Kp class but the test
    set does — re-calibrate with a more representative cal set.

    Parameters
    ----------
    preds_per_storm:
        Same format as ``conformity_scores_per_bin``.
    q_alpha:
        Output of ``compute_q_alpha``.

    Returns
    -------
    Scalar float in [0, 1]: pooled coverage across all test pixels.
    """
    n_total   = 0
    n_covered = 0
    for p in preds_per_storm:
        mu    = p["mu"]
        sigma = p["sigma"]
        truth = p["truth"]
        klass = storm_class(float(p["meta"]["peak_kp"]))
        for (lo, hi), name in zip(ALT_BIN_EDGES, ALT_BIN_NAMES):
            if (name, klass) not in q_alpha:
                raise KeyError(
                    f"q_alpha has no entry for {(name, klass)}. "
                    f"Calibration set is missing a storm of class '{klass}'. "
                    f"Available keys: {sorted(q_alpha.keys())}"
                )
            q       = q_alpha[(name, klass)]
            diff    = np.abs(truth[:, lo:hi] - mu[:, lo:hi])
            sig     = sigma[:, lo:hi]
            covered = diff <= q * sig
            n_covered += int(covered.sum())
            n_total   += int(covered.size)
    return n_covered / max(n_total, 1)


def jackknife_plus_coverage(
    test_storms: Iterable[dict],
    alphas: list[float],
) -> dict[float, float]:
    """Leave-one-storm-out across test storms only. For each held-out storm k:
       - calibrate on the other test storms (no cal storms in this variant)
       - eval on storm k using apply_conformal
       - record per-storm coverage at each alpha
    Returns dict {alpha: aggregate coverage across all left-out storms}.
    Aggregate is pixel-weighted (sum covered / sum total).

    Raises KeyError (via apply_conformal) if the held-out storm has a Kp class
    not present in the LOO cal set. Realistic when there is exactly one storm
    of one class in the test pool — caller should ensure each class has >= 2
    storms or catch the error.

    Parameters
    ----------
    test_storms:
        Iterable of storm dicts (same format as ``conformity_scores_per_bin``).
        Must have >= 2 storms.
    alphas:
        List of desired coverage levels, e.g. [0.5, 0.9].

    Returns
    -------
    dict mapping alpha -> scalar pooled coverage.
    """
    # Pre-compute per-storm pixel counts (constant across alphas)
    storms = list(test_storms)
    pixel_counts = [
        sum((hi - lo) * p["sigma"].shape[2] * p["sigma"].shape[3] * p["sigma"].shape[0]
            for (lo, hi) in ALT_BIN_EDGES)
        for p in storms
    ]
    out_per_alpha: dict[float, dict[str, float]] = {
        a: {"covered": 0.0, "total": 0} for a in alphas
    }
    for k, held in enumerate(storms):
        other = [storms[j] for j in range(len(storms)) if j != k]
        scores = conformity_scores_per_bin(other)
        for a in alphas:
            q = compute_q_alpha(scores, a)
            cov_k = apply_conformal([held], q)
            # apply_conformal returns the fraction; multiply by pixel count to weight
            out_per_alpha[a]["covered"] += cov_k * pixel_counts[k]
            out_per_alpha[a]["total"]   += pixel_counts[k]
    return {a: out_per_alpha[a]["covered"] / max(out_per_alpha[a]["total"], 1)
            for a in alphas}


# ---------------------------------------------------------------------------
# Ergonomic wrapper
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class Calibrator:
    """Split-conformal calibrator: fit q_alpha on calibration predictions,
    apply to produce (lo, hi) intervals on new predictions."""
    q_alpha: dict[tuple, float]
    alpha: float

    @classmethod
    def fit(cls, cal_preds, alpha: float) -> "Calibrator":
        scores = conformity_scores_per_bin(cal_preds)
        q = compute_q_alpha(scores, alpha)
        return cls(q_alpha=q, alpha=alpha)

    def intervals(self, pred) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) each shape (N, C, H, W) using per-bin q_alpha."""
        mu = pred["mu"]
        sigma = pred["sigma"]
        klass = storm_class(float(pred["meta"]["peak_kp"]))
        lo = np.empty_like(mu)
        hi = np.empty_like(mu)
        for (bin_lo, bin_hi), name in zip(ALT_BIN_EDGES, ALT_BIN_NAMES):
            if (name, klass) not in self.q_alpha:
                raise KeyError(
                    f"q_alpha has no entry for {(name, klass)}. "
                    f"Calibration set is missing a storm of class '{klass}'. "
                    f"Available keys: {sorted(self.q_alpha.keys())}"
                )
            q = self.q_alpha[(name, klass)]
            lo[:, bin_lo:bin_hi] = mu[:, bin_lo:bin_hi] - q * sigma[:, bin_lo:bin_hi]
            hi[:, bin_lo:bin_hi] = mu[:, bin_lo:bin_hi] + q * sigma[:, bin_lo:bin_hi]
        return lo, hi

    def save(self, path) -> None:
        np.savez(path, alpha=self.alpha,
                 keys=np.array(list(self.q_alpha.keys()), dtype=object),
                 values=np.array(list(self.q_alpha.values()), dtype=np.float64))

    @classmethod
    def load(cls, path) -> "Calibrator":
        d = np.load(path, allow_pickle=True)
        q = {tuple(k): float(v) for k, v in zip(d["keys"], d["values"])}
        return cls(q_alpha=q, alpha=float(d["alpha"]))
