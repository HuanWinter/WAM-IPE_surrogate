"""Tests for conformal.py: score computation, q_alpha calibration, jackknife+.

Design note: alt_bin_of_channel(c) returns a string "alt_0".."alt_3" (canonical
bin name). The partition test uses dict counting (not list indexing) to match.
"""
import numpy as np
import pytest

from wamcast.conformal import (
    ALT_BIN_EDGES, ALT_BIN_NAMES, KP_THRESHOLD,
    alt_bin_of_channel, storm_class,
    conformity_scores_per_bin,
    compute_q_alpha,
    apply_conformal,
    jackknife_plus_coverage,
)


def test_alt_bin_edges_partition_all_channels():
    """All 41 channels (indices 0..40) belong to exactly one bin."""
    counts = {name: 0 for name in ALT_BIN_NAMES}
    for c in range(41):
        counts[alt_bin_of_channel(c)] += 1
    assert counts == {"alt_0": 10, "alt_1": 10, "alt_2": 10, "alt_3": 11}, counts


def test_storm_class():
    assert storm_class(7.0) == "G3plus"
    assert storm_class(8.7) == "G3plus"
    assert storm_class(6.9) == "below"
    assert storm_class(6.0) == "below"


def test_conformity_scores_synthetic_gaussian():
    """If errors are N(0, sigma^2), scores are folded-normal with median ~0.67."""
    rng = np.random.default_rng(0)
    N, C, H, W = 50, 41, 89, 90
    sigma_true = 0.5
    mu     = np.zeros((N, C, H, W), dtype=np.float32)
    truth  = (sigma_true * rng.standard_normal((N, C, H, W))).astype(np.float32)
    sigma  = np.full((N, C, H, W), sigma_true, dtype=np.float32)
    storms = [{"storm_id": 999, "peak_kp": 6.7}]
    preds_per_storm = [dict(mu=mu, sigma=sigma, truth=truth, meta=storms[0])]

    scores = conformity_scores_per_bin(preds_per_storm)
    # scores is dict keyed by (alt_bin_name, kp_class)
    assert ("alt_0", "below") in scores
    # Folded standard normal mean is sqrt(2/pi) ~= 0.798
    assert 0.7 < scores[("alt_0", "below")].mean() < 0.9


def test_q_alpha_synthetic():
    """For standard normal abs values, q at alpha=0.95 ~= 1.96."""
    rng = np.random.default_rng(1)
    scores = {("alt_0", "below"): np.abs(rng.standard_normal(100_000))}
    q = compute_q_alpha(scores, alpha=0.95)
    assert 1.9 < q[("alt_0", "below")] < 2.0, q


def test_apply_conformal_coverage_on_held_out():
    """Coverage on held-out samples should match alpha within a few percent
    when the calibrator was fit on N samples from the same distribution."""
    rng = np.random.default_rng(2)
    N_cal, N_test = 5_000, 5_000
    sigma_true = 0.5
    # cal predictions
    cal = dict(
        mu=np.zeros((N_cal, 41, 89, 90), dtype=np.float32),
        sigma=np.full((N_cal, 41, 89, 90), sigma_true, dtype=np.float32),
        truth=(sigma_true * rng.standard_normal((N_cal, 41, 89, 90))).astype(np.float32),
        meta={"storm_id": 1, "peak_kp": 6.7},
    )
    # test predictions (held out)
    test = dict(
        mu=np.zeros((N_test, 41, 89, 90), dtype=np.float32),
        sigma=np.full((N_test, 41, 89, 90), sigma_true, dtype=np.float32),
        truth=(sigma_true * rng.standard_normal((N_test, 41, 89, 90))).astype(np.float32),
        meta={"storm_id": 2, "peak_kp": 6.7},
    )
    scores = conformity_scores_per_bin([cal])
    q = compute_q_alpha(scores, alpha=0.90)
    cov = apply_conformal([test], q)
    # cov is a single number (pooled across all test pixels)
    assert abs(cov - 0.90) < 0.02, f"coverage {cov:.3f} not within 2% of 0.90"


def test_jackknife_plus_returns_coverage_per_alpha():
    rng = np.random.default_rng(3)
    sigma_true = 0.5
    storms = []
    for sid in range(34, 39):  # 5 mini-storms
        N = 100
        storms.append(dict(
            mu=np.zeros((N, 41, 89, 90), dtype=np.float32),
            sigma=np.full((N, 41, 89, 90), sigma_true, dtype=np.float32),
            truth=(sigma_true * rng.standard_normal((N, 41, 89, 90))).astype(np.float32),
            meta={"storm_id": sid, "peak_kp": 6.7},
        ))
    cov = jackknife_plus_coverage(storms, alphas=[0.5, 0.9])
    assert set(cov.keys()) == {0.5, 0.9}
    assert 0.4 < cov[0.5] < 0.6, cov[0.5]
    assert 0.85 < cov[0.9] < 0.95, cov[0.9]


def test_conformity_scores_includes_both_storm_classes():
    """A G3plus storm and a below-G3 storm should both populate the scores dict."""
    rng = np.random.default_rng(10)
    N, C, H, W = 20, 41, 89, 90
    sigma_true = 0.5
    def _mk(peak_kp):
        mu     = np.zeros((N, C, H, W), dtype=np.float32)
        truth  = (sigma_true * rng.standard_normal((N, C, H, W))).astype(np.float32)
        sigma  = np.full((N, C, H, W), sigma_true, dtype=np.float32)
        return dict(mu=mu, sigma=sigma, truth=truth,
                    meta={"storm_id": 999, "peak_kp": peak_kp})
    preds = [_mk(6.7), _mk(8.0)]
    scores = conformity_scores_per_bin(preds)
    # All 8 cells should be populated
    expected = {(name, klass)
                for name in ["alt_0", "alt_1", "alt_2", "alt_3"]
                for klass in ["below", "G3plus"]}
    assert set(scores.keys()) == expected, set(scores.keys())


def test_jackknife_plus_with_mixed_storm_classes():
    """Jackknife+ across storms spanning both Kp classes should still report
    coverage matching nominal alpha within a few percent."""
    rng = np.random.default_rng(20)
    sigma_true = 0.5
    storms = []
    for sid, kp in [(34, 6.7), (35, 8.5), (36, 6.3), (37, 7.5), (38, 6.7)]:
        N = 100
        storms.append(dict(
            mu=np.zeros((N, 41, 89, 90), dtype=np.float32),
            sigma=np.full((N, 41, 89, 90), sigma_true, dtype=np.float32),
            truth=(sigma_true * rng.standard_normal((N, 41, 89, 90))).astype(np.float32),
            meta={"storm_id": sid, "peak_kp": kp},
        ))
    cov = jackknife_plus_coverage(storms, alphas=[0.9])
    assert 0.85 < cov[0.9] < 0.95, cov[0.9]


def test_jackknife_plus_raises_keyerror_when_held_storm_unique_class():
    """If the held-out storm is the only G3+ storm in the pool, the LOO cal set
    has no G3+ examples, so compute_q_alpha produces no G3+ keys, and
    apply_conformal raises KeyError with a diagnostic message."""
    rng = np.random.default_rng(30)
    sigma_true = 0.5
    storms = []
    for sid, kp in [(40, 6.5), (41, 8.7), (42, 6.3), (43, 6.7)]:  # 1 G3+ (sid 41)
        N = 50
        storms.append(dict(
            mu=np.zeros((N, 41, 89, 90), dtype=np.float32),
            sigma=np.full((N, 41, 89, 90), sigma_true, dtype=np.float32),
            truth=(sigma_true * rng.standard_normal((N, 41, 89, 90))).astype(np.float32),
            meta={"storm_id": sid, "peak_kp": kp},
        ))
    with pytest.raises(KeyError, match="G3plus"):
        jackknife_plus_coverage(storms, alphas=[0.9])


def test_calibrator_fit_apply_round_trip():
    """Calibrator wraps fit + apply so users don't hand-plumb scores/quantiles."""
    import numpy as np
    from wamcast.conformal import Calibrator

    rng = np.random.default_rng(0)
    cal_preds = [
        dict(
            mu=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            sigma=np.ones((4, 41, 89, 90), dtype=np.float32),
            truth=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            meta=dict(peak_kp=6.5),
        )
    ]
    c = Calibrator.fit(cal_preds, alpha=0.05)
    assert set(c.q_alpha.keys()) == {("alt_0", "below"), ("alt_1", "below"),
                                     ("alt_2", "below"), ("alt_3", "below")}
    # Apply to a new prediction — returns (lo, hi) with the correct shape
    test_pred = dict(
        mu=rng.standard_normal((2, 41, 89, 90)).astype(np.float32),
        sigma=np.ones((2, 41, 89, 90), dtype=np.float32),
        meta=dict(peak_kp=6.5),
    )
    lo, hi = c.intervals(test_pred)
    assert lo.shape == (2, 41, 89, 90)
    assert (hi >= lo).all()


def test_calibrator_save_load_round_trip(tmp_path):
    """Calibrator.save / .load must round-trip."""
    import numpy as np
    from wamcast.conformal import Calibrator

    rng = np.random.default_rng(0)
    cal_preds = [
        dict(
            mu=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            sigma=np.ones((4, 41, 89, 90), dtype=np.float32),
            truth=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            meta=dict(peak_kp=7.5),  # G3+ storm so both classes get populated eventually
        ),
        dict(
            mu=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            sigma=np.ones((4, 41, 89, 90), dtype=np.float32),
            truth=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            meta=dict(peak_kp=6.0),
        ),
    ]
    c1 = Calibrator.fit(cal_preds, alpha=0.05)
    path = tmp_path / "cal.npz"
    c1.save(path)
    c2 = Calibrator.load(path)
    assert c1.alpha == c2.alpha
    assert set(c1.q_alpha.keys()) == set(c2.q_alpha.keys())
    for k in c1.q_alpha:
        assert c1.q_alpha[k] == c2.q_alpha[k]


def test_calibrator_intervals_missing_class_raises_diagnostic():
    """intervals() must give an actionable KeyError when the fit calibrator
    lacks q_alpha for the requested storm class."""
    import numpy as np
    import pytest
    from wamcast.conformal import Calibrator

    rng = np.random.default_rng(0)
    # Cal set contains ONLY "below" storms (peak_kp < 7.0)
    cal_preds = [
        dict(
            mu=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            sigma=np.ones((4, 41, 89, 90), dtype=np.float32),
            truth=rng.standard_normal((4, 41, 89, 90)).astype(np.float32),
            meta=dict(peak_kp=5.0),
        )
    ]
    c = Calibrator.fit(cal_preds, alpha=0.05)
    # Ask for intervals on a G3+ storm — should raise a diagnostic KeyError
    test_pred = dict(
        mu=rng.standard_normal((2, 41, 89, 90)).astype(np.float32),
        sigma=np.ones((2, 41, 89, 90), dtype=np.float32),
        meta=dict(peak_kp=8.0),  # G3plus
    )
    with pytest.raises(KeyError, match="G3plus"):
        c.intervals(test_pred)
