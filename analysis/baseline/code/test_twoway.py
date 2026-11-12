"""Simulation check of OLS/ORIV point estimates and two-way clustered standard errors.

Data mimic the Study 2 structure: focal players with 20 outcome matches, four teammates per match,
some matches shared by two focal players (teammates share rows), focal-level and match-level shocks,
and habits measured with error in two halves.
"""
from __future__ import annotations

import numpy as np

from config import child_seed
from twoway import oneway_covariance, ols, oriv, twoway_covariance

BETA = -0.05
RELIABILITY_HALF = 0.75


def simulate(rng: np.random.Generator, n_focal: int = 2000, per_focal: int = 20, share: float = 0.15):
    true = rng.normal(size=n_focal)
    noise_sd = np.sqrt((1 - RELIABILITY_HALF) / RELIABILITY_HALF)
    ta = true + rng.normal(scale=noise_sd, size=n_focal)
    tb = true + rng.normal(scale=noise_sd, size=n_focal)
    focal_shock = rng.normal(scale=0.3, size=n_focal)
    units_f = np.repeat(np.arange(n_focal), per_focal)
    n_units = len(units_f)
    match = np.arange(n_units)
    paired = rng.random(n_units) < share
    idx = np.flatnonzero(paired)
    idx = idx[: len(idx) // 2 * 2]
    rng.shuffle(idx)
    a, b = idx[0::2], idx[1::2]
    keep = units_f[a] != units_f[b]
    a, b = a[keep], b[keep]
    match[b] = match[a]
    teammates_pair = rng.random(len(a)) < 0.45
    match_shock = rng.normal(scale=0.5, size=n_units)
    # Teammate identities: four per unit; for teammate pairs the units share three teammates.
    mate = np.arange(n_units * 4).reshape(n_units, 4)
    mate[b[teammates_pair], :3] = mate[a[teammates_pair], :3]
    mate_shock = rng.normal(scale=0.7, size=n_units * 4)
    rows_unit = np.repeat(np.arange(n_units), 4)
    rows_mate = mate.ravel()
    f = units_f[rows_unit]
    m = match[rows_unit]
    # Teammates shared by two focal players in the same match receive both focal influences.
    exposure_sum = np.zeros(n_units * 4)
    np.add.at(exposure_sum, rows_mate, true[f] * BETA + focal_shock[f])
    y = exposure_sum[rows_mate] + match_shock[m] + mate_shock[rows_mate]
    return y, true[f], ta[f], tb[f], f, m


def main() -> None:
    rng = np.random.default_rng(child_seed("twoway_simulation"))
    reps = 300
    records = []
    for _ in range(reps):
        y, true, ta, tb, f, m = simulate(rng)
        y = y - y.mean()
        full = (ta + tb) / 2
        full = full - full.mean()
        ta_c, tb_c = ta - ta.mean(), tb - tb.mean()
        e_ols = ols(y, full, ["full"], f, m)
        e_oriv = oriv(y, ta_c, tb_c, "oriv", f, m)
        v2 = twoway_covariance([e_ols, e_oriv])
        v_f = oneway_covariance([e_ols, e_oriv], "focal")
        records.append([e_ols.beta[0], e_oriv.beta[0], np.sqrt(v2[0, 0]), np.sqrt(v2[1, 1]), np.sqrt(v_f[1, 1])])
    r = np.array(records)
    reliability_full = 2 * RELIABILITY_HALF / (1 + RELIABILITY_HALF)
    print(f"true beta {BETA}; expected OLS attenuation to {BETA * reliability_full:.4f}")
    print(f"OLS mean {r[:, 0].mean():.4f}; ORIV mean {r[:, 1].mean():.4f}")
    print(f"ORIV empirical SD {r[:, 1].std(ddof=1):.4f}; mean two-way SE {r[:, 3].mean():.4f}; "
          f"mean focal-only SE {r[:, 4].mean():.4f}")
    print(f"OLS empirical SD {r[:, 0].std(ddof=1):.4f}; mean two-way SE {r[:, 2].mean():.4f}")
    cover = np.mean(np.abs(r[:, 1] - BETA) <= 1.96 * r[:, 3])
    cover_f = np.mean(np.abs(r[:, 1] - BETA) <= 1.96 * r[:, 4])
    print(f"ORIV 95% coverage two-way {cover:.3f}; focal-only {cover_f:.3f}")
    assert abs(r[:, 1].mean() - BETA) < 3 * r[:, 1].std(ddof=1) / np.sqrt(reps)
    assert 0.85 < r[:, 3].mean() / r[:, 1].std(ddof=1) < 1.15
    assert 0.92 <= cover <= 0.98


if __name__ == "__main__":
    main()
