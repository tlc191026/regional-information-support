"""Describe subsequent objectives and validate prospective estimates and intervals."""

from common import *
import numpy as np
import pandas as pd
from scipy import sparse, special, stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings, gc

STATE = [
    "gold_at_window_diff",
    "xp_at_window_diff",
    "level_at_window_diff",
    "kills_pre_diff",
    "neutral_objectives_pre_diff",
    "turret_plates_pre_diff",
]
cats = ["region", "tier", "patch"]
log("Read outcome-available matches")
f = pd.read_parquet(
    ROOT / "manuscript_20260915/03_reproducibility/data/analysis_scored_0_10.parquet",
    columns=[
        "match_id",
        "region",
        "tier",
        "patch",
        "analysis_cutoff_ms",
        "first_post_neutral_ms",
        "first_monster_type",
        "first_status",
        "first_post_neutral_blue",
        "maintenance_diff_z",
        *STATE,
    ],
)
assert f.match_id.is_unique
f.region = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
status = (
    f.groupby(["region", "first_status"], observed=True)
    .size()
    .reset_index(name="n_matches")
)
save("objective_status.csv", status)
f = f[f.first_post_neutral_blue.notna()].reset_index(drop=True)
assert len(f) == 2210921
f["lag_seconds"] = (f.first_post_neutral_ms - f.analysis_cutoff_ms) / 1000
assert (f.lag_seconds > 0).all()
rows = []
for r, g in [("All", f), *list(f.groupby("region", observed=True))]:
    for typ, k in g.groupby("first_monster_type", observed=True):
        q = k.lag_seconds.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
        rows.append(
            dict(
                region=r,
                objective=typ,
                n_matches=len(k),
                percent=100 * len(k) / len(g),
                lag_p05=q[0],
                lag_p25=q[1],
                lag_median=q[2],
                lag_p75=q[3],
                lag_p95=q[4],
                within_60_seconds=int((k.lag_seconds <= 60).sum()),
            )
        )
    q = g.lag_seconds.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
    rows.append(
        dict(
            region=r,
            objective="All assigned objectives",
            n_matches=len(g),
            percent=100,
            lag_p05=q[0],
            lag_p25=q[1],
            lag_median=q[2],
            lag_p75=q[3],
            lag_p95=q[4],
            within_60_seconds=int((g.lag_seconds <= 60).sum()),
        )
    )
save("objective_type_and_lag.csv", pd.DataFrame(rows))


def design(df):
    e = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=True, dtype=float
    )
    cat = e.fit_transform(df[cats].astype(str))
    v = df[STATE].to_numpy(float)
    v = (v - v.mean(0)) / np.where(v.std(0) > 0, v.std(0), 1)
    return sparse.hstack(
        [np.ones((len(df), 1)), cat, v, df[["maintenance_diff_z"]].to_numpy(float)],
        format="csr",
    )


def fit(X, y, weights=None):
    m = LogisticRegression(
        C=1e6,
        solver="newton-cholesky",
        fit_intercept=False,
        max_iter=100,
        tol=1e-8,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        m.fit(X, y, sample_weight=weights)
    assert not any(issubclass(w.category, ConvergenceWarning) for w in warn)
    p = m.predict_proba(X)[:, 1]
    res = y - p
    wr = res if weights is None else res * weights
    err = float(
        np.max(abs(np.asarray(X.T @ wr).ravel() - m.coef_[0] / 1e6))
        / (len(y) if weights is None else weights.sum())
    )
    assert err < 2e-7
    return m, p, res, err


rows = []
diagnostics = []
for spec, sub in [
    ("original", f),
    ("original_first_objective_after_60_seconds", f[f.lag_seconds > 60]),
]:
    for region, g in [("All", sub), *list(sub.groupby("region", observed=True))]:
        log(f"Fit {spec}, {region}, n={len(g):,}")
        X = design(g)
        y = g.first_post_neutral_blue.to_numpy(float)
        model, p, res, err = fit(X, y)
        bread = np.linalg.pinv(
            (X.T @ X.multiply((p * (1 - p))[:, None])).toarray(), rcond=1e-12
        )
        psi = np.asarray(X @ bread[:, -1]).ravel() * res
        se = float(np.sqrt(psi @ psi))
        b = float(model.coef_[0, -1])
        rows.append(
            dict(
                specification=spec,
                region=region,
                n_matches=len(g),
                beta=b,
                robust_se=se,
                odds_ratio=np.exp(b),
                or_low=np.exp(b - 1.96 * se),
                or_high=np.exp(b + 1.96 * se),
                p_value=2 * stats.norm.sf(abs(b / se)),
            )
        )
        diagnostics.append(
            dict(
                specification=spec,
                region=region,
                score_error=err,
                iterations=int(model.n_iter_[0]),
            )
        )
save("delayed_objective_associations.csv", pd.DataFrame(rows))
dump("logistic_checks.json", diagnostics)

log("Paired one-step and full-refit bootstrap on 21,000 matches")
rng = np.random.default_rng(SEED)
ix = np.sort(
    np.concatenate(
        [
            rng.choice(g, 1000, replace=False)
            for g in f.groupby(["region", "tier"], observed=True).indices.values()
        ]
    )
)
sub = f.iloc[ix].reset_index(drop=True)
X = design(sub)
y = sub.first_post_neutral_blue.to_numpy(float)
model, p, res, err = fit(X, y)
b = float(model.coef_[0, -1])
bread = np.linalg.pinv(
    (X.T @ X.multiply((p * (1 - p))[:, None])).toarray(), rcond=1e-12
)
psi = np.asarray(X @ bread[:, -1]).ravel() * res
groups = list(sub.groupby(["region", "tier"], observed=True).indices.values())
one = []
full = []
errs = []
for i in range(300):
    wt = np.zeros(len(sub))
    for g in groups:
        wt[g] = rng.multinomial(len(g), np.full(len(g), 1 / len(g)))
    one.append(b + (wt - 1) @ psi)
    m, _, _, er = fit(X, y, wt)
    full.append(float(m.coef_[0, -1]))
    errs.append(er)
    if (i + 1) % 50 == 0:
        log(f"Full-refit logistic bootstrap {i+1}/300")
one = np.array(one)
full = np.array(full)
a = np.quantile(one, [0.025, 0.975])
c = np.quantile(full, [0.025, 0.975])
save(
    "bootstrap_logistic_comparison.csv",
    pd.DataFrame(
        [
            dict(
                estimate=b,
                one_step_sd=one.std(ddof=1),
                full_refit_sd=full.std(ddof=1),
                sd_ratio=one.std(ddof=1) / full.std(ddof=1),
                one_step_low=a[0],
                one_step_high=a[1],
                full_refit_low=c[0],
                full_refit_high=c[1],
                paired_rmse=np.sqrt(np.mean((one - full) ** 2)),
                paired_correlation=np.corrcoef(one, full)[0, 1],
                n_matches=len(sub),
                replicates=300,
                max_score_error=max(errs),
            )
        ]
    ),
)
np.savez_compressed(
    W / "data/bootstrap_logistic_draws.npz", one_step=one, full_refit=full
)
log("Timeline checks complete")
