"""Two explicitly recorded supplementary analyses using the fixed sample."""

from pathlib import Path
import os, json, gc, sys, time

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import sparse, stats

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
REG = ["NA", "EUW", "KR"]
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
PINGS = [
    s + "_pings_pm"
    for s in [
        "enemy_missing",
        "enemy_vision",
        "command",
        "need_vision",
        "assist_me",
        "on_my_way",
        "push",
        "all_in",
        "get_back",
    ]
]


def save(d, name):
    d.to_csv(OUT / "tables" / name, index=False, encoding="utf-8-sig")


def dump(v, name):
    (OUT / "qa" / name).write_text(
        json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def probability_strings(z):
    lp = (np.log(2) + stats.norm.logsf(abs(z))) / np.log(10)
    return lp, f"{10**(lp-np.floor(lp)):.10f}e{int(np.floor(lp)):+d}"


for directory in ["data", "tables", "qa", "logs"]:
    (OUT / directory).mkdir(parents=True, exist_ok=True)


def equal_signal():
    log("Load frozen scores for nine-indicator equal weighting")
    f = pd.read_parquet(
        ROOT / "exp_20260717/data/context_unresidualized_participant_scores.parquet",
        columns=[
            "puuid",
            "match_id",
            "region",
            "tier",
            "team_position",
            "patch",
            "game_duration",
            "shared_information_maintenance",
            "team_signalling",
        ],
    )
    t = pd.read_parquet(
        ROOT / "exp_20260717/data/participant_metrics_transformed_z.parquet",
        columns=PINGS,
    )
    assert len(f) == len(t) == 2520000
    raw = t.to_numpy(float).mean(1)
    mu = raw.mean()
    sd = raw.std(ddof=0)
    f["signalling_equal"] = (raw - mu) / sd
    del t, raw
    gc.collect()
    f.region = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
    valid = f.team_position.astype(str).isin(POS)
    assert (~valid).sum() == 6
    f = f.loc[valid].reset_index(drop=True)
    n = len(f)
    ids, pc = np.unique(f.puuid.astype(str), return_inverse=True)
    ng = len(ids)
    assert ng == 126000
    players = (
        f[["puuid", "region", "tier"]]
        .drop_duplicates("puuid")
        .set_index("puuid")
        .loc[ids]
    )
    strata = players.groupby(["region", "tier"], sort=True, observed=True).indices
    G = sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ng, n))
    patches = sorted(
        f.patch.astype(str).unique(), key=lambda s: tuple(map(int, s.split(".")))
    )
    levels = {"region": REG, "tier": TIERS, "team_position": POS, "patch": patches}
    names = ["Intercept"]
    row = [np.arange(n)]
    col = [np.zeros(n, dtype=int)]
    data = [np.ones(n)]
    L = np.zeros((3, 1 + 2 + 6 + 4 + len(patches) - 1 + 1))
    L[:, 0] = 1
    for variable, cats in levels.items():
        codes = pd.Categorical(f[variable].astype(str), categories=cats).codes
        assert (codes >= 0).all()
        use = codes > 0
        start = len(names)
        row.append(np.flatnonzero(use))
        col.append(start + codes[use] - 1)
        data.append(np.ones(use.sum()))
        names.extend([f"{variable}[{c}]" for c in cats[1:]])
        if variable == "region":
            L[1, start] = 1
            L[2, start + 1] = 1
        elif variable in ["tier", "team_position"]:
            L[:, start : start + len(cats) - 1] = 1 / len(cats)
        else:
            L[:, start : start + len(cats) - 1] = (
                np.bincount(codes, minlength=len(cats))[1:] / n
            )
    d = f.game_duration.to_numpy(float)
    row.append(np.arange(n))
    col.append(np.full(n, len(names)))
    data.append((d - d.mean()) / d.std(ddof=0))
    names.append("duration_z")
    X = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
        shape=(n, len(names)),
    )
    del row, col, data, d
    Y = f[
        ["shared_information_maintenance", "team_signalling", "signalling_equal"]
    ].to_numpy(float)
    gram = (X.T @ X).toarray()
    bread = np.linalg.pinv(gram, rcond=1e-11, hermitian=True)
    beta = bread @ (X.T @ Y)
    E = Y - X @ beta
    means = L @ beta
    assert np.linalg.matrix_rank(gram) == len(names)
    old = pd.read_csv(
        OUT / "source_data/regional/regional_adjusted_means.csv", keep_default_na=False
    )
    err = max(
        abs(
            means[i, j]
            - float(old[(old.region == r) & (old.measure == m)].estimate.iloc[0])
        )
        for i, r in enumerate(REG)
        for j, m in enumerate(["maintenance", "signalling"])
    )
    assert err < 1e-8, err
    influence = np.empty((ng, 3, 3))
    projected = X @ (bread @ L.T)
    for j in range(3):
        influence[:, :, j] = G @ (E[:, j, None] * projected)
    flat = influence.reshape(ng, 9)
    for ix in strata.values():
        assert len(ix) == 6000
        flat[ix] -= flat[ix].mean(0)
    cov = flat.T @ flat * (6000 / 5999)
    weights = np.load(
        ROOT / "exp_20260717/data/bootstrap_weights_primary_region_tier_1000.npy",
        mmap_mode="r",
    )
    assert weights.shape == (1000, ng)
    draws = np.empty((1000, 9))
    for start in range(0, 1000, 50):
        draws[start : start + 50] = (
            means.ravel() + (np.asarray(weights[start : start + 50], float) - 1) @ flat
        )
    draws = draws.reshape(1000, 3, 3)
    ci = np.quantile(draws, [0.025, 0.975], axis=0)
    mr = []
    for i, r in enumerate(REG):
        for j, m in enumerate(
            ["maintenance", "signalling_primary", "signalling_equal"]
        ):
            mr.append(
                dict(
                    region=r,
                    measure=m,
                    estimate=means[i, j],
                    ci_low=ci[0, i, j],
                    ci_high=ci[1, i, j],
                    n_players=42000,
                )
            )
    save(pd.DataFrame(mr), "equal_signal_regional_means.csv")
    rows = []
    for a, b in [("KR", "NA"), ("KR", "EUW"), ("EUW", "NA")]:
        ia, ib = REG.index(a), REG.index(b)
        v = np.zeros((3, 3))
        v[ia, 0] = 1
        v[ib, 0] = -1
        v[ia, 2] = -1
        v[ib, 2] = 1
        delta = means[ia] - means[ib]
        ds = draws[:, ia] - draws[:, ib]
        change = ds[:, 0] - ds[:, 2]
        se = float(np.sqrt(v.ravel() @ cov @ v.ravel()))
        estimate = delta[0] - delta[2]
        lp, p = probability_strings(estimate / se)
        lo, hi = np.quantile(change, [0.025, 0.975])
        ml, mh = np.quantile(ds[:, 0], [0.025, 0.975])
        sl, sh = np.quantile(ds[:, 2], [0.025, 0.975])
        rows.append(
            dict(
                contrast=a + "-" + b,
                maintenance_difference=delta[0],
                maintenance_low=ml,
                maintenance_high=mh,
                signal_difference=delta[2],
                signal_low=sl,
                signal_high=sh,
                estimate=estimate,
                ci_low=lo,
                ci_high=hi,
                se=se,
                z=estimate / se,
                p_value=p,
                log10_p=lp,
                primary_process_difference=delta[0] - delta[1],
                n_players=84000,
            )
        )
    result = pd.DataFrame(rows)
    lp = result.log10_p.to_numpy()
    order = np.argsort(lp)
    adjust = np.empty(3)
    adjust[order] = np.minimum(
        0, np.maximum.accumulate(lp[order] + np.log10([3, 2, 1]))
    )
    result["log10_p_holm"] = adjust
    result["p_holm"] = [
        f"{10**(x-np.floor(x)):.10f}e{int(np.floor(x)):+d}" for x in adjust
    ]
    result["family"] = (
        "3 supplementary between-process contrasts under nine-signal equal weighting"
    )
    save(result, "equal_signal_process_comparisons.csv")
    np.savez_compressed(
        OUT / "data/equal_signal_joint_fit.npz",
        beta=beta,
        names=np.array(names),
        reference=L,
        means=means,
        cov=cov,
        bootstrap=draws,
    )
    dump(
        {
            "n_records": n,
            "n_players": ng,
            "n_matches": int(f.match_id.nunique()),
            "equal_signal_raw_mean": mu,
            "equal_signal_raw_sd": sd,
            "scoring_reference_n": 2520000,
            "max_error_reproducing_primary_means": err,
            "seed": 191026,
            "bootstrap_replicates": 1000,
            "scale": "mean of nine frozen standardized indicators, restandardized over full original record sample; scoring parameters fixed in inference",
        },
        "equal_signal_audit.json",
    )
    log(result.to_string(index=False))


def tier_reference():
    log("Build one patch reference supported in all 21 region by tier cells")
    f = pd.read_parquet(
        OUT.parent / "03_reproducibility/data/analysis_scored_0_10.parquet",
        columns=["region", "tier", "patch", "blue_maintenance", "red_maintenance"],
    )
    f.region = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
    f["score"] = (f.blue_maintenance + f.red_maintenance) / 2
    counts = (
        f.groupby(["patch", "region", "tier"], observed=True)
        .size()
        .unstack(["region", "tier"], fill_value=0)
    )
    assert counts.shape[1] == 21
    supported = counts.index[counts.min(axis=1) >= 30]
    counts = counts.loc[supported]
    weights = counts.sum(1) / counts.to_numpy().sum()
    mean = []
    composition = []
    for region in REG:
        for tier in TIERS:
            sub = f[(f.region == region) & (f.tier == tier) & f.patch.isin(supported)]
            g = (
                sub.groupby("patch", observed=True)
                .score.agg(["mean", "var", "size"])
                .reindex(supported)
            )
            value = float(weights @ g["mean"])
            variance = float((weights**2 * g["var"] / g["size"]).sum())
            se = np.sqrt(variance)
            mean.append(
                dict(
                    region=region,
                    tier=tier,
                    estimate=value,
                    se=se,
                    ci_low=value - 1.96 * se,
                    ci_high=value + 1.96 * se,
                    n_matches=len(sub),
                    n_teams=2 * len(sub),
                    reference="one pooled patch distribution shared by all 21 cells",
                    method="within-cell match mean variance; fixed reference weights; pointwise Wald CI",
                )
            )
            for patch, r in g.iterrows():
                composition.append(
                    dict(
                        region=region,
                        tier=tier,
                        patch=patch,
                        n_matches=int(r["size"]),
                        reference_weight=float(weights[patch]),
                    )
                )
    m = pd.DataFrame(mean)
    save(m, "early_tier_common_patch_means.csv")
    save(pd.DataFrame(composition), "early_tier_common_patch_composition.csv")
    effects = []
    for a, b in [("KR", "NA"), ("KR", "EUW"), ("EUW", "NA")]:
        for tier in TIERS:
            aa = m[(m.region == a) & (m.tier == tier)].iloc[0]
            bb = m[(m.region == b) & (m.tier == tier)].iloc[0]
            d = aa.estimate - bb.estimate
            se = np.sqrt(aa.se**2 + bb.se**2)
            effects.append(
                dict(
                    contrast=a + "-" + b,
                    tier=tier,
                    estimate=d,
                    ci_low=d - 1.96 * se,
                    ci_high=d + 1.96 * se,
                )
            )
    save(pd.DataFrame(effects), "early_tier_common_patch_contrasts.csv")
    dump(
        {
            "n_full_matches": len(f),
            "n_common_matches": int(m.n_matches.sum()),
            "retained_fraction": float(m.n_matches.sum() / len(f)),
            "minimum_cell_n": int(counts.min().min()),
            "patches": list(supported),
            "n_patches": len(supported),
            "reference_weights_sum": float(weights.sum()),
            "global_score_reused": True,
        },
        "early_tier_common_reference_audit.json",
    )
    log(m.to_string(index=False))


if __name__ == "__main__":
    equal_signal()
    gc.collect()
    tier_reference()
