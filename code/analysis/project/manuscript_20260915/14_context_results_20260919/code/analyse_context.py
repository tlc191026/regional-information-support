"""Reproduce context scores and estimate directly adjusted natural-unit signal rates."""

from pathlib import Path
import os, sys, json, hashlib, gc, time

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "4"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import sparse, stats, special

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
REG = ["NA", "EUW", "KR"]
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
PINGS = [
    "enemy_missing_pings_pm",
    "enemy_vision_pings_pm",
    "command_pings_pm",
    "need_vision_pings_pm",
    "assist_me_pings_pm",
    "on_my_way_pings_pm",
    "get_back_pings_pm",
    "push_pings_pm",
    "all_in_pings_pm",
]
M = "shared_information_maintenance"
S = "team_signalling"
for d in [
    "data",
    "source_data",
    "tables",
    "figures",
    "manuscript",
    "qa",
    "qa/docx_render",
    "logs",
]:
    (OUT / d).mkdir(parents=True, exist_ok=True)


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def save(df, rel):
    df.to_csv(OUT / rel, index=False, encoding="utf-8-sig")


def dump(rel, v):
    (OUT / rel).write_text(
        json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


paths = [
    ROOT / "exp_20260717/data/context_unresidualized_participant_scores.parquet",
    ROOT / "outputs/study0/data/study0_canonical_participant_match_sample.pkl",
    ROOT / "exp_20260717/data/bootstrap_weights_primary_region_tier_1000.npy",
    OUT.parent / "manuscript_20260805.docx",
]
dump("qa/input_manifest.json", {str(p): sha(p) for p in paths})
log("Read frozen scores and verify canonical identities")
f = pd.read_parquet(
    paths[0],
    columns=[
        "match_id",
        "puuid",
        "region",
        "tier",
        "team_position",
        "patch",
        "game_duration",
        M,
        S,
    ],
)
raw = pd.read_pickle(paths[1])
assert len(f) == len(raw) == 2520000
for key in ["match_id", "puuid"]:
    assert np.array_equal(f[key].astype(str), raw[key].astype(str))
assert not f[["puuid", "match_id"]].duplicated().any()
for m in PINGS:
    f[m] = raw[m].to_numpy(float) * 10
del raw
gc.collect()
f.region = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
valid = f.team_position.astype(str).isin(POS)
assert (~valid).sum() == 6
f = f.loc[valid].reset_index(drop=True)
n = len(f)
assert n == 2519994 and f.match_id.nunique() == 2240349
ids, pc = np.unique(f.puuid.astype(str), return_inverse=True)
ng = len(ids)
assert ng == 126000
players = (
    f[["puuid", "region", "tier"]]
    .drop_duplicates("puuid")
    .set_index("puuid")
    .loc[ids]
    .reset_index()
)
strata = players.groupby(["region", "tier"], observed=True).indices
assert len(strata) == 21 and all(len(i) == 6000 for i in strata.values())
G = sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ng, n))
weights = np.load(paths[2], mmap_mode="r")
assert weights.shape == (1000, ng)
for ix in strata.values():
    assert np.all(np.asarray(weights[:, ix]).sum(1) == 6000)
patches = sorted(
    f.patch.astype(str).unique(), key=lambda s: tuple(map(int, s.split(".")))
)
codes = {
    k: pd.Categorical(f[k].astype(str), categories=v).codes
    for k, v in {
        "region": REG,
        "tier": TIERS,
        "team_position": POS,
        "patch": patches,
    }.items()
}
assert all((a >= 0).all() for a in codes.values())
duration = f.game_duration.to_numpy(float)
duration = (duration - duration.mean()) / duration.std(ddof=0)
audit = {
    "n_records": n,
    "n_players": ng,
    "n_unique_matches": int(f.match_id.nunique()),
    "bootstrap_reps": 1000,
    "seed": 191026,
}


def design(context):
    cat = "tier" if context == "tier" else "team_position"
    lev = TIERS if context == "tier" else POS
    row = [np.arange(n)]
    col = [np.zeros(n, dtype=int)]
    data = [np.ones(n)]
    names = ["Intercept"]
    where = {}
    for key, categories in [
        ("region", REG),
        ("tier", TIERS),
        ("team_position", POS),
        ("patch", patches),
    ]:
        c = codes[key]
        use = c > 0
        start = len(names)
        where[key] = (start, categories)
        row.append(np.flatnonzero(use))
        col.append(start + c[use] - 1)
        data.append(np.ones(use.sum()))
        names.extend([key + "_" + v for v in categories[1:]])
    row.append(np.arange(n))
    col.append(np.full(n, len(names)))
    data.append(duration)
    names.append("duration_z")
    for r in [1, 2]:
        for t in range(1, len(lev)):
            ix = np.flatnonzero((codes["region"] == r) & (codes[cat] == t))
            row.append(ix)
            col.append(np.full(len(ix), len(names)))
            data.append(np.ones(len(ix)))
            names.append("interaction_" + REG[r] + "_" + lev[t])
    X = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
        shape=(n, len(names)),
    )
    L = []
    meta = []
    for r, region in enumerate(REG):
        for t, level in enumerate(lev):
            v = np.zeros(len(names))
            v[0] = 1
            for key, (start, categories) in where.items():
                if key == "region":
                    if r:
                        v[start + r - 1] = 1
                elif key == cat:
                    if t:
                        v[start + t - 1] = 1
                elif key == "patch":
                    v[start : start + len(categories) - 1] = (
                        np.bincount(codes[key], minlength=len(categories))[1:] / n
                    )
                else:
                    v[start : start + len(categories) - 1] = 1 / len(categories)
            if r and t:
                v[names.index("interaction_" + region + "_" + level)] = 1
            L.append(v)
            meta.append((region, level))
    return X, np.array(L), names, meta


def sci(lp):
    exponent = int(np.floor(lp))
    return f"{10**(lp-exponent):.10f}e{exponent:+d}"


def inference(df):
    df = df.copy()
    df["log10_p"] = (np.log(2) + stats.norm.logsf(abs(df.estimate / df.se))) / np.log(
        10
    )
    df["log10_p_holm"] = np.nan
    for _, idx in df.groupby("family", sort=False).groups.items():
        idx = np.array(idx)
        lp = df.loc[idx, "log10_p"].to_numpy()
        o = np.argsort(lp)
        adj = np.minimum(
            0, np.maximum.accumulate(lp[o] + np.log10(np.arange(len(idx), 0, -1)))
        )
        out = np.empty(len(idx))
        out[o] = adj
        df.loc[idx, "log10_p_holm"] = out
    df["p_value"] = [sci(v) for v in df.log10_p]
    df["p_holm"] = [sci(v) for v in df.log10_p_holm]
    return df


for context, outcomes in [("tier", [M, S, *PINGS]), ("position", [M, S])]:
    log("Fit " + context)
    X, L, names, meta = design(context)
    Y = f[outcomes].to_numpy(float)
    gram = (X.T @ X).toarray()
    assert np.linalg.matrix_rank(gram) == len(names)
    bread = np.linalg.inv(gram)
    beta = bread @ (X.T @ Y)
    E = Y - X @ beta
    est0 = L @ beta
    k = len(meta)
    q = len(outcomes)
    J = np.empty((ng, k, q))
    R = bread @ L.T
    for j, m in enumerate(outcomes):
        J[:, :, j] = (G @ X.multiply(E[:, j, None])).toarray() @ R
    # Include overall in joint inference; its dependence is retained exactly.
    J = np.concatenate([((J[:, :, 0] + J[:, :, 1]) / 2)[:, :, None], J], axis=2)
    est = np.column_stack([(est0[:, 0] + est0[:, 1]) / 2, est0])
    measures = ["overall", *outcomes]
    q = len(measures)
    J = J.reshape(ng, k * q)
    for ix in strata.values():
        J[ix] -= J[ix].mean(0)
    cov = J.T @ J * (6000 / 5999)
    draw = np.empty((1000, k * q))
    for b in range(0, 1000, 25):
        draw[b : b + 25] = (
            est.ravel() + (np.asarray(weights[b : b + 25], dtype=float) - 1) @ J
        )
    ci = np.quantile(draw, [0.025, 0.975], axis=0)
    rows = []
    for i, (r, lev) in enumerate(meta):
        use = f.region.eq(r) & f["tier" if context == "tier" else "team_position"].eq(
            lev
        )
        nr = int(use.sum())
        np_ = int(f.loc[use, "puuid"].nunique())
        for j, m in enumerate(measures):
            z = i * q + j
            rows.append(
                dict(
                    context=context,
                    region=r,
                    level=lev,
                    measure=m,
                    estimate=est.ravel()[z],
                    ci_low=ci[0, z],
                    ci_high=ci[1, z],
                    se=np.sqrt(cov[z, z]),
                    n_records=nr,
                    n_players=np_,
                    unit="adjusted score" if j < 3 else "events per player per 10 min",
                )
            )
    means = pd.DataFrame(rows)
    save(means, f"source_data/{context}_adjusted_means.csv")
    keys = [r + "|" + lev + "|" + m for r, lev in meta for m in measures]
    np.savez_compressed(
        OUT / "data" / f"{context}_joint_inference.npz",
        estimate=est.ravel(),
        cov=cov,
        draws=draw,
        keys=np.array(keys),
    )
    save(
        pd.DataFrame(beta, index=names, columns=outcomes).reset_index(names="term"),
        f"source_data/{context}_coefficients.csv",
    )
    old = pd.read_csv(
        ROOT
        / "exp_final_construct_v3/tables"
        / f"context_{context}_process_adjusted_means.csv",
        keep_default_na=False,
    )
    check = means.merge(
        old,
        left_on=["region", "level", "measure"],
        right_on=["region", "moderator_level", "measure"],
        suffixes=("_new", "_old"),
    )
    err = float((check.estimate_new - check.estimate_old).abs().max())
    assert err < 1e-4, err
    audit[context + "_frozen_max_abs_error"] = err
    tests = []

    def contrast(m, label, typ, coef):
        v = np.zeros(k * q)
        for (r, l), value in coef.items():
            v[keys.index(r + "|" + l + "|" + m)] = value
        d = draw @ v
        point = float(est.ravel() @ v)
        low, high = np.quantile(d, [0.025, 0.975])
        se = float(np.sqrt(max(0, v @ cov @ v)))
        tests.append(
            dict(
                measure=m,
                contrast=label,
                type=typ,
                estimate=point,
                ci_low=low,
                ci_high=high,
                se=se,
                family=typ
                + ("_processes" if m in ["overall", M, S] else "_nine_signals"),
            )
        )

    levs = TIERS if context == "tier" else POS
    for m in measures:
        if context == "tier":
            for r in REG:
                contrast(
                    m,
                    r + " Diamond-Iron",
                    "endpoint",
                    {(r, "DIAMOND"): 1, (r, "IRON"): -1},
                )
                for a, b in zip(TIERS[:-1], TIERS[1:]):
                    contrast(
                        m, r + " " + b + "-" + a, "adjacent", {(r, b): 1, (r, a): -1}
                    )
                for j in range(1, 6):
                    contrast(
                        m,
                        r + " " + TIERS[j],
                        "local_change",
                        {(r, TIERS[j + 1]): 1, (r, TIERS[j]): -2, (r, TIERS[j - 1]): 1},
                    )
            for a, b in [("KR", "NA"), ("KR", "EUW"), ("EUW", "NA")]:
                contrast(
                    m,
                    a + "-" + b + " Diamond-Iron",
                    "between_region_endpoint",
                    {
                        (a, "DIAMOND"): 1,
                        (a, "IRON"): -1,
                        (b, "DIAMOND"): -1,
                        (b, "IRON"): 1,
                    },
                )
        else:
            for lev in POS:
                for a, b in [("KR", "NA"), ("KR", "EUW"), ("EUW", "NA")]:
                    contrast(
                        m,
                        a + "-" + b + " " + lev,
                        "position_regional",
                        {(a, lev): 1, (b, lev): -1},
                    )
    save(inference(pd.DataFrame(tests)), f"tables/{context}_contrasts.csv")
    # The process interaction asks whether regional context changes differ between M and S.
    C = []
    for r in ["EUW", "KR"]:
        for lev in levs[1:]:
            v = np.zeros(k * q)
            for m, sgn in [(M, 1), (S, -1)]:
                for rr, ll, w in [
                    (r, lev, 1),
                    (r, levs[0], -1),
                    ("NA", lev, -1),
                    ("NA", levs[0], 1),
                ]:
                    v[keys.index(rr + "|" + ll + "|" + m)] += w * sgn
            C.append(v)
    C = np.array(C)
    d = C @ est.ravel()
    vc = C @ cov @ C.T
    chi = float(d @ np.linalg.pinv(vc) @ d)
    # Even-df chi-square survival in log space, avoiding very-small-P underflow.
    powers = np.arange(len(C) // 2)
    logp = -chi / 2 + special.logsumexp(
        powers * np.log(chi / 2) - special.gammaln(powers + 1)
    )
    audit[context + "_region_process_interaction"] = {
        "chi2": chi,
        "df": len(C),
        "p_value": sci(logp / np.log(10)),
    }
    audit[context + "_minimum_raw_adjusted_rate"] = (
        float(means[~means.measure.isin(["overall", M, S])].estimate.min())
        if context == "tier"
        else None
    )
    log(context + " complete; frozen estimate error " + str(err))
    del X, L, Y, E, J, cov, draw, est, est0
    gc.collect()
dump("qa/analysis_audit.json", audit)
log(json.dumps(audit))
