"""Re-estimate nine signals on the current section's reference and sample."""

from pathlib import Path
import os, json, hashlib, shutil, gc

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy import sparse, stats

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
BASE = OUT.parent / "10_overall_and_process_differences_20260918"
for d in [
    "data",
    "source_data",
    "tables",
    "figures",
    "manuscript",
    "qa",
    "qa/docx_render",
]:
    (OUT / d).mkdir(parents=True, exist_ok=True)
REGIONS = ["NA", "EUW", "KR"]
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
METRICS = [
    "enemy_missing_pings_pm",
    "enemy_vision_pings_pm",
    "command_pings_pm",
    "need_vision_pings_pm",
    "assist_me_pings_pm",
    "on_my_way_pings_pm",
    "push_pings_pm",
    "all_in_pings_pm",
    "get_back_pings_pm",
]
PAIRS = [("KR", "NA"), ("KR", "EUW"), ("EUW", "NA")]
manifest = {}
audit = {}


def fingerprint(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    manifest[str(p.relative_to(ROOT))] = {
        "sha256": h.hexdigest(),
        "bytes": p.stat().st_size,
    }


def save(d, rel):
    d.to_csv(OUT / rel, index=False, encoding="utf-8-sig")


def sci(logp):
    if logp == 0:
        return "1"
    k = int(np.floor(logp))
    return f"{10**(logp-k):.10f}e{k:+d}"


def pvals(d):
    d = d.copy()
    z = np.abs(d.estimate / d.se)
    logs = (np.log(2) + stats.norm.logsf(z)) / np.log(10)
    order = np.argsort(logs)
    a = np.minimum(
        0,
        np.maximum.accumulate(logs[order] + np.log10(len(logs) - np.arange(len(logs)))),
    )
    q = np.empty(len(d))
    q[order] = a
    d["log10_p"] = logs
    d["log10_p_holm"] = q
    d["p_value"] = [sci(x) for x in logs]
    d["p_holm"] = [sci(x) for x in q]
    d["correction_family"] = "27 signal by regional comparisons"
    return d


for rel in [
    "source_data/regional_adjusted_means.csv",
    "source_data/player_distribution_summaries.csv",
    "source_data/all_player_densities.csv",
    "tables/primary_comparisons.csv",
    "tables/indicator_equal_weight_sensitivity.csv",
    "tables/two_way_cluster_sensitivity.csv",
    "source_data/reference_vectors.csv",
    "source_data/adjustment_coefficients.csv",
]:
    fingerprint(BASE / rel)
    shutil.copyfile(BASE / rel, OUT / rel)
paths = [
    ROOT / "exp_20260717/data/context_unresidualized_participant_scores.parquet",
    ROOT / "exp_20260717/data/participant_metrics_transformed_z.parquet",
    ROOT / "outputs/study0/data/study0_canonical_participant_match_sample.pkl",
    ROOT / "exp_20260717/data/bootstrap_weights_primary_region_tier_1000.npy",
    ROOT / "manuscript_20260915/manuscript_20260805.docx",
    ROOT / "exp_final_construct_v3/tables/metric_localization_18_items.csv",
]
for p in paths:
    fingerprint(p)
print("Loading fixed telemetry and checking row identities", flush=True)
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
    ],
)
trans = pd.read_parquet(paths[1], columns=METRICS)
raw = pd.read_pickle(paths[2])
assert len(f) == len(trans) == len(raw) == 2520000
for key in ["match_id", "puuid"]:
    assert np.array_equal(f[key].astype(str), raw[key].astype(str)), key
assert not raw[["puuid", "match_id"]].duplicated().any()
raw_values = raw[METRICS].to_numpy(float)
del raw
gc.collect()
assert np.isfinite(raw_values).all() and (raw_values >= 0).all()
reconstruction = {}
for j, m in enumerate(METRICS):
    v = np.log1p(raw_values[:, j])
    lo, hi = np.quantile(v, [0.01, 0.99])
    v = np.clip(v, lo, hi)
    v = (v - v.mean()) / v.std(ddof=0)
    err = float(np.max(np.abs(v - trans[m].to_numpy(float))))
    reconstruction[m] = err
    assert err < 2e-5, (m, err)
f.region = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
valid = f.team_position.astype(str).isin(POSITIONS).to_numpy()
assert (~valid).sum() == 6
save(
    f.loc[~valid, ["puuid", "match_id", "region", "team_position"]],
    "data/excluded_records.csv",
)
Y = trans.to_numpy(float)[valid]
del trans
raw_values = raw_values[valid] * 10
f = f.loc[valid].reset_index(drop=True)
n = len(f)
assert n == 2519994 and f.match_id.nunique() == 2240349
ids, pc = np.unique(f.puuid.astype(str), return_inverse=True)
pc = pc.astype(np.int32)
ng = len(ids)
counts = np.bincount(pc)
assert ng == 126000 and (counts == 19).sum() == 6 and (counts == 20).sum() == 125994
players = (
    f[["puuid", "region", "tier"]]
    .drop_duplicates("puuid")
    .set_index("puuid")
    .loc[ids]
    .reset_index()
)
assert players.region.value_counts().to_dict() == {
    "NA": 42000,
    "EUW": 42000,
    "KR": 42000,
}
strata = players.groupby(["region", "tier"], sort=True, observed=True).indices
assert len(strata) == 21 and all(len(g) == 6000 for g in strata.values())
G = sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ng, n))
patches = sorted(
    f.patch.astype(str).unique(), key=lambda x: tuple(map(int, x.split(".")))
)
levels = {
    "region": REGIONS,
    "tier": TIERS,
    "team_position": POSITIONS,
    "patch": patches,
}
names = ["Intercept"]
rows = [np.arange(n)]
cols = [np.zeros(n, dtype=np.int32)]
data = [np.ones(n)]
L = np.zeros((3, 1 + 2 + 6 + 4 + len(patches) - 1 + 1))
L[:, 0] = 1
for variable, cats in levels.items():
    codes = pd.Categorical(f[variable].astype(str), categories=cats).codes
    assert (codes >= 0).all()
    use = codes > 0
    start = len(names)
    rows.append(np.flatnonzero(use))
    cols.append(start + codes[use] - 1)
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
duration = f.game_duration.to_numpy(float)
rows.append(np.arange(n))
cols.append(np.full(n, len(names)))
data.append((duration - duration.mean()) / duration.std(ddof=0))
names.append("duration_z")
X = sparse.csr_matrix(
    (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
    shape=(n, len(names)),
)
del rows, cols, data, duration
previous_L = pd.read_csv(
    BASE / "source_data/reference_vectors.csv", keep_default_na=False
)
assert list(previous_L.columns[1:]) == names and np.allclose(
    L, previous_L[names].to_numpy(), atol=1e-12
)
print("Fitting nine responses with the identical 63-column adjustment", flush=True)
gram = (X.T @ X).toarray()
bread = np.linalg.pinv(gram, rcond=1e-11, hermitian=True)
assert np.linalg.matrix_rank(gram) == len(names)
beta = bread @ (X.T @ Y)
E = Y - X @ beta
est = L @ beta
save(
    pd.DataFrame(beta, index=names, columns=METRICS).reset_index(names="term"),
    "source_data/signal_adjustment_coefficients.csv",
)
pr = pd.Categorical(players.region, categories=REGIONS).codes
px = (G @ X).toarray() / counts[:, None]
vals = (G @ E) / counts[:, None] + est[pr]
players["n_records"] = counts
for j, m in enumerate(METRICS):
    players[m] = vals[:, j]
players.to_parquet(OUT / "data/adjusted_player_signals.parquet", index=False)
mean_inf = np.empty((ng, 3, 9))
var_inf = np.empty_like(mean_inf)
variances = np.empty((3, 9))
for j, m in enumerate(METRICS):
    influence = (G @ X.multiply(E[:, j, None])).toarray() @ bread
    for r in range(3):
        use = pr == r
        a = vals[use, j]
        dev = a - a.mean()
        H = px[use] - L[r]
        v = dev @ dev / (len(a) - 1)
        variances[r, j] = v
        mean_inf[:, r, j] = influence @ L[r]
        grad = -2 * (dev @ H) / (len(a) - 1)
        var_inf[:, r, j] = influence @ grad
        var_inf[use, r, j] += (dev**2 - np.mean(dev**2)) / (len(a) - 1)
    print("Cluster influence", m, flush=True)
del X, Y, E, G, px, influence
gc.collect()
joint = np.column_stack([mean_inf.reshape(ng, 27), var_inf.reshape(ng, 27)])
for ix in strata.values():
    joint[ix] -= joint[ix].mean(0)
cov = joint[:, :27].T @ joint[:, :27] * (6000 / 5999)
weights = np.load(paths[3], mmap_mode="r")
assert weights.shape == (1000, ng)
for ix in strata.values():
    assert np.all(np.asarray(weights[:, ix]).sum(1) == 6000)
point = np.r_[est.ravel(), variances.ravel()]
boot = np.empty((1000, 54))
for i in range(0, 1000, 25):
    boot[i : i + 25] = (
        point + (np.asarray(weights[i : i + 25], dtype=float) - 1) @ joint
    )
    if i % 250 == 0:
        print("Joint bootstrap", i + 25, flush=True)
assert np.isfinite(boot).all() and (boot[:, 27:] > 0).all()
save(
    pd.DataFrame(
        boot,
        columns=[
            f"{k}_{r}_{m}"
            for k in ["mean", "variance"]
            for r in REGIONS
            for m in METRICS
        ],
    ),
    "data/signal_bootstrap_replicates.csv",
)
bm = boot[:, :27].reshape(1000, 3, 9)
bv = boot[:, 27:].reshape(1000, 3, 9)
ci = np.quantile(bm, [0.025, 0.975], axis=0)
means = []
contrasts = []
for r, region in enumerate(REGIONS):
    for j, m in enumerate(METRICS):
        means.append(
            dict(
                region=region,
                metric=m,
                estimate=est[r, j],
                ci_low=ci[0, r, j],
                ci_high=ci[1, r, j],
                n_players=42000,
                n_records=int(counts[pr == r].sum()),
                player_mean=vals[pr == r, j].mean(),
                player_sd=np.sqrt(variances[r, j]),
            )
        )
for a, b in PAIRS:
    ia, ib = REGIONS.index(a), REGIONS.index(b)
    for j, m in enumerate(METRICS):
        v = np.zeros(27)
        v[ia * 9 + j] = 1
        v[ib * 9 + j] = -1
        delta = float(v @ est.ravel())
        se = float(np.sqrt(v @ cov @ v))
        draws = bm[:, ia, j] - bm[:, ib, j]
        low, high = np.quantile(draws, [0.025, 0.975])
        denom = np.sqrt((variances[ia, j] + variances[ib, j]) / 2)
        dl, dh = np.quantile(
            draws / np.sqrt((bv[:, ia, j] + bv[:, ib, j]) / 2), [0.025, 0.975]
        )
        contrasts.append(
            dict(
                metric=m,
                contrast=a + "-" + b,
                estimate=delta,
                se=se,
                ci_low=low,
                ci_high=high,
                standardized_difference=delta / denom,
                d_ci_low=dl,
                d_ci_high=dh,
                pooled_player_sd=denom,
                n_a=42000,
                n_b=42000,
            )
        )
means = pd.DataFrame(means)
contrasts = pvals(pd.DataFrame(contrasts))
save(means, "source_data/Fig2d_signal_adjusted_means.csv")
save(contrasts, "tables/signal_regional_comparisons.csv")
# Equal region shifts across all nine globally standardized signal measures.
C = []
for r in [1, 2]:
    for j in range(8):
        c = np.zeros(27)
        c[r * 9 + j] = 1
        c[j] = -1
        c[r * 9 + 8] = -1
        c[8] = 1
        C.append(c)
C = np.asarray(C)
v = C @ est.ravel()
V = C @ cov @ C.T
df = int(np.linalg.matrix_rank(V))
wald = float(v @ np.linalg.pinv(V) @ v)
p = float(stats.chi2.sf(wald, df))
pstr = f"{p:.12g}" if p else "<1e-300"
omnibus = dict(
    wald_chi2=wald,
    df=df,
    p_value=pstr,
    n_players=ng,
    hypothesis="Both region contrasts are constant across nine transformed standardized signal metrics",
    inference="Joint stratified player-cluster covariance; secondary follow-up analysis",
)
(OUT / "tables/signal_profile_omnibus.json").write_text(
    json.dumps(omnibus, indent=2), encoding="utf-8"
)

print("Computing record occurrence and positive-record intensity", flush=True)
G = sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ng, n))
total = G @ raw_values
active = G @ (raw_values > 0).astype(float)
descriptive = []
for r, region in enumerate(REGIONS):
    use = pr == r
    nc = counts[use].astype(float)
    for j, m in enumerate(METRICS):
        nn = active[use, j]
        num = total[use, j]
        p = nn.sum() / nc.sum()
        freq = num.sum() / nn.sum()
        marginal = num.sum() / nc.sum()
        ip = (nn - p * nc) / nc.mean()
        ic = (num - freq * nn) / nn.mean()
        im = (num - marginal * nc) / nc.mean()
        sp = ip.std(ddof=1) / np.sqrt(len(nc))
        sc = ic.std(ddof=1) / np.sqrt(len(nc))
        sm = im.std(ddof=1) / np.sqrt(len(nc))
        assert abs(p * freq - marginal) < 1e-12
        descriptive.append(
            dict(
                region=region,
                metric=m,
                prevalence=p,
                prevalence_ci_low=max(0, p - 1.96 * sp),
                prevalence_ci_high=min(1, p + 1.96 * sp),
                conditional_mean_per_10_min=freq,
                conditional_ci_low=max(0, freq - 1.96 * sc),
                conditional_ci_high=freq + 1.96 * sc,
                marginal_mean_per_10_min=marginal,
                marginal_ci_low=max(0, marginal - 1.96 * sm),
                marginal_ci_high=marginal + 1.96 * sm,
                n_players=len(nc),
                n_records=int(nc.sum()),
                n_active_records=int(nn.sum()),
                interval_method="Player-cluster ratio delta-method normal 95% CI; unadjusted",
            )
        )
desc = pd.DataFrame(descriptive)
save(desc, "source_data/Fig2ef_signal_occurrence_frequency.csv")
old = pd.read_csv(paths[-1], keep_default_na=False)
old = old[old.metric.isin(METRICS)]
recon = contrasts.merge(
    old[["metric", "contrast", "estimate"]],
    on=["metric", "contrast"],
    suffixes=("", "_old_d"),
)
recon["d_change"] = recon.standardized_difference - recon.estimate_old_d
save(recon, "qa/old_new_signal_reconciliation.csv")
coverage = []
overall_players = pd.read_parquet(
    BASE / "data/adjusted_player_scores.parquet", columns=["region", "overall"]
)
for r in REGIONS:
    a = overall_players.loc[overall_players.region.eq(r), "overall"].to_numpy()
    coverage.append(
        dict(
            region=r,
            lower=-1.2,
            upper=2.0,
            n_players=len(a),
            n_below=int((a < -1.2).sum()),
            n_above=int((a > 2).sum()),
            fraction_visible=float(((a >= -1.2) & (a <= 2)).mean()),
        )
    )
save(pd.DataFrame(coverage), "source_data/Fig2a_display_coverage.csv")
audit.update(
    n_players=ng,
    n_records=n,
    n_unique_matches=int(f.match_id.nunique()),
    excluded_records=6,
    design_rank=len(names),
    raw_and_score_row_identities_verified=True,
    raw_to_frozen_signal_max_errors=reconstruction,
    common_reference_matches_previous_section=True,
    bootstrap_replicates=1000,
    seed=191026,
    n_signal_tests=len(contrasts),
    n_signal_Holm_p_lt_001=int((contrasts.log10_p_holm < -3).sum()),
    old_new_signal_d_max_difference=float(recon.d_change.abs().max()),
    descriptive_identity_max_error=float(
        np.max(
            np.abs(
                desc.prevalence * desc.conditional_mean_per_10_min
                - desc.marginal_mean_per_10_min
            )
        )
    ),
    signal_omnibus=omnibus,
    record_count_distribution={
        str(k): int(v) for k, v in pd.Series(counts).value_counts().items()
    },
)
(OUT / "qa/analysis_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "qa/input_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    contrasts[contrasts.metric.isin(METRICS[5:])][
        [
            "metric",
            "contrast",
            "estimate",
            "standardized_difference",
            "d_ci_low",
            "d_ci_high",
            "p_holm",
        ]
    ].to_string(index=False),
    flush=True,
)
print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
