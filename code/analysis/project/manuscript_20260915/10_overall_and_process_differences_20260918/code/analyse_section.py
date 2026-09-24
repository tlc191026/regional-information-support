"""Joint regional estimates, adjusted player distributions and reproducible inference."""

from pathlib import Path
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import hashlib, json, gc
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.ndimage import gaussian_filter1d

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
REGIONS = ["NA", "EUW", "KR"]
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
MEASURES = ["overall", "maintenance", "signalling", "indicator_equal_weight"]
COMBINATIONS = np.array([[0.5, 0.5, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
PAIRS = [("KR", "NA"), ("KR", "EUW"), ("EUW", "NA")]
M = ["wards_placed_pm", "wards_killed_pm", "control_wards_bought_pm"]
SUB = [
    ["enemy_missing_pings_pm", "enemy_vision_pings_pm", "command_pings_pm"],
    ["need_vision_pings_pm", "assist_me_pings_pm"],
    ["on_my_way_pings_pm", "push_pings_pm", "all_in_pings_pm", "get_back_pings_pm"],
]
METRICS = M + sum(SUB, [])
manifest = {}
audit = {}


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    manifest[str(path.relative_to(ROOT))] = {
        "sha256": h.hexdigest(),
        "bytes": path.stat().st_size,
    }


def csv(frame, name, folder="source_data"):
    # Preserve extremely small Wald probabilities without exporting P=0.
    if "log10_p" in frame.columns:
        frame = frame.copy()
        logs = frame.log10_p.to_numpy(float)

        def sci(logp):
            exponent = int(np.floor(logp))
            return f"{10**(logp-exponent):.10f}e{exponent:+d}"

        frame["p_value"] = [sci(v) for v in logs]
        if "p_holm" in frame.columns:
            order = np.argsort(logs)
            adjusted = np.minimum(
                0,
                np.maximum.accumulate(
                    logs[order] + np.log10(len(logs) - np.arange(len(logs)))
                ),
            )
            corrected = np.empty(len(logs))
            corrected[order] = adjusted
            frame["log10_p_holm"] = corrected
            frame["p_holm"] = [sci(v) for v in corrected]
    frame.to_csv(OUT / folder / (name + ".csv"), index=False, encoding="utf-8-sig")


def z(x):
    return (x - x.mean()) / x.std(ddof=0)


def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    a = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    out = np.empty(len(p))
    out[o] = a
    return out


def p_fields(est, se):
    zz = est / se
    logp = float(np.log(2) + stats.norm.logsf(abs(zz)))
    return dict(
        se=float(se),
        z=float(zz),
        p_value=float(np.exp(logp)),
        log10_p=logp / np.log(10),
    )


print("Reading fixed scores and verifying component definitions", flush=True)
paths = [
    ROOT / "exp_20260717/data/context_unresidualized_participant_scores.parquet",
    ROOT / "exp_20260717/data/participant_metrics_transformed_z.parquet",
    ROOT / "exp_20260717/data/bootstrap_weights_primary_region_tier_1000.npy",
    ROOT / "exp_final_construct_v3/tables/regional_process_adjusted_means.csv",
    ROOT / "exp_final_construct_v3/code/constructs.py",
    ROOT / "exp_20260717/code/run_context_rerun.py",
    ROOT / "manuscript_20260915/manuscript_20260805.docx",
]
for p in paths:
    fingerprint(p)
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
        "shared_information_maintenance",
        "team_signalling",
    ],
)
metrics = pd.read_parquet(paths[1], columns=METRICS)
assert len(f) == len(metrics) == 2520000
rebuilt_m = z(metrics[M].to_numpy(float).mean(1))
sub = np.column_stack([z(metrics[names].to_numpy(float).mean(1)) for names in SUB])
rebuilt_s = z(sub.mean(1))
err_m = np.max(np.abs(rebuilt_m - f.shared_information_maintenance))
err_s = np.max(np.abs(rebuilt_s - f.team_signalling))
assert err_m < 5e-6 and err_s < 5e-6, (err_m, err_s)
f["indicator_equal_weight"] = metrics.to_numpy(float).mean(1)
audit["score_reconstruction_max_error"] = {
    "maintenance": float(err_m),
    "signalling": float(err_s),
}
del metrics, rebuilt_m, rebuilt_s, sub
f["region"] = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
valid = f.team_position.astype(str).isin(POSITIONS)
assert (~valid).sum() == 6
csv(
    f.loc[~valid, ["match_id", "puuid", "region", "tier", "team_position"]],
    "excluded_nonstandard_position_records",
    folder="data",
)
f = f.loc[valid].reset_index(drop=True)
assert len(f) == 2519994 and f.match_id.nunique() == 2240349
assert not f[["puuid", "match_id"]].duplicated().any()
labels, pc = np.unique(f.puuid.astype(str).to_numpy(), return_inverse=True)
pc = pc.astype(np.int32)
ng = len(labels)
n = len(f)
assert ng == 126000
counts = np.bincount(pc)
assert (counts == 19).sum() == 6 and (counts == 20).sum() == 125994
players = (
    f[["puuid", "region", "tier"]]
    .drop_duplicates("puuid")
    .set_index("puuid")
    .loc[labels]
    .reset_index()
)
assert players.region.value_counts().to_dict() == {
    "NA": 42000,
    "EUW": 42000,
    "KR": 42000,
}
strata = players.groupby(["region", "tier"], observed=True, sort=True).indices
assert len(strata) == 21 and all(len(v) == 6000 for v in strata.values())
G = sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ng, n))

print("Fitting the common regional adjustment", flush=True)
patches = sorted(
    f.patch.astype(str).unique(), key=lambda s: tuple(map(int, s.split(".")))
)
levels = {
    "region": REGIONS,
    "tier": TIERS,
    "team_position": POSITIONS,
    "patch": patches,
}
names = ["Intercept"]
row = [np.arange(n)]
col = [np.zeros(n, dtype=np.int32)]
data = [np.ones(n)]
L = np.zeros((3, 1 + 2 + 6 + 4 + len(patches) - 1 + 1))
L[:, 0] = 1
for variable, cats in levels.items():
    codes = pd.Categorical(f[variable].astype(str), categories=cats).codes
    assert (codes >= 0).all(), variable
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
duration = f.game_duration.to_numpy(float)
duration_mean = duration.mean()
duration_sd = duration.std(ddof=0)
row.append(np.arange(n))
col.append(np.full(n, len(names)))
data.append((duration - duration_mean) / duration_sd)
names.append("duration_z")
X = sparse.csr_matrix(
    (np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
    shape=(n, len(names)),
)
del row, col, data, duration
Y = f[
    ["shared_information_maintenance", "team_signalling", "indicator_equal_weight"]
].to_numpy(float)
gram = (X.T @ X).toarray()
bread = np.linalg.pinv(gram, rcond=1e-11, hermitian=True)
beta = bread @ (X.T @ Y)
assert np.linalg.matrix_rank(gram) == len(names)
E = Y - X @ beta
base_means = L @ beta
est = base_means @ COMBINATIONS.T
csv(
    pd.DataFrame(
        beta,
        index=names,
        columns=["maintenance", "signalling", "indicator_equal_weight"],
    ).reset_index(names="term"),
    "adjustment_coefficients",
)
csv(
    pd.DataFrame(L, index=REGIONS, columns=names).reset_index(names="region"),
    "reference_vectors",
)
old = pd.read_csv(paths[3], keep_default_na=False)
errs = []
reconciliation = []
for r in old.itertuples():
    j = {"shared_information_maintenance": 0, "team_signalling": 1}[r.measure]
    current = base_means[REGIONS.index(r.region), j]
    errs.append(abs(current - r.estimate))
    reconciliation.append(
        dict(
            region=r.region,
            measure=r.measure,
            frozen_estimate=r.estimate,
            current_estimate=current,
            difference=current - r.estimate,
        )
    )
audit["max_difference_from_frozen_process_means"] = max(errs)
audit["numeric_precision"] = (
    "Float64 design cross-products; previous stacked implementation accumulated float32 sparse cross-products."
)
csv(pd.DataFrame(reconciliation), "frozen_process_mean_reconciliation")
assert max(errs) < 1e-4, max(errs)

print(
    "Preparing adjusted player scores and joint cluster influence functions", flush=True
)
record_region = pd.Categorical(f.region, categories=REGIONS).codes
player_region = pd.Categorical(players.region, categories=REGIONS).codes
px = (G @ X).toarray() / counts[:, None]
base_player = (G @ E) / counts[:, None] + base_means[player_region]
vals = base_player @ COMBINATIONS.T
assert np.max(np.abs(vals[:, 0] - (vals[:, 1] + vals[:, 2]) / 2)) < 1e-12
players["n_records"] = counts
for j, m in enumerate(MEASURES):
    players[m] = vals[:, j]
players.to_parquet(OUT / "data/adjusted_player_scores.parquet", index=False)
inf_beta = []
for j in range(3):
    scores = (G @ X.multiply(E[:, j, None])).toarray()
    inf_beta.append(scores @ bread)
inf_beta = np.stack(inf_beta, axis=2)
mean_inf = np.empty((ng, 3, 4))
var_inf = np.empty((ng, 3, 4))
variances = np.empty((3, 4))
mean_differences = []
for r in range(3):
    use = player_region == r
    nn = use.sum()
    H = px[use] - L[r]
    for j, w in enumerate(COMBINATIONS):
        influence = inf_beta @ w
        mean_inf[:, r, j] = influence @ L[r]
        a = vals[use, j]
        dev = a - a.mean()
        v = dev @ dev / (nn - 1)
        variances[r, j] = v
        gradient = -2 * (dev @ H) / (nn - 1)
        var_inf[:, r, j] = influence @ gradient
        var_inf[use, r, j] += (dev**2 - np.mean(dev**2)) / (nn - 1)
        mean_differences.append(abs(a.mean() - est[r, j]))
audit["max_player_distribution_mean_minus_reference_estimate"] = max(mean_differences)
assert max(mean_differences) < 2e-5
mean_inf = mean_inf.reshape(ng, 12)
var_inf = var_inf.reshape(ng, 12)
joint = np.column_stack([mean_inf, var_inf])
for members in strata.values():
    joint[members] -= joint[members].mean(axis=0)
player_cov = joint[:, :12].T @ joint[:, :12] * (6000 / 5999)

print("Applying 1,000 paired, stratified player resamples", flush=True)
weights = np.load(paths[2], mmap_mode="r")
assert weights.shape == (1000, ng)
for members in strata.values():
    assert np.all(np.asarray(weights[:, members]).sum(axis=1) == 6000)
boot = np.empty((1000, 24))
point = np.r_[est.ravel(), variances.ravel()]
for start in range(0, 1000, 25):
    end = min(1000, start + 25)
    boot[start:end] = point + (np.asarray(weights[start:end], dtype=float) - 1) @ joint
    if start % 250 == 0:
        print(f"Bootstrap {end}/1000", flush=True)
assert np.isfinite(boot).all() and (boot[:, 12:] > 0).all()
boot_mean = boot[:, :12].reshape(1000, 3, 4)
boot_var = boot[:, 12:].reshape(1000, 3, 4)
csv(
    pd.DataFrame(
        boot,
        columns=[
            f"{kind}_{r}_{m}"
            for kind in ["mean", "variance"]
            for r in REGIONS
            for m in MEASURES
        ],
    ),
    "joint_player_bootstrap_replicates",
    folder="data",
)
ci = np.quantile(boot_mean, [0.025, 0.975], axis=0)
means = []
for r, region in enumerate(REGIONS):
    for j, m in enumerate(MEASURES):
        means.append(
            dict(
                region=region,
                measure=m,
                estimate=est[r, j],
                ci_low=ci[0, r, j],
                ci_high=ci[1, r, j],
                n_players=42000,
                n_records=int(counts[player_region == r].sum()),
                distribution_mean=vals[player_region == r, j].mean(),
            )
        )
csv(pd.DataFrame(means), "regional_adjusted_means")


def estimate_row(vector, measure, contrast, draws):
    ee = float(vector @ est.ravel())
    se = np.sqrt(vector @ player_cov @ vector)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(
        measure=measure,
        contrast=contrast,
        estimate=ee,
        ci_low=lo,
        ci_high=hi,
        **p_fields(ee, se),
    )


primary = []
secondary = []
vectors = []
for a, b in PAIRS:
    ia, ib = REGIONS.index(a), REGIONS.index(b)
    for j, m in enumerate(MEASURES):
        v = np.zeros(12)
        v[ia * 4 + j] = 1
        v[ib * 4 + j] = -1
        draws = boot_mean[:, ia, j] - boot_mean[:, ib, j]
        rr = estimate_row(v, m, a + "-" + b, draws)
        denom = np.sqrt((variances[ia, j] + variances[ib, j]) / 2)
        d_draw = draws / np.sqrt((boot_var[:, ia, j] + boot_var[:, ib, j]) / 2)
        dl, dh = np.quantile(d_draw, [0.025, 0.975])
        rr.update(
            standardized_difference=rr["estimate"] / denom,
            d_ci_low=dl,
            d_ci_high=dh,
            pooled_player_sd=denom,
        )
        if j < 3:
            primary.append(rr)
            vectors.append(v)
        else:
            secondary.append(rr)
for a, b in PAIRS:
    ia, ib = REGIONS.index(a), REGIONS.index(b)
    v = np.zeros(12)
    v[ia * 4 + 1] = 1
    v[ib * 4 + 1] = -1
    v[ia * 4 + 2] = -1
    v[ib * 4 + 2] = 1
    draws = (boot_mean[:, ia, 1] - boot_mean[:, ib, 1]) - (
        boot_mean[:, ia, 2] - boot_mean[:, ib, 2]
    )
    rr = estimate_row(v, "maintenance_minus_signalling", a + "-" + b, draws)
    primary.append(rr)
    vectors.append(v)
results = pd.DataFrame(primary)
results["p_holm"] = holm(results.p_value)
results["correction_family"] = "12 primary regional and between-process comparisons"
csv(results, "primary_comparisons", folder="tables")
secondary = pd.DataFrame(secondary)
secondary["p_holm"] = holm(secondary.p_value)
csv(secondary, "indicator_equal_weight_sensitivity", folder="tables")

print("Checking shared-match dependence with two-way clustered covariance", flush=True)
# Record influence for the 12 region/measure means. Duplicated matches alone
# determine the difference between the match and record covariance terms.
projected = X @ (bread @ L.T)
rec_inf = np.empty((n, 12))
for r in range(3):
    rec_inf[:, r * 4 : (r + 1) * 4] = (E * projected[:, r, None]) @ COMBINATIONS.T
assert np.max(np.abs(G @ rec_inf - mean_inf)) < 1e-9
rec_cov = rec_inf.T @ rec_inf
mc, mlabels = pd.factorize(f.match_id, sort=False)
match_counts = np.bincount(mc)
nm = len(mlabels)
dup = match_counts[mc] > 1
duplabels, di = np.unique(mc[dup], return_inverse=True)
J = sparse.csr_matrix(
    (np.ones(dup.sum()), (di, np.arange(dup.sum()))), shape=(len(duplabels), dup.sum())
)
match_sums = J @ rec_inf[dup]
match_cov = rec_cov + match_sums.T @ match_sums - rec_inf[dup].T @ rec_inf[dup]
two_way = (
    (
        ng / (ng - 1) * (mean_inf.T @ mean_inf)
        + nm / (nm - 1) * match_cov
        - n / (n - 1) * rec_cov
    )
    * (n - 1)
    / (n - len(names))
)
tw = []
for rr, v in zip(primary, vectors):
    se = float(np.sqrt(v @ two_way @ v))
    ee = rr["estimate"]
    tw.append(
        dict(
            measure=rr["measure"],
            contrast=rr["contrast"],
            estimate=ee,
            ci_low=ee - 1.96 * se,
            ci_high=ee + 1.96 * se,
            **p_fields(ee, se),
            primary_se=rr["se"],
            se_ratio=se / rr["se"],
        )
    )
tw = pd.DataFrame(tw)
tw["p_holm"] = holm(tw.p_value)
csv(tw, "two_way_cluster_sensitivity", folder="tables")
audit["two_way_covariance_min_eigenvalue"] = float(np.linalg.eigvalsh(two_way).min())
audit["shared_match_records"] = int(dup.sum())
audit["shared_matches"] = int(len(duplabels))
assert np.all(tw.ci_low * tw.ci_high > 0)
del X, E, Y, px, inf_beta, projected, rec_inf, J, match_sums, joint, G
gc.collect()

print("Writing all-player empirical distributions and figure source data", flush=True)
summary = []
ecdf = []
densities = []
quantiles = []
for j, m in enumerate(MEASURES[:3]):
    pooled = vals[:, j]
    lower = pooled.min()
    upper = pooled.max()
    # A single pooled bandwidth and density width apply to all regions of a measure.
    bins = np.linspace(lower - 1e-8, upper + 1e-8, 1601)
    centers = (bins[:-1] + bins[1:]) / 2
    bw = bins[1] - bins[0]
    bandwidth = pooled.std(ddof=1) * 42000 ** (-0.2)
    for r, region in enumerate(REGIONS):
        a = vals[player_region == r, j]
        sorted_a = np.sort(a)
        if m == "overall":
            ecdf.append(
                pd.DataFrame(
                    {
                        "region": region,
                        "score": sorted_a,
                        "cumulative_fraction": np.arange(1, len(a) + 1) / len(a),
                    }
                )
            )
        prob = [
            0,
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
            1,
        ]
        q = np.quantile(a, prob)
        s = dict(
            region=region,
            measure=m,
            n_players=len(a),
            mean=a.mean(),
            sd=a.std(ddof=1),
            min=q[0],
            q001=q[1],
            q005=q[2],
            q01=q[3],
            q025=q[4],
            q05=q[5],
            q10=q[6],
            q25=q[7],
            median=q[8],
            q75=q[9],
            q90=q[10],
            q95=q[11],
            q975=q[12],
            q99=q[13],
            q995=q[14],
            q999=q[15],
            max=q[16],
        )
        summary.append(s)
        quantiles.extend(
            dict(region=region, measure=m, probability=p, value=v)
            for p, v in zip(prob, q)
        )
        hist, _ = np.histogram(a, bins)
        assert hist.sum() == len(a)
        density = gaussian_filter1d(hist.astype(float), bandwidth / bw, mode="constant")
        density /= density.sum() * bw
        densities.append(
            pd.DataFrame(
                {
                    "region": region,
                    "measure": m,
                    "score": centers,
                    "count": hist,
                    "density": density,
                    "bandwidth": bandwidth,
                    "bin_width": bw,
                }
            )
        )
csv(pd.DataFrame(summary), "player_distribution_summaries")
csv(pd.concat(ecdf, ignore_index=True), "Fig2a_all_player_ECDF")
csv(pd.concat(densities, ignore_index=True), "all_player_densities")
csv(pd.DataFrame(quantiles), "player_quantiles")
csv(
    players.groupby(["region", "tier"], observed=True)
    .agg(n_players=("puuid", "size"), n_records=("n_records", "sum"))
    .reset_index(),
    "sample_by_region_tier",
)
audit.update(
    n_players=ng,
    n_records=n,
    n_unique_matches=nm,
    n_excluded_records=6,
    seed=191026,
    bootstrap_replicates=1000,
    region_player_counts=players.region.value_counts().to_dict(),
    record_count_distribution=pd.Series(counts).value_counts().to_dict(),
    design_rank=len(names),
    design_condition_number=float(np.linalg.cond(gram)),
    duration_reference=float(duration_mean),
    all_primary_Holm_p_below_001=bool((results.p_holm < 0.001).all()),
    all_two_way_intervals_preserve_direction=bool(np.all(tw.ci_low * tw.ci_high > 0)),
    all_equal_indicator_weight_comparisons_positive=bool((secondary.ci_low > 0).all()),
    primary_analysis_unit="participant-match scores; player-cluster inference and player-mean displays",
    reference="equal weights for seven sampling tiers and five positions; empirical patch frequencies; pooled mean duration",
    standardization="M and S frozen global input scales; I=(M+S)/2; adjusted player SD for standardized contrasts",
    bootstrap="joint one-step coefficient and variance updates; region x sampling-tier stratified player pairs; fixed reference distribution",
    primary_test="two-sided Wald using stratified player-cluster influence variance; Holm over 12 contrasts",
    sensitivity_test="two-way cluster by focal player and unique match, subtract player-match intersection; normal 95% CI",
)
(OUT / "qa/input_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "qa/analysis_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    results[
        [
            "measure",
            "contrast",
            "estimate",
            "ci_low",
            "ci_high",
            "standardized_difference",
            "p_holm",
        ]
    ].to_string(index=False),
    flush=True,
)
print(
    pd.DataFrame(summary)[
        ["measure", "region", "q01", "q25", "median", "q75", "q99", "min", "max"]
    ].to_string(index=False),
    flush=True,
)
print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
