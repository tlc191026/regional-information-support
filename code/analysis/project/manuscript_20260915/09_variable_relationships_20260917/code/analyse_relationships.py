"""Player-level relationship diagnostics; no participant-level independence tests.

Run with python. All outputs stay beside this script's parent.
"""

from pathlib import Path
from itertools import combinations
import hashlib
import json
import platform
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
for directory in ["source_data", "tables", "qa", "figures"]:
    (OUT / directory).mkdir(exist_ok=True)
DATA = OUT / "source_data"
FROZEN = ROOT / "exp_final_construct_v3"
PLAYER = ROOT / "exp_20260717/data/player_metric_and_process_scores_minimal.parquet"
DICTIONARY = FROZEN / "tables/construct_dictionary_18_metrics.csv"
REGIONS = ["NA", "EUW", "KR"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


inputs = [
    PLAYER,
    DICTIONARY,
    FROZEN / "tables/data_audit_canonical_invariants.json",
    FROZEN / "tables/metric_preprocessing_audit_minimal.csv",
    ROOT / "exp_20260717/code/common.py",
    ROOT / "exp_20260717/code/run_core_analysis.py",
    FROZEN / "tables/construct_boundary_summary.csv",
    FROZEN / "tables/construct_boundary_permutation_test.csv",
]
manifest = {
    str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
    for p in inputs
}
sample_audit = json.loads(
    (FROZEN / "tables/data_audit_canonical_invariants.json").read_text(encoding="utf-8")
)
assert sample_audit["unique_players"] == 126000
assert (
    sample_audit["participant_match_rows"] == 2520000
    and sample_audit["unique_matches"] == 2240354
)
assert (
    sample_audit["matches_per_player_min"]
    == sample_audit["matches_per_player_max"]
    == 20
)
(OUT / "qa/inherited_sample_invariants.json").write_text(
    json.dumps(sample_audit, indent=2), encoding="utf-8"
)

dictionary = pd.read_csv(DICTIONARY, keep_default_na=False)
# Move the diagnostic to the end. All primary indicators retain the frozen order.
dictionary = pd.concat([dictionary.iloc[1:], dictionary.iloc[:1]], ignore_index=True)
dictionary.insert(0, "display_index", np.arange(1, 19))
dictionary.to_csv(DATA / "metric_dictionary.csv", index=False, encoding="utf-8-sig")
metrics = dictionary.metric.tolist()
player = pd.read_parquet(PLAYER)
assert len(player) == 126000 and player.puuid.nunique() == 126000
player["region"] = player.region.replace({"NA1": "NA", "EUW1": "EUW"})
counts = player.groupby("region", observed=True).size()
assert set(counts.index) == set(REGIONS) and (counts == 42000).all()
assert np.isfinite(player[metrics].to_numpy()).all()
assert (player[metrics].std() > 0).all()
assert dictionary.loc[dictionary.metric.eq("deaths_pm"), "reverse"].item()
player.groupby(["region", "tier"], observed=True).size().rename(
    "n_players"
).reset_index().to_csv(DATA / "sample_by_region_tier.csv", index=False)

centered = player[metrics] - player.groupby("region", observed=True)[metrics].transform(
    "mean"
)
matrices = {
    "player_region_centered_Pearson": centered.corr(),
    "player_pooled_Pearson": player[metrics].corr(),
    "player_pooled_Spearman": player[metrics].corr(method="spearman"),
}
for region in REGIONS:
    x = player.loc[player.region.eq(region), metrics]
    matrices[f"player_{region}_Pearson"] = x.corr()
    matrices[f"player_{region}_Spearman"] = x.corr(method="spearman")
# Partial rank correlation: rank in the full player sample, then remove region means.
ranks = player[metrics].rank(method="average")
rank_centered = ranks - ranks.groupby(player.region, observed=True).transform("mean")
matrices["player_region_centered_rank"] = rank_centered.corr()

frozen_checks = []
for current, old in [
    ("player_region_centered_Pearson", "region_adjusted"),
    ("player_pooled_Pearson", "pearson"),
    ("player_pooled_Spearman", "spearman"),
    *[(f"player_{r}_Pearson", r) for r in REGIONS],
]:
    path = FROZEN / "tables" / f"player_residual_correlation_{old}.csv"
    assert path.exists(), path
    frozen = pd.read_csv(path, index_col=0).loc[metrics, metrics]
    error = float(np.max(np.abs(frozen.to_numpy() - matrices[current].to_numpy())))
    assert error < 1e-6, (current, error)
    frozen_checks.append(dict(matrix=current, max_absolute_error=error))
    manifest[str(path.relative_to(ROOT))] = {
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }

record_path = FROZEN / "tables/residual_correlation_matrix_18_metrics.csv"
matrices["legacy_record_Pearson"] = pd.read_csv(record_path, index_col=0).loc[
    metrics, metrics
]
manifest[str(record_path.relative_to(ROOT))] = {
    "sha256": sha(record_path),
    "bytes": record_path.stat().st_size,
}

M = metrics[:3]
S = metrics[3:12]
C = metrics[13:17]
A = ["assists_pm"]
V = ["vision_score_pm"]


def pairs_in(xs):
    return list(combinations(xs, 2))


def pairs_between(xs, ys):
    return [(x, y) for x in xs for y in ys]


signal_groups = [S[:3], S[3:5], S[5:]]
within_signal_sub = [p for group in signal_groups for p in pairs_in(group)]
within_set = {frozenset(p) for p in within_signal_sub}
between_signal_sub = [p for p in pairs_in(S) if frozenset(p) not in within_set]
blocks = {
    "within_M": pairs_in(M),
    "within_S": pairs_in(S),
    "between_M_S": pairs_between(M, S),
    "within_M_and_S": pairs_in(M) + pairs_in(S),
    "within_signal_subgroups": within_signal_sub,
    "between_signal_subgroups": between_signal_sub,
    "within_combat": pairs_in(C),
    "M_combat": pairs_between(M, C),
    "S_combat": pairs_between(S, C),
    "assists_M": pairs_between(A, M),
    "assists_S": pairs_between(A, S),
    "assists_combat": pairs_between(A, C),
    "vision_M": pairs_between(V, M),
}
summary_rows = []
pair_rows = []
for key, matrix in matrices.items():
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
    assert np.abs(matrix.to_numpy()).max() <= 1 + 1e-12
    matrix.to_csv(DATA / f"{key}.csv", index_label="metric", encoding="utf-8-sig")
    for i, j in combinations(range(18), 2):
        pair_rows.append(
            dict(
                matrix=key,
                index_a=i + 1,
                index_b=j + 1,
                metric_a=metrics[i],
                metric_b=metrics[j],
                r=float(matrix.iloc[i, j]),
            )
        )
    for block, pairs in blocks.items():
        values = np.array([matrix.loc[x, y] for x, y in pairs], dtype=float)
        summary_rows.append(
            dict(
                matrix=key,
                relationship=block,
                n_pairs=len(values),
                mean_r=values.mean(),
                mean_abs_r=np.abs(values).mean(),
                median_r=np.median(values),
                min_r=values.min(),
                max_r=values.max(),
                n_negative=int((values < 0).sum()),
            )
        )
summary = pd.DataFrame(summary_rows)
summary.to_csv(DATA / "relationship_summaries.csv", index=False)
pd.DataFrame(pair_rows).to_csv(DATA / "all_pair_correlations.csv", index=False)


def exhaustive_labels(sizes):
    n = sum(sizes)
    universe = tuple(range(n))

    def recurse(remaining, k, parts):
        if k == len(sizes) - 1:
            labels = np.empty(n, dtype=int)
            for label, indices in enumerate(parts + [remaining]):
                labels[list(indices)] = label
            yield labels
            return
        for part in combinations(remaining, sizes[k]):
            chosen = set(part)
            yield from recurse(
                tuple(x for x in remaining if x not in chosen), k + 1, parts + [part]
            )

    return np.array(list(recurse(universe, 0, [])))


tests = []
null_rows = []
for name, subset, sizes in [
    ("maintenance_vs_signalling", M + S, [3, 9]),
    ("signal_functional_subgroups", S, [3, 2, 4]),
]:
    allocations = exhaustive_labels(sizes)
    expected = 220 if len(sizes) == 2 else 1260
    assert len(allocations) == expected
    assert len({tuple(x) for x in allocations}) == expected
    observed_labels = np.repeat(np.arange(len(sizes)), sizes)
    observed_index = np.flatnonzero(np.all(allocations == observed_labels, axis=1))
    assert len(observed_index) == 1
    ii, jj = np.triu_indices(len(subset), 1)
    within = allocations[:, ii] == allocations[:, jj]
    assert np.all(within.sum(axis=1) == sum(n * (n - 1) // 2 for n in sizes))
    for key, matrix in matrices.items():
        values = np.abs(matrix.loc[subset, subset].to_numpy()[ii, jj])
        null = (within @ values) / within.sum(axis=1) - ((~within) @ values) / (
            ~within
        ).sum(axis=1)
        observed = float(null[observed_index[0]])
        assert abs(float(null.mean())) < 1e-12
        n_at_least = int(np.count_nonzero(null >= observed - 1e-12))
        p = n_at_least / len(null)
        tests.append(
            dict(
                matrix=key,
                test=name,
                statistic=observed,
                n_allocations=len(null),
                n_at_least_observed=n_at_least,
                p_exact=p,
                null_q025=float(np.quantile(null, 0.025)),
                null_q975=float(np.quantile(null, 0.975)),
                tail="upper; ties included; exhaustive enumeration",
                interpretation="indicator-label structure diagnostic; not a player-sampling P value",
            )
        )
        if key == "player_region_centered_Pearson":
            for k, value in enumerate(null):
                null_rows.append(
                    dict(
                        test=name,
                        allocation=k,
                        statistic=value,
                        is_observed=bool(k == observed_index[0]),
                    )
                )
tests = pd.DataFrame(tests)
tests["p_holm_primary_two_tests"] = np.nan
mask = tests.matrix.eq("player_region_centered_Pearson")
indices = tests.index[mask].to_numpy()
sorted_indices = indices[np.argsort(tests.loc[indices, "p_exact"].to_numpy())]
adjusted = np.minimum(
    1.0, np.maximum.accumulate(tests.loc[sorted_indices, "p_exact"].to_numpy() * [2, 1])
)
tests.loc[sorted_indices, "p_holm_primary_two_tests"] = adjusted
tests.to_csv(DATA / "exact_grouping_diagnostics.csv", index=False)
pd.DataFrame(null_rows).to_csv(
    DATA / "primary_exact_label_distributions.csv", index=False
)

similarity = []
for r1, r2 in combinations(REGIONS, 2):
    for domain, subset in [("all_18", metrics), ("information_12", M + S)]:
        a = matrices[f"player_{r1}_Pearson"].loc[subset, subset].to_numpy()
        b = matrices[f"player_{r2}_Pearson"].loc[subset, subset].to_numpy()
        ii, jj = np.triu_indices(len(subset), 1)
        va, vb = a[ii, jj], b[ii, jj]
        difference = va - vb
        similarity.append(
            dict(
                region_a=r1,
                region_b=r2,
                domain=domain,
                n_pairs=len(va),
                matrix_profile_correlation=np.corrcoef(va, vb)[0, 1],
                mean_absolute_difference=np.mean(np.abs(difference)),
                max_absolute_difference=np.max(np.abs(difference)),
                sign_agreement_fraction=np.mean(np.sign(va) == np.sign(vb)),
            )
        )
pd.DataFrame(similarity).to_csv(
    DATA / "regional_matrix_similarity_descriptive.csv", index=False
)

# Deliberately separate changed unit / pair sets from old frozen tests.
old_summary = pd.read_csv(FROZEN / "tables/construct_boundary_summary.csv")
old_summary.to_csv(DATA / "legacy_record_boundary_summary_unchanged.csv", index=False)
old_tests = pd.read_csv(FROZEN / "tables/construct_boundary_permutation_test.csv")
old_tests.to_csv(DATA / "legacy_record_permutation_tests_unchanged.csv", index=False)

manifest[str(Path(__file__).relative_to(ROOT))] = {
    "sha256": sha(Path(__file__)),
    "bytes": Path(__file__).stat().st_size,
}
(OUT / "qa/input_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
qa = dict(
    n_players=len(player),
    unique_players=int(player.puuid.nunique()),
    players_per_region=counts.to_dict(),
    finite_all_18=True,
    no_second_death_reversal=True,
    frozen_matrix_reproduction=frozen_checks,
    exact_allocation_counts=[220, 1260],
    permutation_null_mean_zero=True,
    rows_do_not_identify_players=True,
    source_aggregation="20-match mean of context-adjusted indicator residuals",
    preprocessing_inherited="log1p nonnegative per-minute rates, 1st/99th percentile winsorization; Ridge alpha=1e-6 on tier, position, patch and game duration; residual standardization; deaths reversed once",
    uncertainty="No CI over correlated metric pairs; label-null ranges are not confidence intervals. Shared matches remain a player-dependence limitation.",
    versions=dict(
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__
    ),
    seed=191026,
)
(OUT / "qa/analysis_checks.json").write_text(
    json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    summary.loc[
        summary.matrix.isin(
            [
                "player_region_centered_Pearson",
                "player_NA_Pearson",
                "player_EUW_Pearson",
                "player_KR_Pearson",
            ]
        )
    ].to_string(index=False)
)
print(tests.loc[mask].to_string(index=False))
print(pd.DataFrame(similarity).to_string(index=False))
