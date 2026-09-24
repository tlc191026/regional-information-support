"""Repair only the comparison affected by historical match-tier assignments.

The historical Iron-Gold match IDs are locked before attaching current canonical
tier metadata. B and C then use identical score transforms and control definitions.
"""

from settings import *
from statistics_core import *
from run_models import regional_profiles, clean_for

df = pd.read_parquet(DATA / "analysis_scored_0_10.parquet")
old = pd.read_parquet(DATA / "old_timeline_membership.parquet")
meta = pd.read_parquet(
    DATA / "timeline_targets_all.parquet", columns=["match_id", "tier"]
)
audit = old.merge(
    meta, on="match_id", suffixes=("_old", "_current"), validate="one_to_one"
)
audit.groupby(["tier_old", "tier_current"], observed=True).size().rename(
    "n_matches"
).reset_index().to_csv(TABLES / "old_current_tier_assignment.csv", index=False)
oldmask = df.match_id.isin(set(old.match_id))
lowids = set(old.loc[old.tier.isin(TIERS[:4]), "match_id"])
oldcommon = df[df.match_id.isin(lowids)]
scale = MaintenanceScale().fit(df[oldmask])
reference = scale.apply(df).maintenance_diff_raw
mean = float(reference.loc[oldcommon.index].mean())
sd = float(reference.loc[oldcommon.index].std(ddof=0))
work = df.copy()
work["maintenance_diff_z"] = (reference - mean) / sd
write_json(
    DATA / "old_sample_comparison_scale.json",
    {
        "transform_fit_sample": "old overlapping timeline matches",
        "difference_reference": "old Iron–Gold match IDs locked using old cached tier labels; regression controls use current canonical metadata in B and C",
        "diff_mean": mean,
        "diff_sd": sd,
        "maintenance_scale": scale.metadata(),
    },
)
result = pd.read_csv(TABLES / "sensitivity_models.csv", keep_default_na=False)
specs = [
    ("old_all_timeline_matches_corrected", work[oldmask]),
    ("old_Iron_Gold_matches_corrected", work.loc[oldcommon.index]),
    ("all_available_common_old_scale", work),
]
newrows = []
for name, sample in specs:
    fit = binary_fit(clean_for(sample))
    e = fit["effects"][0]
    e.update(specification=name, status="estimated")
    newrows.append(e)
    log(f'Reconciled {name}: n={e["n_matches"]:,}, beta={e["beta"]:.6f}')
result = result[~result.specification.isin([a for a, _ in specs])]
result = pd.concat([pd.DataFrame(newrows), result], ignore_index=True)
result["q_value_within_sensitivity_family"] = bh_q(
    pd.to_numeric(result.p_value, errors="coerce")
)
result.to_csv(TABLES / "sensitivity_models.csv", index=False)
om, oe, _ = regional_profiles(oldcommon)
om.to_csv(TABLES / "old_Iron_Gold_adjusted_means_corrected.csv", index=False)
oe.to_csv(TABLES / "old_Iron_Gold_contrasts_corrected.csv", index=False)
comparison = pd.read_csv(TABLES / "old_new_model_comparison.csv", keep_default_na=False)
for stage_prefix, spec in [
    ("B old", "old_Iron_Gold_matches_corrected"),
    ("B2 old", "old_all_timeline_matches_corrected"),
    ("C full", "all_available_common_old_scale"),
]:
    r = result[result.specification.eq(spec)].iloc[0]
    mask = comparison.stage.str.startswith(stage_prefix)
    for dest, src in [
        ("beta", "beta"),
        ("or", "odds_ratio"),
        ("ci_low", "ci_low"),
        ("ci_high", "ci_high"),
        ("n_matches", "n_matches"),
    ]:
        comparison.loc[mask, dest] = r[src]
    comparison.loc[mask, "source"] = (
        "old cohort IDs locked using historical tier; current canonical tier controls in B/C; fixed old-reference transform and difference SD; match-score HC0 CI"
    )
comparison.to_csv(TABLES / "old_new_model_comparison.csv", index=False)
write_json(
    QA / "old_sample_scope_reconciliation.json",
    {
        "old_match_count": len(old),
        "old_low_tier_match_count": len(lowids),
        "changed_tier_labels": int((audit.tier_old != audit.tier_current).sum()),
        "retained_old_low_tier_primary_outcomes": len(clean_for(oldcommon)),
        "historical_cohort_rule": "select old IDs using old cached tier, then join current canonical metadata",
        "model_controls": "same current canonical region/tier/patch in B and C",
        "previous_problem": "filtering old match IDs by current tier silently changed the historic low-tier cohort",
        "affected_models_only": [name for name, _ in specs],
        "code_sha256": sha(__file__),
        "time": datetime.datetime.now().isoformat(),
    },
)
log("Old-sample comparison reconciled")
