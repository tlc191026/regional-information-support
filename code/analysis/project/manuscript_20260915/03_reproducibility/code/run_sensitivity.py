from settings import *
from statistics_core import *
from run_models import regional_profiles, load_scored, clean_for
import gc


def main():
    df = pd.read_parquet(DATA / "analysis_scored_0_10.parquet")
    rows = []

    def run(
        name,
        work,
        outcome="first_post_neutral_blue",
        predictors=["maintenance_diff_z"],
        controls=STATE,
        cats=["region", "tier", "patch"],
        weights=None,
    ):
        work = clean_for(work, outcome, extra=[*predictors, *controls])
        if len(work) < 100 or work[outcome].nunique() < 2:
            rows.append(
                {
                    "specification": name,
                    "n_matches": len(work),
                    "status": "insufficient_outcome_variation",
                }
            )
            return
        fitted = binary_fit(work, outcome, predictors, cats, controls, weights)
        for e in fitted["effects"]:
            e.update(specification=name, status="estimated")
            rows.append(e)
        log(
            f'Sensitivity {name}: n={len(work):,}, beta={fitted["effects"][0]["beta"]:.6f}'
        )
        pd.DataFrame(rows).to_csv(TABLES / "sensitivity_models.csv", index=False)
        del fitted
        gc.collect()

    old = pd.read_parquet(DATA / "old_timeline_membership.parquet")
    old_ids = set(old.match_id)
    oldmask = df.match_id.isin(old_ids)
    old_low_ids = set(old.loc[old.tier.isin(TIERS[:4]), "match_id"])
    oldcommon = df[df.match_id.isin(old_low_ids)]
    old_scale = MaintenanceScale().fit(df[oldmask])
    common_reference = old_scale.apply(df).maintenance_diff_raw
    reference_mean = float(common_reference.loc[oldcommon.index].mean())
    reference_sd = float(common_reference.loc[oldcommon.index].std(ddof=0))
    comparable = df.copy()
    comparable["maintenance_diff_z"] = (
        common_reference - reference_mean
    ) / reference_sd
    run("old_all_timeline_matches_corrected", comparable[oldmask])
    run("old_Iron_Gold_matches_corrected", comparable.loc[oldcommon.index])
    run("all_available_common_old_scale", comparable)
    write_json(
        DATA / "old_sample_comparison_scale.json",
        {
            "transform_fit_sample": "old overlapping timeline matches",
            "difference_reference": "old Iron–Gold match IDs locked using old cached tier labels; regression controls use current canonical metadata in B and C",
            "diff_mean": reference_mean,
            "diff_sd": reference_sd,
            "maintenance_scale": old_scale.metadata(),
        },
    )
    del comparable
    gc.collect()
    om, oe, _ = regional_profiles(oldcommon)
    om.to_csv(TABLES / "old_Iron_Gold_adjusted_means_corrected.csv", index=False)
    oe.to_csv(TABLES / "old_Iron_Gold_contrasts_corrected.csv", index=False)
    run("newly_available_matches_only", df[~oldmask])
    run("all_available_Iron_Gold", df[df.tier.isin(TIERS[:4])])
    run("all_available_Platinum_Diamond", df[df.tier.isin(TIERS[4:])])
    run("add_final_duration", df, controls=STATE + ["game_duration"])
    run("omit_sampling_time_tier", df, cats=["region", "patch"])
    run("first_legally_assigned_objective", df, outcome="first_legal_post_neutral_blue")
    for col in COMPONENTS:
        raw = df["blue_" + col + "_z"] - df["red_" + col + "_z"]
        df[col + "_diff_z"] = (raw - raw.mean()) / raw.std(ddof=0)
        run("component_" + col, df, predictors=[col + "_diff_z"])
    run(
        "three_components_simultaneously",
        df,
        predictors=[c + "_diff_z" for c in COMPONENTS],
    )
    for name, components in [
        ("deployment_and_purchase", ["valid_wards_placed", "control_ward_purchases"]),
        ("placement_and_clearing", ["valid_wards_placed", "valid_ward_kills"]),
        (
            "purchase_net_of_undo",
            ["valid_wards_placed", "valid_ward_kills", "control_ward_undo_net"],
        ),
    ]:
        scale = MaintenanceScale().fit(df, components)
        scored = scale.apply(df)
        work = df.copy()
        work["maintenance_diff_z"] = scored.maintenance_diff_z
        run(name, work)
        del work, scored
    # Same common global SD in tier-specific slope estimates.
    tiermeans = []
    tiercontrasts = []
    for tier in TIERS:
        subset = df[df.tier.eq(tier)]
        run("tier_" + tier, subset, cats=["region", "patch"])
        tm, tc, _ = regional_profiles(subset)
        tm["tier"] = tier
        tc["tier"] = tier
        tiermeans.append(tm)
        tiercontrasts.append(tc)
    pd.concat(tiermeans).to_csv(TABLES / "early_tier_region_profiles.csv", index=False)
    pd.concat(tiercontrasts).to_csv(
        TABLES / "early_tier_region_contrasts.csv", index=False
    )
    for period, mask in [
        ("2024", df.game_date.dt.year.eq(2024)),
        ("2025", df.game_date.dt.year.eq(2025)),
        ("2026", df.game_date.dt.year.eq(2026)),
    ]:
        run("calendar_" + period, df[mask])
    for patch, work in df.groupby("patch", observed=True):
        if len(work) >= 1000:
            run("patch_" + str(patch), work, cats=["region", "tier"])
    disjoint = set(
        pd.read_parquet(DATA / "all_participant_disjoint_matches.parquet").match_id
    )
    run("all_ten_participants_disjoint", df[df.match_id.isin(disjoint)])
    composition = []
    for label, sample in [
        ("all_primary_outcomes", clean_for(df)),
        ("all_participants_disjoint", clean_for(df[df.match_id.isin(disjoint)])),
    ]:
        g = (
            sample.groupby(["region", "tier"], observed=True)
            .agg(
                n_matches=("match_id", "size"),
                mean_duration=("game_duration", "mean"),
                mean_maintenance_difference=("maintenance_diff_z", "mean"),
                mean_absolute_gold_difference=(
                    "gold_at_window_diff",
                    lambda x: x.abs().mean(),
                ),
                first_game=("game_date", "min"),
                last_game=("game_date", "max"),
            )
            .reset_index()
        )
        g["sample"] = label
        g["sample_fraction"] = g.n_matches / len(sample)
        composition.append(g)
    pd.concat(composition).to_csv(
        TABLES / "participant_disjoint_selection_composition.csv", index=False
    )
    base = clean_for(df)
    minimum = int(base.groupby(["region", "tier"], observed=True).size().min())
    balanced = base.groupby(["region", "tier"], observed=True, group_keys=False).sample(
        n=minimum, random_state=SEED
    )
    run("region_tier_balanced_matches", balanced)
    # Coverage sensitivity uses the fixed canonical target universe, not observed outcomes.
    meta = pd.read_parquet(DATA / "canonical_match_audit.parquet")
    meta["included"] = meta.match_id.isin(df.match_id).astype(int)
    design = Design(["region", "tier", "patch"], ["game_duration"]).fit(meta)
    x = design.transform(meta)
    coverage = LogisticRegression(
        C=1.0,
        solver="newton-cholesky",
        max_iter=100,
        tol=1e-8,
        fit_intercept=False,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coverage.fit(x, meta.included)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Coverage model did not converge")
    p = np.maximum(coverage.predict_proba(x)[:, 1], 0.05)
    weight = pd.Series(meta.included.mean() / p, index=meta.match_id)
    w = base.match_id.map(weight).to_numpy(float)
    limits = np.quantile(w, [0.01, 0.99])
    w = np.clip(w, *limits)
    run("coverage_inverse_probability_weighted", base, weights=w)
    write_json(
        QA / "coverage_weighting.json",
        {
            "overall_inclusion_probability": float(meta.included.mean()),
            "minimum_predicted_probability": float(p.min()),
            "weight_p01": float(limits[0]),
            "weight_p99": float(limits[1]),
            "effective_sample_size": float(w.sum() ** 2 / (w @ w)),
            "specification": "regularized inclusion model using region, sampling tier, patch and duration; stabilized IPW winsorized at p01/p99",
        },
    )
    del meta, x, coverage
    gc.collect()
    for threshold in [1000, 1500, 2000]:
        work = df[df.gold_at_window_diff.abs().ge(threshold)].copy()
        orientation = np.where(work.gold_at_window_diff < 0, 1.0, -1.0)
        work["underdog_win"] = np.where(
            orientation == 1, work.blue_win, 1 - work.blue_win
        )
        work["blue_is_underdog"] = (orientation == 1).astype(int)
        global_scale = json.loads(
            (DATA / "maintenance_scale_primary.json").read_text(encoding="utf-8")
        )
        work["maintenance_diff_z"] = (
            orientation * work.maintenance_diff_raw / global_scale["diff_sd"]
        )
        for c in STATE:
            work[c] = work[c] * orientation
        run(
            "exploratory_comeback_" + str(threshold),
            work,
            outcome="underdog_win",
            controls=STATE + ["blue_is_underdog"],
        )
    strict, strict_scale = load_scored("strict_0_10")
    run("strict_nominal_10min_earlier_state_frame", strict)
    del strict
    gc.collect()
    later, scale15 = load_scored("0_15")
    run("window_0_15", later)
    write_json(DATA / "maintenance_scale_0_15.json", scale15.metadata())
    result = pd.DataFrame(rows)
    result["q_value_within_sensitivity_family"] = bh_q(result.p_value)
    result.to_csv(TABLES / "sensitivity_models.csv", index=False)
    # Explicit A/B/C comparison: historic frozen results versus corrected overlapping/full samples.
    frozen = json.loads(
        (
            ROOT / "exp_final_construct_v3/timeline_machine_readable_summary.json"
        ).read_text(encoding="utf-8")
    )
    prior = next(
        x
        for x in frozen["prospective_outcomes"]
        if x["outcome"] == "first_post_neutral_blue"
    )
    comparison = [
        {
            "stage": "A frozen old Iron–Gold",
            "beta": prior["beta"],
            "or": prior["odds_ratio"],
            "ci_low": prior["ci_low"],
            "ci_high": prior["ci_high"],
            "n_matches": prior["n_matches"],
            "source": "exp_final_construct_v3 frozen JSON; historic model-based CI",
        }
    ]
    for spec, label in [
        ("old_Iron_Gold_matches_corrected", "B old Iron–Gold, corrected method"),
        (
            "old_all_timeline_matches_corrected",
            "B2 old all-tier coverage, corrected method",
        ),
    ]:
        r = result[result.specification.eq(spec)].iloc[0]
        comparison.append(
            {
                "stage": label,
                "beta": r.beta,
                "or": r.odds_ratio,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "n_matches": int(r.n_matches),
                "source": "corrected transform fit on old coverage; old Iron–Gold difference SD; match-score HC0 CI",
            }
        )
    cr = result[result.specification.eq("all_available_common_old_scale")].iloc[0]
    comparison.append(
        {
            "stage": "C full seven tiers, fixed old reference scale",
            "beta": cr.beta,
            "or": cr.odds_ratio,
            "ci_low": cr.ci_low,
            "ci_high": cr.ci_high,
            "n_matches": int(cr.n_matches),
            "source": "same score transformation and SD as B, isolating changed sample composition",
        }
    )
    r = (
        pd.read_csv(
            TABLES / "primary_prospective_region_slopes.csv", keep_default_na=False
        )
        .query("region == 'pooled'")
        .iloc[0]
    )
    comparison.append(
        {
            "stage": "D primary seven tiers, new full-sample SD",
            "beta": r.beta,
            "or": r.odds_ratio,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "n_matches": int(r.n_matches),
            "source": "current primary with full-sample scale; stratified match-score bootstrap",
        }
    )
    pd.DataFrame(comparison).to_csv(
        TABLES / "old_new_model_comparison.csv", index=False
    )
    log("Sensitivity models finished")


if __name__ == "__main__":
    main()
