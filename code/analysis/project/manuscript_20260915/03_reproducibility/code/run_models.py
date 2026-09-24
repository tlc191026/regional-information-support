from settings import *
from statistics_core import *
import gc


def regional_profiles(df):
    n = len(df)
    cats = ["region", "tier", "patch", "team_side"]
    team = pd.concat(
        [df[["region", "tier", "patch"]].assign(team_side=s) for s in ["blue", "red"]],
        ignore_index=True,
    )
    y = np.r_[df.blue_maintenance, df.red_maintenance]
    design = Design(cats, []).fit(team)
    x = design.transform(team)
    bread = np.linalg.pinv((x.T @ x).toarray(), rcond=1e-12)
    beta = bread @ np.asarray(x.T @ y).ravel()
    residual = y - x @ beta
    freq = team.tier.value_counts()
    weights = team.tier.map(lambda t: 1 / freq[t]).to_numpy(float)
    common = np.asarray(x.T @ weights).ravel() / weights.sum()
    vectors = {}
    means = []
    psis = []
    for region in REGIONS:
        row = common.copy()
        for j, name in enumerate(design.names):
            if name.startswith("region_"):
                row[j] = float(name == "region_" + region)
        vectors[region] = row
        influence = np.asarray(x @ (bread @ row)) * residual
        influence = influence[:n] + influence[n:]
        value = float(row @ beta)
        se = float(np.sqrt(influence @ influence))
        means.append(
            {
                "region": region,
                "display_region": LABELS[region],
                "adjusted_mean": value,
                "ci_low": value - 1.96 * se,
                "ci_high": value + 1.96 * se,
                "n_matches": int(df.region.eq(region).sum()),
                "n_teams": int(2 * df.region.eq(region).sum()),
                "standardization": "equal sampling-tier distribution; shared patch mix; equal blue/red sides",
                "interval_method": "match-score HC0 Wald",
            }
        )
        psis.append(influence)
    effects = []
    for a, b in [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]:
        c = vectors[a] - vectors[b]
        delta = float(c @ beta)
        influence = np.asarray(x @ (bread @ c)) * residual
        influence = influence[:n] + influence[n:]
        selected = df.region.isin([a, b]).to_numpy()
        nt = 2 * selected.sum()
        rss = residual[:n] ** 2 + residual[n:] ** 2
        variance = float(rss[selected].sum() / nt)
        sd = np.sqrt(variance)
        vp = np.where(selected, (rss - 2 * variance) / nt, 0)
        d = delta / sd
        dpsi = influence / sd - delta / (2 * sd**3) * vp
        se = float(np.sqrt(dpsi @ dpsi))
        effects.append(
            {
                "contrast": LABELS[a] + "-" + LABELS[b],
                "adjusted_mean_difference": delta,
                "d": d,
                "ci_low": d - 1.96 * se,
                "ci_high": d + 1.96 * se,
                "standardizer": "pooled within-region model residual SD",
                "pooled_residual_sd": sd,
                "interval_method": "match-score HC0 Wald",
            }
        )
        psis.append(dpsi)
    return pd.DataFrame(means), pd.DataFrame(effects), np.column_stack(psis)


def load_scored(window="0_10", scaler=None):
    df = (
        pd.read_parquet(DATA / f"timeline_features_{window}.parquet")
        .sort_values("match_id")
        .reset_index(drop=True)
    )
    df = add_differences(df)
    if scaler is None:
        scaler = MaintenanceScale().fit(df)
    return pd.concat([df, scaler.apply(df)], axis=1), scaler


def clean_for(df, outcome="first_post_neutral_blue", extra=[]):
    return df.dropna(subset=[outcome, "maintenance_diff_z", *STATE, *extra])


def main():
    df, scale = load_scored()
    n = len(df)
    write_json(DATA / "maintenance_scale_primary.json", scale.metadata())
    log(f"Primary dataset {n:,} eligible unique matches")
    means, effects, reg_psi = regional_profiles(df)
    all_psi = [reg_psi]
    points = [*means.adjusted_mean, *effects.d]
    labels = [*("mean_" + r for r in REGIONS), *("d_" + c for c in effects.contrast)]
    primary = []
    prows = []
    start_indices = {}
    for r in ["pooled", *REGIONS]:
        work = clean_for(df if r == "pooled" else df[df.region.eq(r)])
        fit = binary_fit(
            work,
            cats=["region", "tier", "patch"] if r == "pooled" else ["tier", "patch"],
        )
        e = fit["effects"][0]
        e.update(region=r, display_region=LABELS.get(r, "Pooled"))
        primary.append(e)
        psi = np.zeros((n, 1))
        psi[work.index] = fit["psi"]
        start_indices[r] = len(points)
        all_psi.append(psi)
        points.append(e["beta"])
        labels.append("beta_" + r)
        if r == "pooled":
            prows, ppsi = probability_effects(fit, work)
            prob_indices = list(range(len(points), len(points) + 3))
            padded = np.zeros((n, 3))
            padded[work.index] = ppsi
            all_psi.append(padded)
            points.extend(p["probability"] for p in prows)
            labels.extend("prob_" + str(v) for v in [-1, 0, 1])
        log(
            f'Primary {r}: beta={e["beta"]:.6f}, OR={e["odds_ratio"]:.6f}, n={len(work):,}'
        )
        del fit, work
        gc.collect()
    influence = np.column_stack(all_psi)
    del all_psi, reg_psi
    strata = df.region.astype(str) + "|" + df.tier.astype(str)
    boot = match_bootstrap(
        influence, strata.to_numpy(), "primary_match", B=1000
    ) + np.asarray(points)
    pd.DataFrame(boot, columns=labels).to_csv(
        DATA / "primary_match_bootstrap_replicates.csv", index=False
    )
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    means["ci_low"] = lo[:3]
    means["ci_high"] = hi[:3]
    effects["ci_low"] = lo[3:6]
    effects["ci_high"] = hi[3:6]
    effects["p_value"] = 2 * stats.norm.sf(
        np.abs(effects.d.to_numpy() / boot[:, 3:6].std(axis=0, ddof=1))
    )
    effects["p_holm_three_regional_contrasts"] = holm_p(effects.p_value)
    method = "1,000 region-by-tier stratified match pairs, one-step score bootstrap"
    means["interval_method"] = method
    effects["interval_method"] = method
    means["interval_coverage"] = "pointwise 95%"
    effects["interval_coverage"] = (
        "pointwise 95%; three contrast p-values adjusted by Holm"
    )
    for e in primary:
        j = start_indices[e["region"]]
        e.update(
            ci_low=float(lo[j]),
            ci_high=float(hi[j]),
            or_low=float(np.exp(lo[j])),
            or_high=float(np.exp(hi[j])),
            interval_method=method,
        )
    for row, j in zip(prows, prob_indices):
        row.update(ci_low=float(lo[j]), ci_high=float(hi[j]), interval_method=method)
    pd.DataFrame(primary).to_csv(
        TABLES / "primary_prospective_region_slopes.csv", index=False
    )
    means.to_csv(TABLES / "early_adjusted_region_means.csv", index=False)
    effects.to_csv(TABLES / "early_region_contrasts.csv", index=False)
    pd.DataFrame(prows).to_csv(TABLES / "absolute_probabilities.csv", index=False)
    change = boot[:, prob_indices[2]] - boot[:, prob_indices[0]]
    write_json(
        TABLES / "absolute_probability_change.json",
        {
            "comparison": "+1 SD minus -1 SD",
            "difference": points[prob_indices[2]] - points[prob_indices[0]],
            "ci_low": float(np.quantile(change, 0.025)),
            "ci_high": float(np.quantile(change, 0.975)),
            "interval_method": method,
        },
    )
    del influence, boot
    gc.collect()
    # Region interaction on a common global maintenance SD and common controls.
    work = clean_for(df).copy()
    for r in ["EUW1", "KR"]:
        work["maintenance_" + r] = work.maintenance_diff_z * work.region.eq(r)
    fit = binary_fit(
        work, predictors=["maintenance_diff_z", "maintenance_EUW1", "maintenance_KR"]
    )
    b = np.array([e["beta"] for e in fit["effects"][1:]])
    cov = fit["psi"][:, 1:].T @ fit["psi"][:, 1:]
    wald = float(b @ np.linalg.pinv(cov) @ b)
    write_json(
        TABLES / "region_functional_heterogeneity.json",
        {
            "wald_chi2": wald,
            "df": 2,
            "p_value": float(stats.chi2.sf(wald, 2)),
            "n_matches": len(work),
            "interpretation": "association heterogeneity; common global score SD; match-score sandwich",
        },
    )
    del work, fit
    gc.collect()
    # Main-family secondary outcomes; inference is match-level robust Wald.
    outcomes = []
    for outcome, kind in [
        ("blue_win", "binary"),
        ("neutral_objectives_post_diff", "linear"),
        ("building_objectives_post_diff", "linear"),
        ("kills_post_diff", "linear"),
        ("gold_growth_post_diff", "linear"),
        ("damage_growth_post_diff", "linear"),
    ]:
        work = clean_for(df, outcome)
        if kind == "binary":
            fit = binary_fit(work, outcome=outcome)
            e = fit["effects"][0]
            del fit
        else:
            e = linear_fit(work, outcome)[0]
        outcomes.append(e)
        log("Secondary " + outcome + ": " + str(e["beta"]))
    out = pd.DataFrame(outcomes)
    out["q_value_secondary_family"] = bh_q(out.p_value)
    out.to_csv(TABLES / "secondary_outcome_models.csv", index=False)
    # Scores, not raw restricted identifiers, needed by later validation scripts.
    df.to_parquet(
        DATA / "analysis_scored_0_10.parquet", index=False, compression="zstd"
    )
    log("Primary models finished")


if __name__ == "__main__":
    main()
