"""Prospective timeline validation with a non-duplicated infrastructure score."""

from __future__ import annotations

import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from common import (
    DIRS,
    ROOT,
    bootstrap_d,
    child_seed,
    cohen_d,
    control_design,
    ensure_dirs,
    fit_sparse_linear,
    fit_sparse_logistic,
    one_hot,
    rng,
    save_figure,
    set_figure_style,
    transform_metric,
    write_json,
    zscore,
)
from constructs import (
    EARLY_INFRA_PRIMARY,
    EARLY_INFRA_SENSITIVITIES,
    MASTER_SEED,
    REGION_COLORS,
    REGION_LABELS,
    REGION_ORDER,
    TIER_ORDER,
)

PRIOR_DATA = ROOT / "outputs" / "study1_information_support_validation" / "data"
DIFF_PATH = PRIOR_DATA / "study1_match_difference_table_scored.parquet"
ROLE_PATH = PRIOR_DATA / "study1_role_origin_scores.parquet"
WARD_AUDIT_PATH = PRIOR_DATA / "study1_ward_type_audit_events.parquet"
CANONICAL_META = DIRS["data"] / "canonical_unique_match_metadata.parquet"

COMMON_TIERS = ["IRON", "BRONZE", "SILVER", "GOLD"]
HIGH_TIERS = ["PLATINUM", "EMERALD", "DIAMOND"]
STATE_CONTROLS = [
    "gold_at_window_diff",
    "xp_at_window_diff",
    "level_at_window_diff",
    "kills_pre_diff",
    "neutral_objectives_pre_diff",
    "turret_plates_pre_diff",
]


def add_nonduplicated_scores(diff: pd.DataFrame) -> pd.DataFrame:
    out = diff.copy()
    out["blue_noncontrol_valid_wards_placed"] = (
        out["blue_valid_wards_placed"] - out["blue_control_wards_placed"]
    ).clip(lower=0)
    out["red_noncontrol_valid_wards_placed"] = (
        out["red_valid_wards_placed"] - out["red_control_wards_placed"]
    ).clip(lower=0)
    for window in out["window"].dropna().unique():
        mask = out["window"].eq(window)
        n = int(mask.sum())
        for score_name, components in EARLY_INFRA_SENSITIVITIES.items():
            blue_z, red_z = [], []
            for component in components:
                blue = out.loc[mask, f"blue_{component}"].fillna(0).to_numpy()
                red = out.loc[mask, f"red_{component}"].fillna(0).to_numpy()
                transformed, _, _ = transform_metric(np.concatenate([blue, red]))
                standardized = zscore(transformed)
                blue_z.append(standardized[:n])
                red_z.append(standardized[n:])
            blue_score = zscore(np.vstack(blue_z).mean(axis=0))
            red_score = zscore(np.vstack(red_z).mean(axis=0))
            out.loc[mask, f"blue_{score_name}"] = blue_score
            out.loc[mask, f"red_{score_name}"] = red_score
            out.loc[mask, f"{score_name}_diff"] = blue_score - red_score
    out.to_parquet(
        DIRS["data"] / "timeline_match_differences_nonduplicated_scores.parquet",
        index=False,
    )
    return out


def build_team_long(diff: pd.DataFrame) -> pd.DataFrame:
    work = diff[diff["window"].eq("0_10")].copy()
    rows = []
    for side, prefix, team_id in [("Blue", "blue", 100), ("Red", "red", 200)]:
        frame = work[["match_id", "region", "tier", "patch", "game_duration"]].copy()
        frame["team_side"] = side
        frame["team_id"] = team_id
        frame["team_win"] = (
            work["blue_win"].to_numpy()
            if side == "Blue"
            else 1 - work["blue_win"].to_numpy()
        )
        for score in EARLY_INFRA_SENSITIVITIES:
            frame[score] = work[f"{prefix}_{score}"].to_numpy()
        rows.append(frame)
    team = pd.concat(rows, ignore_index=True)
    team.to_parquet(
        DIRS["data"] / "timeline_match_team_early_infrastructure.parquet", index=False
    )
    return team


def adjusted_region_investment(team: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = (
        team[team["tier"].isin(COMMON_TIERS)]
        .dropna(subset=["primary_three_component"])
        .copy()
    )
    x, _ = control_design(work, ["tier", "patch", "team_side"], ["game_duration"])
    y = work["primary_three_component"].to_numpy(dtype=float)
    model = Ridge(alpha=1e-8, solver="lsqr", fit_intercept=False).fit(x, y)
    work["adjusted_infrastructure"] = y - model.predict(x)
    summary = (
        work.groupby("region", observed=True)["adjusted_infrastructure"]
        .agg(n="size", mean="mean", sd="std")
        .reset_index()
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci_high"] = summary["mean"] + 1.96 * summary["se"]
    effects = []
    for a, b in [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]:
        d, low, high = bootstrap_d(
            work.loc[work.region.eq(a), "adjusted_infrastructure"],
            work.loc[work.region.eq(b), "adjusted_infrastructure"],
            f"timeline_region_investment_{a}_{b}",
            1000,
        )
        effects.append(
            {
                "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                "region_a": a,
                "region_b": b,
                "cohens_d": d,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_n": 1000,
            }
        )
    effects = pd.DataFrame(effects)
    summary.to_csv(
        DIRS["tables"] / "timeline_early_infrastructure_adjusted_region_means.csv",
        index=False,
    )
    effects.to_csv(
        DIRS["tables"] / "timeline_early_infrastructure_region_effects.csv", index=False
    )
    summary.to_csv(
        DIRS["source_data"] / "figure3a_adjusted_region_means.csv", index=False
    )
    effects.to_csv(DIRS["source_data"] / "figure3a_region_effects.csv", index=False)
    return summary, effects


def prepare_primary(
    diff: pd.DataFrame,
    tiers=COMMON_TIERS,
    window="0_10",
    score="primary_three_component",
) -> pd.DataFrame:
    needed = [
        "match_id",
        "region",
        "tier",
        "patch",
        "game_duration",
        "blue_win",
        "first_post_neutral_blue",
        "post10_neutral_objective_diff",
        "building_objectives_post_diff",
        "post_window_kill_diff",
        "gold_growth_post_diff",
        "damage_to_champions_growth_post_diff",
        *STATE_CONTROLS,
        "early_assists_pre_diff",
        f"{score}_diff",
    ]
    work = (
        diff[diff["window"].eq(window) & diff["tier"].isin(tiers)][needed]
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )
    return work.rename(columns={f"{score}_diff": "infrastructure_diff"})


def design_with_interest(work: pd.DataFrame, include_execution=False):
    numeric = STATE_CONTROLS + (["early_assists_pre_diff"] if include_execution else [])
    x_control, _ = control_design(work, ["region", "tier", "patch"], numeric)
    infra = zscore(work["infrastructure_diff"]).reshape(-1, 1)
    x = sparse.hstack([x_control, sparse.csr_matrix(infra)], format="csr")
    return x, x.shape[1] - 1


def prospective_models(
    diff: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = prepare_primary(diff)
    outcomes = [
        (
            "first_post_neutral_blue",
            "First post-10 neutral objective",
            "binary",
            "Team-control outcome",
        ),
        (
            "post10_neutral_objective_diff",
            "Post-10 neutral objective differential",
            "linear",
            "Team-control outcome",
        ),
        (
            "building_objectives_post_diff",
            "Post-10 building control",
            "linear",
            "Team-control outcome",
        ),
        ("blue_win", "Final win", "binary", "Distal team success"),
        (
            "post_window_kill_diff",
            "Post-window kills",
            "linear",
            "Combat/economy outcome",
        ),
        (
            "gold_growth_post_diff",
            "Post-window gold growth",
            "linear",
            "Combat/economy outcome",
        ),
        (
            "damage_to_champions_growth_post_diff",
            "Post-window damage growth",
            "linear",
            "Combat/economy outcome",
        ),
    ]
    rows = []
    probability = []
    for outcome, label, kind, domain in outcomes:
        work = primary.dropna(
            subset=[
                outcome,
                "infrastructure_diff",
                *STATE_CONTROLS,
                "region",
                "tier",
                "patch",
            ]
        ).copy()
        x_context, _ = control_design(work, ["region", "tier", "patch"], STATE_CONTROLS)
        infrastructure_z = zscore(work["infrastructure_diff"])
        outcome_values = work[outcome].to_numpy(dtype=float)
        infra_fit = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False).fit(
            x_context, infrastructure_z
        )
        outcome_fit = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False).fit(
            x_context, outcome_values
        )
        infra_resid = infrastructure_z - infra_fit.predict(x_context)
        outcome_resid = outcome_values - outcome_fit.predict(x_context)
        partial_r = float(np.corrcoef(infra_resid, outcome_resid)[0, 1])
        fisher_z = np.arctanh(np.clip(partial_r, -0.999999, 0.999999))
        fisher_se = 1 / math.sqrt(max(len(work) - x_context.shape[1] - 3, 1))
        partial_low = float(np.tanh(fisher_z - 1.96 * fisher_se))
        partial_high = float(np.tanh(fisher_z + 1.96 * fisher_se))
        x, idx = design_with_interest(work)
        if kind == "binary":
            fit = fit_sparse_logistic(x, work[outcome].astype(int).to_numpy(), [idx])
            beta, se, low, high, odds, ame = fit["effects"][0]
            x_minus = x.copy().tolil()
            x_minus[:, idx] = -1.0
            x_minus = x_minus.tocsr()
            x_zero = x.copy().tolil()
            x_zero[:, idx] = 0.0
            x_zero = x_zero.tocsr()
            x_plus = x.copy().tolil()
            x_plus[:, idx] = 1.0
            x_plus = x_plus.tocsr()
            p_minus = float(fit["model"].predict_proba(x_minus)[:, 1].mean())
            p_zero = float(fit["model"].predict_proba(x_zero)[:, 1].mean())
            p_plus = float(fit["model"].predict_proba(x_plus)[:, 1].mean())
            probability.extend(
                [
                    {
                        "outcome": outcome,
                        "infrastructure_z": -1,
                        "predicted_probability": p_minus,
                    },
                    {
                        "outcome": outcome,
                        "infrastructure_z": 0,
                        "predicted_probability": p_zero,
                    },
                    {
                        "outcome": outcome,
                        "infrastructure_z": 1,
                        "predicted_probability": p_plus,
                    },
                ]
            )
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": label,
                    "outcome_domain": domain,
                    "model": "logistic",
                    "beta": beta,
                    "se": se,
                    "ci_low": low,
                    "ci_high": high,
                    "odds_ratio": odds,
                    "average_marginal_effect": ame,
                    "partial_r": partial_r,
                    "partial_r_ci_low": partial_low,
                    "partial_r_ci_high": partial_high,
                    "probability_minus_1sd": p_minus,
                    "probability_zero": p_zero,
                    "probability_plus_1sd": p_plus,
                    "n_matches": len(work),
                }
            )
        else:
            fit = fit_sparse_linear(x, zscore(work[outcome]), [idx])
            beta, se, low, high = fit["effects"][0]
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": label,
                    "outcome_domain": domain,
                    "model": "linear_standardized",
                    "beta": beta,
                    "se": se,
                    "ci_low": low,
                    "ci_high": high,
                    "odds_ratio": np.nan,
                    "average_marginal_effect": np.nan,
                    "partial_r": partial_r,
                    "partial_r_ci_low": partial_low,
                    "partial_r_ci_high": partial_high,
                    "probability_minus_1sd": np.nan,
                    "probability_zero": np.nan,
                    "probability_plus_1sd": np.nan,
                    "n_matches": len(work),
                    "r2": fit["r2"],
                }
            )
    result = pd.DataFrame(rows)
    prob = pd.DataFrame(probability)
    result.to_csv(
        DIRS["tables"] / "timeline_prospective_outcome_models.csv", index=False
    )
    result.to_csv(
        DIRS["source_data"] / "figure3c_context_adjusted_partial_correlations.csv",
        index=False,
    )
    prob.to_csv(
        DIRS["tables"] / "timeline_absolute_probability_changes.csv", index=False
    )

    # Region-specific slopes use the same model within each ecology. The pooled
    # interaction test below is the formal test of slope heterogeneity.
    slope_rows = []
    outcome = "first_post_neutral_blue"
    for region in ["pooled", *REGION_ORDER]:
        work = primary if region == "pooled" else primary[primary.region.eq(region)]
        work = work.dropna(
            subset=[outcome, "infrastructure_diff", *STATE_CONTROLS, "tier", "patch"]
        ).copy()
        cats = ["region", "tier", "patch"] if region == "pooled" else ["tier", "patch"]
        x_ctrl, _ = control_design(work, cats, STATE_CONTROLS)
        infra = zscore(work["infrastructure_diff"]).reshape(-1, 1)
        x = sparse.hstack([x_ctrl, sparse.csr_matrix(infra)], format="csr")
        idx = x.shape[1] - 1
        fit = fit_sparse_logistic(x, work[outcome].astype(int).to_numpy(), [idx])
        beta, se, low, high, odds, ame = fit["effects"][0]
        slope_rows.append(
            {
                "region": region,
                "display_region": (
                    "Pooled" if region == "pooled" else REGION_LABELS[region]
                ),
                "beta": beta,
                "se": se,
                "ci_low": low,
                "ci_high": high,
                "odds_ratio": odds,
                "average_marginal_effect": ame,
                "base_rate": work[outcome].mean(),
                "n_matches": len(work),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(
        DIRS["tables"] / "timeline_region_specific_functional_slopes.csv", index=False
    )
    slopes.to_csv(
        DIRS["source_data"] / "figure3b_region_specific_slopes.csv", index=False
    )

    interaction_work = primary.dropna(
        subset=[
            outcome,
            "infrastructure_diff",
            *STATE_CONTROLS,
            "region",
            "tier",
            "patch",
        ]
    ).copy()
    x_base_ctrl, _ = control_design(
        interaction_work, ["region", "tier", "patch"], STATE_CONTROLS
    )
    infra = zscore(interaction_work["infrastructure_diff"])
    reg_euw = interaction_work["region"].eq("EUW1").to_numpy(dtype=float)
    reg_kr = interaction_work["region"].eq("KR").to_numpy(dtype=float)
    x_base = sparse.hstack(
        [x_base_ctrl, sparse.csr_matrix(infra[:, None])], format="csr"
    )
    x_full = sparse.hstack(
        [x_base, sparse.csr_matrix(np.column_stack([infra * reg_euw, infra * reg_kr]))],
        format="csr",
    )
    y = interaction_work[outcome].astype(int).to_numpy()
    m0 = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=500,
        fit_intercept=False,
        random_state=MASTER_SEED,
    ).fit(x_base, y)
    m1 = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=500,
        fit_intercept=False,
        random_state=MASTER_SEED,
    ).fit(x_full, y)
    ll0 = -log_loss(y, m0.predict_proba(x_base)[:, 1], normalize=False)
    ll1 = -log_loss(y, m1.predict_proba(x_full)[:, 1], normalize=False)
    lr = 2 * (ll1 - ll0)
    interaction = pd.DataFrame(
        [
            {
                "term": "region_by_infrastructure",
                "likelihood_ratio_chi2": lr,
                "df": 2,
                "p_value": stats.chi2.sf(lr, 2),
                "delta_log_loss": (ll1 - ll0) / len(y),
                "n_matches": len(y),
            }
        ]
    )
    interaction.to_csv(
        DIRS["tables"] / "timeline_region_by_infrastructure_interaction.csv",
        index=False,
    )
    return result, slopes, interaction


def nested_prediction(diff: pd.DataFrame) -> pd.DataFrame:
    work = (
        prepare_primary(diff)
        .dropna(
            subset=[
                "first_post_neutral_blue",
                "infrastructure_diff",
                *STATE_CONTROLS,
                "early_assists_pre_diff",
                "region",
                "tier",
                "patch",
            ]
        )
        .copy()
    )
    strat = (
        work["region"].astype(str)
        + "_"
        + work["tier"].astype(str)
        + "_"
        + work["first_post_neutral_blue"].astype(int).astype(str)
    )
    train_idx, test_idx = train_test_split(
        np.arange(len(work)), test_size=0.30, random_state=MASTER_SEED, stratify=strat
    )
    train, test = work.iloc[train_idx], work.iloc[test_idx]
    blocks = [
        ("M0 task context", []),
        (
            "M1 early economy",
            ["gold_at_window_diff", "xp_at_window_diff", "level_at_window_diff"],
        ),
        ("M2 early combat/objectives", STATE_CONTROLS),
        ("M3 joint-action proxy", [*STATE_CONTROLS, "early_assists_pre_diff"]),
        (
            "M4 shared-information maintenance",
            [*STATE_CONTROLS, "early_assists_pre_diff", "infrastructure_diff"],
        ),
    ]
    rows, previous = [], None
    for label, nums in blocks:
        x_train, encoder = control_design(train, ["region", "tier", "patch"], nums)
        x_test, _ = control_design(test, ["region", "tier", "patch"], nums, fit=encoder)
        model = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=500,
            fit_intercept=False,
            random_state=MASTER_SEED,
        ).fit(x_train, train["first_post_neutral_blue"].astype(int))
        pred = model.predict_proba(x_test)[:, 1]
        row = {
            "model": label,
            "n_train": len(train),
            "n_test": len(test),
            "log_loss": log_loss(test["first_post_neutral_blue"], pred),
            "auc": roc_auc_score(test["first_post_neutral_blue"], pred),
            "brier": brier_score_loss(test["first_post_neutral_blue"], pred),
        }
        if previous:
            row.update(
                {
                    "delta_log_loss_improvement": previous["log_loss"]
                    - row["log_loss"],
                    "delta_auc": row["auc"] - previous["auc"],
                    "delta_brier_improvement": previous["brier"] - row["brier"],
                }
            )
        else:
            row.update(
                {
                    "delta_log_loss_improvement": np.nan,
                    "delta_auc": np.nan,
                    "delta_brier_improvement": np.nan,
                }
            )
        rows.append(row)
        previous = row
    result = pd.DataFrame(rows)
    result.to_csv(
        DIRS["tables"] / "timeline_nested_out_of_sample_prediction.csv", index=False
    )
    result.to_csv(DIRS["source_data"] / "figure3d_nested_prediction.csv", index=False)
    return result


def coverage_weights(diff: pd.DataFrame) -> pd.DataFrame | None:
    if not CANONICAL_META.exists():
        return None
    meta = pd.read_parquet(CANONICAL_META)
    included = set(diff.loc[diff.window.eq("0_10"), "match_id"].astype(str))
    meta["timeline_included"] = meta["match_id"].astype(str).isin(included).astype(int)
    work = meta.dropna(
        subset=["target_region", "target_tier", "target_patch", "target_duration"]
    ).copy()
    x, _ = control_design(
        work, ["target_region", "target_tier", "target_patch"], ["target_duration"]
    )
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=500,
        fit_intercept=False,
        random_state=MASTER_SEED,
    ).fit(x, work["timeline_included"])
    p = np.clip(model.predict_proba(x)[:, 1], 0.01, 0.99)
    work["coverage_probability"] = p
    included_meta = work[work.timeline_included.eq(1)][
        ["match_id", "coverage_probability"]
    ].copy()
    included_meta["ipw"] = 1 / included_meta["coverage_probability"]
    lo, hi = included_meta["ipw"].quantile([0.01, 0.99])
    included_meta["ipw"] = included_meta["ipw"].clip(lo, hi)
    included_meta.to_parquet(
        DIRS["data"] / "timeline_coverage_ipw.parquet", index=False
    )
    audit = (
        work.groupby(["target_region", "target_tier"], observed=True)[
            "timeline_included"
        ]
        .agg(n="size", coverage_rate="mean")
        .reset_index()
    )
    audit.to_csv(
        DIRS["data_audit"] / "timeline_coverage_by_region_tier.csv", index=False
    )
    return included_meta


def sensitivity_models(diff: pd.DataFrame, ipw: pd.DataFrame | None) -> pd.DataFrame:
    specs = [
        (
            "primary_common_support",
            COMMON_TIERS,
            "0_10",
            "primary_three_component",
            None,
        ),
        ("minimal_two_component", COMMON_TIERS, "0_10", "minimal_two_component", None),
        ("deployment_only", COMMON_TIERS, "0_10", "deployment_only", None),
        (
            "disjoint_four_component",
            COMMON_TIERS,
            "0_10",
            "disjoint_four_component",
            None,
        ),
        ("window_0_15", COMMON_TIERS, "0_15", "primary_three_component", None),
        ("high_tier_exploratory", HIGH_TIERS, "0_10", "primary_three_component", None),
    ]
    rows = []
    for name, tiers, window, score, weight_name in specs:
        work = (
            prepare_primary(diff, tiers, window, score)
            .dropna(
                subset=[
                    "first_post_neutral_blue",
                    "infrastructure_diff",
                    *STATE_CONTROLS,
                    "region",
                    "tier",
                    "patch",
                ]
            )
            .copy()
        )
        x, idx = design_with_interest(work)
        fit = fit_sparse_logistic(x, work["first_post_neutral_blue"].astype(int), [idx])
        beta, se, low, high, odds, ame = fit["effects"][0]
        rows.append(
            {
                "sensitivity": name,
                "beta": beta,
                "ci_low": low,
                "ci_high": high,
                "odds_ratio": odds,
                "average_marginal_effect": ame,
                "n_matches": len(work),
                "seed": MASTER_SEED,
            }
        )

    if ipw is not None:
        work = (
            prepare_primary(diff)
            .merge(ipw, on="match_id", how="inner")
            .dropna(
                subset=[
                    "first_post_neutral_blue",
                    "infrastructure_diff",
                    *STATE_CONTROLS,
                    "region",
                    "tier",
                    "patch",
                    "ipw",
                ]
            )
        )
        x, idx = design_with_interest(work)
        fit = fit_sparse_logistic(
            x,
            work["first_post_neutral_blue"].astype(int),
            [idx],
            sample_weight=work["ipw"].to_numpy(),
        )
        beta, se, low, high, odds, ame = fit["effects"][0]
        rows.append(
            {
                "sensitivity": "inverse_probability_of_coverage_weighted",
                "beta": beta,
                "ci_low": low,
                "ci_high": high,
                "odds_ratio": odds,
                "average_marginal_effect": ame,
                "n_matches": len(work),
                "seed": MASTER_SEED,
            }
        )

    base = (
        prepare_primary(diff)
        .dropna(
            subset=[
                "first_post_neutral_blue",
                "infrastructure_diff",
                *STATE_CONTROLS,
                "region",
                "tier",
                "patch",
            ]
        )
        .copy()
    )
    minimum = int(base.groupby(["region", "tier"], observed=True).size().min())
    balanced = base.groupby(["region", "tier"], observed=True, group_keys=False).sample(
        n=minimum, random_state=MASTER_SEED
    )
    x, idx = design_with_interest(balanced)
    fit = fit_sparse_logistic(x, balanced["first_post_neutral_blue"].astype(int), [idx])
    beta, se, low, high, odds, ame = fit["effects"][0]
    rows.append(
        {
            "sensitivity": "region_by_tier_balanced_resampling",
            "beta": beta,
            "ci_low": low,
            "ci_high": high,
            "odds_ratio": odds,
            "average_marginal_effect": ame,
            "n_matches": len(balanced),
            "seed": MASTER_SEED,
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(DIRS["tables"] / "timeline_sensitivity_models.csv", index=False)
    return result


def role_origin_and_comeback(diff: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_result = pd.DataFrame()
    if ROLE_PATH.exists():
        role = pd.read_parquet(ROLE_PATH)
        role["blue_noncontrol_valid_wards_placed"] = (
            role["blue_valid_wards_placed"] - role["blue_control_wards_placed"]
        ).clip(lower=0)
        role["red_noncontrol_valid_wards_placed"] = (
            role["red_valid_wards_placed"] - role["red_control_wards_placed"]
        ).clip(lower=0)
        mask = role.window.eq("0_10")
        n = int(mask.sum())
        blue_z, red_z = [], []
        for component in EARLY_INFRA_PRIMARY:
            values, _, _ = transform_metric(
                np.concatenate(
                    [
                        role.loc[mask, f"blue_{component}"],
                        role.loc[mask, f"red_{component}"],
                    ]
                )
            )
            values = zscore(values)
            blue_z.append(values[:n])
            red_z.append(values[n:])
        role.loc[mask, "role_infrastructure_diff"] = np.vstack(blue_z).mean(
            axis=0
        ) - np.vstack(red_z).mean(axis=0)
        wide = (
            role.loc[mask]
            .pivot_table(
                index="match_id",
                columns="role_group",
                values="role_infrastructure_diff",
                aggfunc="mean",
            )
            .reset_index()
        )
        rename = {
            c: "role_" + str(c).lower().replace(" ", "_")
            for c in wide.columns
            if c != "match_id"
        }
        wide = wide.rename(columns=rename)
        work = prepare_primary(diff).merge(wide, on="match_id", how="left")
        predictors = list(rename.values())
        work = work.dropna(
            subset=[
                "first_post_neutral_blue",
                *predictors,
                *STATE_CONTROLS,
                "region",
                "tier",
                "patch",
            ]
        )
        x_ctrl, _ = control_design(work, ["region", "tier", "patch"], STATE_CONTROLS)
        xp = np.column_stack([zscore(work[p]) for p in predictors])
        x = sparse.hstack([x_ctrl, sparse.csr_matrix(xp)], format="csr")
        idxs = list(range(x.shape[1] - len(predictors), x.shape[1]))
        fit = fit_sparse_logistic(x, work["first_post_neutral_blue"].astype(int), idxs)
        rows = []
        for predictor, effect in zip(predictors, fit["effects"]):
            beta, se, low, high, odds, ame = effect
            rows.append(
                {
                    "role_origin": predictor,
                    "beta": beta,
                    "ci_low": low,
                    "ci_high": high,
                    "odds_ratio": odds,
                    "average_marginal_effect": ame,
                    "n_matches": len(work),
                }
            )
        role_result = pd.DataFrame(rows)
        role_result.to_csv(
            DIRS["extended_data"] / "timeline_role_origin_validity.csv", index=False
        )

    work = prepare_primary(diff)
    work = (
        work[work.gold_at_window_diff.abs().ge(1500)]
        .dropna(
            subset=[
                "blue_win",
                "infrastructure_diff",
                *STATE_CONTROLS,
                "region",
                "tier",
                "patch",
            ]
        )
        .copy()
    )
    work["behind_blue"] = work.gold_at_window_diff.lt(0)
    work["behind_wins"] = np.where(work.behind_blue, work.blue_win, 1 - work.blue_win)
    work["behind_infra_advantage"] = np.where(
        work.behind_blue, work.infrastructure_diff, -work.infrastructure_diff
    )
    work["gold_deficit"] = work.gold_at_window_diff.abs()
    work["xp_deficit"] = work.xp_at_window_diff.abs()
    work["kill_deficit"] = work.kills_pre_diff.abs()
    work["objective_deficit"] = work.neutral_objectives_pre_diff.abs()
    x_ctrl, _ = control_design(
        work,
        ["region", "tier", "patch"],
        ["gold_deficit", "xp_deficit", "kill_deficit", "objective_deficit"],
    )
    infra = zscore(work.behind_infra_advantage).reshape(-1, 1)
    x = sparse.hstack([x_ctrl, sparse.csr_matrix(infra)], format="csr")
    idx = x.shape[1] - 1
    fit = fit_sparse_logistic(x, work.behind_wins.astype(int), [idx])
    beta, se, low, high, odds, ame = fit["effects"][0]
    comeback = pd.DataFrame(
        [
            {
                "gold_deficit_threshold": 1500,
                "beta": beta,
                "ci_low": low,
                "ci_high": high,
                "odds_ratio": odds,
                "average_marginal_effect": ame,
                "n_matches": len(work),
                "interpretation": "conditional exploratory association",
            }
        ]
    )
    comeback.to_csv(
        DIRS["extended_data"] / "timeline_comeback_analysis.csv", index=False
    )
    return role_result, comeback


def ward_audit():
    if not WARD_AUDIT_PATH.exists():
        return pd.DataFrame()
    audit = pd.read_parquet(WARD_AUDIT_PATH)
    summary = (
        audit.groupby(["window", "event_type", "ward_type"], observed=True)["count"]
        .sum()
        .reset_index()
    )
    summary["included_primary_vision_event"] = summary.ward_type.isin(
        ["YELLOW_TRINKET", "CONTROL_WARD", "SIGHT_WARD", "BLUE_TRINKET"]
    )
    summary.to_csv(DIRS["data_audit"] / "timeline_ward_type_audit.csv", index=False)
    return summary


def draw_figure3(region_means, effects, slopes, outcomes, nested, sensitivity):
    set_figure_style()
    fig = plt.figure(figsize=(7.2, 5.45))
    gs = fig.add_gridspec(
        2, 2, left=0.10, right=0.985, bottom=0.10, top=0.91, wspace=0.43, hspace=0.50
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    rm = region_means.set_index("region").reindex(REGION_ORDER).reset_index()
    x = np.arange(3)
    ax_a.plot(x, rm["mean"], color="#B9B9B9", lw=1.0, zorder=1)
    for i, row in rm.iterrows():
        ax_a.errorbar(
            i,
            row["mean"],
            yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            fmt="o",
            ms=5.2,
            capsize=2,
            lw=1.0,
            color=REGION_COLORS[row["region"]],
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=3,
        )
    ax_a.axhline(0, color="#8A8A8A", lw=0.65)
    ax_a.set_xticks(x, [REGION_LABELS[r] for r in REGION_ORDER])
    ax_a.set_ylabel("Adjusted early maintenance (z)")
    eff = effects.set_index("contrast")
    ax_a.text(
        0.02,
        0.98,
        f"KR–NA  d = {eff.loc['KR-NA','cohens_d']:.3f}\n"
        f"KR–EUW  d = {eff.loc['KR-EUW','cohens_d']:.3f}\n"
        f"EUW–NA  d = {eff.loc['EUW-NA','cohens_d']:.3f}",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color="#3B3B3B",
    )

    slope_order = ["pooled", "NA1", "EUW1", "KR"]
    ss = slopes.set_index("region").reindex(slope_order).reset_index()
    yy = np.arange(len(ss))[::-1]
    slope_colors = [
        "#303030",
        REGION_COLORS["NA1"],
        REGION_COLORS["EUW1"],
        REGION_COLORS["KR"],
    ]
    for y, row, col in zip(yy, ss.itertuples(), slope_colors):
        ax_b.errorbar(
            row.beta,
            y,
            xerr=[[row.beta - row.ci_low], [row.ci_high - row.beta]],
            fmt="o",
            color=col,
            ecolor=col,
            ms=4.6,
            lw=1.15,
            capsize=2,
            markeredgecolor="white",
            markeredgewidth=0.45,
        )
    ax_b.axvline(0, color="#8A8A8A", lw=0.65)
    ax_b.set_yticks(yy, ["Pooled", "NA", "EUW", "KR"])
    ax_b.set_xlabel("Log-odds coefficient for first post-10 objective")
    ax_b.set_xlim(left=min(0, ss.ci_low.min() - 0.008))

    oo = outcomes.copy()
    oo["y"] = np.arange(len(oo))[::-1]
    colors = oo.outcome_domain.map(
        {
            "Team-control outcome": "#2F7F78",
            "Distal team success": "#6F9F98",
            "Combat/economy outcome": "#999999",
        }
    )
    # Matplotlib expects one RGBA value per errorbar call. Draw the compact
    # forest row-by-row so each outcome family retains its intended colour.
    for row, colour in zip(oo.itertuples(), colors):
        ax_c.errorbar(
            row.partial_r,
            row.y,
            xerr=[
                [row.partial_r - row.partial_r_ci_low],
                [row.partial_r_ci_high - row.partial_r],
            ],
            fmt="o",
            color=colour,
            ecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.2,
            lw=1.2,
            capsize=2,
            zorder=3,
        )
    ax_c.axvline(0, color="#888", lw=0.7)
    ax_c.set_yticks(oo.y, oo.outcome_label)
    ax_c.set_xlabel("Context-adjusted partial r")

    nn = nested.copy()
    xx = np.arange(len(nn))
    ax_d.axvspan(3.55, 4.35, color="#DCEBE8", alpha=0.65, lw=0, zorder=0)
    ax_d.plot(
        xx,
        nn.auc,
        marker="o",
        color="#2F7F78",
        lw=1.55,
        ms=4.1,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=2,
    )
    ax_d.set_xticks(xx, ["M0", "M1", "M2", "M3", "M4"])
    ax_d.set_ylabel("Held-out AUC")
    ax_d.set_xlabel("Nested model sequence")
    if len(nn) >= 5:
        ax_d.annotate(
            f"M4 ΔAUC = {nn.iloc[-1].delta_auc:.4f}",
            (xx[-1], nn.iloc[-1].auc),
            xytext=(-62, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#555"),
            fontsize=6,
        )
    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            -0.14,
            1.045,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
    save_figure(
        fig, DIRS["figures"] / "figure3_early_information_maintenance_and_later_control"
    )


def main():
    ensure_dirs()
    started = time.time()
    if not DIFF_PATH.exists():
        raise FileNotFoundError(DIFF_PATH)
    diff = pd.read_parquet(DIFF_PATH)
    diff = add_nonduplicated_scores(diff)
    team = build_team_long(diff)
    region_means, effects = adjusted_region_investment(team)
    outcomes, slopes, interaction = prospective_models(diff)
    nested = nested_prediction(diff)
    ipw = coverage_weights(diff)
    sensitivity = sensitivity_models(diff, ipw)
    role, comeback = role_origin_and_comeback(diff)
    wards = ward_audit()
    draw_figure3(region_means, effects, slopes, outcomes, nested, sensitivity)
    sample = {
        "all_windows_rows": len(diff),
        "unique_matches": diff.match_id.nunique(),
        "primary_common_support_matches": prepare_primary(diff).match_id.nunique(),
        "master_seed": MASTER_SEED,
    }
    summary = {
        "sample": sample,
        "regional_investment": effects.to_dict("records"),
        "prospective_outcomes": outcomes.to_dict("records"),
        "regional_slopes": slopes.to_dict("records"),
        "interaction": interaction.to_dict("records"),
        "nested_prediction": nested.to_dict("records"),
        "sensitivity": sensitivity.to_dict("records"),
        "comeback": comeback.to_dict("records"),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(DIRS["reports"] / "timeline_machine_readable_summary.json", summary)
    print(
        json.dumps(
            {"status": "complete", "elapsed_minutes": summary["elapsed_minutes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
