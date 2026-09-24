"""Core 2026-07-17 analyses: construct audit, function and regional profiles."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.patches import Rectangle
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    DIRS,
    EXP,
    ROOT,
    bootstrap_d,
    child_seed,
    cohen_d,
    control_design,
    ensure_dirs,
    file_sha256,
    fit_sparse_linear,
    fit_sparse_logistic,
    one_hot,
    residualize_metrics,
    rng,
    runtime_metadata,
    save_figure,
    set_figure_style,
    transform_metric,
    write_json,
    zscore,
)
from constructs import (
    BOUNDARY_FAMILY_ORDER,
    DOMAIN_COLORS,
    DOMAIN_COMPONENTS,
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    INTERDEPENDENCE_ORDER,
    LEGACY_FOUR_CHANNELS,
    MASTER_SEED,
    METRICS,
    METRIC_NAMES,
    PRIMARY_MEASURE_TYPES,
    PRIMARY_BOUNDARY_METRICS,
    REGION_COLORS,
    REGION_LABELS,
    REGION_ORDER,
    REGION_PAIR_COLORS,
    ROLE_INTERDEPENDENCE,
    SUBCOMPONENT_COLORS,
    SUBCOMPONENT_LABELS,
    SUBCOMPONENT_METRICS,
    SUBCOMPONENT_ORDER,
    SIGNAL_SUBTYPE_ORDER,
    SCORE_ORDER,
    TIER_ORDER,
)

CANONICAL = (
    ROOT
    / "outputs"
    / "study0"
    / "data"
    / "study0_canonical_participant_match_sample.pkl"
)
SNAPSHOT_DB = ROOT / "lol_data_snapshot_20260708_1404" / "database" / "lol_data.db"

REGION_PAIRS = [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]
N_PERMUTATIONS = 10_000


def validate_and_load() -> pd.DataFrame:
    if not CANONICAL.exists():
        raise FileNotFoundError(CANONICAL)
    needed = [
        "match_id",
        "puuid",
        "team_id",
        "region",
        "tier",
        "tier_num",
        "team_position",
        "champion_id",
        "patch",
        "game_duration",
        "duration_min",
        "win",
        "baron_kills",
        "dragon_kills",
        "rift_herald_kills",
        "tower_kills",
        "inhibitor_kills",
        "kills",
        "deaths",
        "assists",
        "gold_earned",
        "total_damage_dealt",
        "vision_score",
        "wards_placed",
        "wards_killed",
        "control_wards_bought",
        "all_in_pings",
        "assist_me_pings",
        "command_pings",
        "enemy_missing_pings",
        "enemy_vision_pings",
        "need_vision_pings",
        "on_my_way_pings",
        "push_pings",
        "get_back_pings",
        *METRIC_NAMES,
    ]
    df = pd.read_pickle(CANONICAL)[list(dict.fromkeys(needed))].copy()
    checks = {
        "participant_match_rows": int(len(df)),
        "unique_players": int(df["puuid"].nunique()),
        "unique_matches": int(df["match_id"].nunique()),
        "matches_per_player_min": int(df.groupby("puuid").size().min()),
        "matches_per_player_max": int(df.groupby("puuid").size().max()),
        "metric_count": len(METRIC_NAMES),
        "primary_metric_count": int(
            sum(bool(item["primary_score"]) for item in METRICS)
        ),
        "damage_taken_in_primary_metrics": "total_damage_taken_pm" in METRIC_NAMES,
        "master_seed": MASTER_SEED,
    }
    expected = (126_000, 2_520_000, 2_240_354)
    actual = (
        checks["unique_players"],
        checks["participant_match_rows"],
        checks["unique_matches"],
    )
    if actual != expected:
        raise AssertionError(
            f"Canonical counts changed: expected={expected}, actual={actual}"
        )
    if checks["matches_per_player_min"] != 20 or checks["matches_per_player_max"] != 20:
        raise AssertionError(
            "Canonical sample is not exactly 20 focal matches per player"
        )
    if (
        len(METRIC_NAMES) != 18
        or checks["primary_metric_count"] != 17
        or checks["damage_taken_in_primary_metrics"]
    ):
        raise AssertionError("Primary metric contract failed")
    write_json(DIRS["data_audit"] / "canonical_invariants.json", checks)
    return df


def sample_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap = (
        df.groupby("match_id", observed=True)
        .size()
        .rename("n_focal_players")
        .reset_index()
    )
    dist = (
        overlap["n_focal_players"]
        .value_counts()
        .sort_index()
        .rename_axis("n_focal_players")
        .reset_index(name="n_unique_matches")
    )
    dist["n_focal_rows"] = dist["n_focal_players"] * dist["n_unique_matches"]
    dist.to_csv(DIRS["data_audit"] / "shared_match_multiplicity.csv", index=False)

    regional = []
    for region, part in df.groupby("region", observed=True):
        counts = part.groupby("match_id", observed=True).size()
        regional.append(
            {
                "region": region,
                "display_region": REGION_LABELS.get(region, region),
                "participant_match_rows": int(len(part)),
                "unique_players": int(part["puuid"].nunique()),
                "unique_matches": int(part["match_id"].nunique()),
                "shared_unique_matches": int((counts > 1).sum()),
                "shared_match_fraction": float((counts > 1).mean()),
                "rows_from_shared_matches": int(counts[counts > 1].sum()),
            }
        )
    regional_df = pd.DataFrame(regional)
    regional_df.to_csv(
        DIRS["data_audit"] / "shared_match_overlap_by_region.csv", index=False
    )

    # Deterministic one-focal-per-match selection. The generated random stream is
    # stable because canonical row order is frozen and recorded in the manifest.
    selector = pd.DataFrame(
        {
            "match_id": df["match_id"].to_numpy(),
            "u": rng("one_focal_per_match").random(len(df)),
        }
    )
    one_idx = selector.groupby("match_id", sort=False)["u"].idxmin().to_numpy()
    np.save(DIRS["data"] / "one_focal_per_match_indices.npy", one_idx)
    singleton = (
        df["match_id"]
        .map(overlap.set_index("match_id")["n_focal_players"])
        .eq(1)
        .to_numpy()
    )
    np.save(DIRS["data"] / "singleton_match_mask.npy", singleton)

    summary = {
        "shared_unique_matches": int((overlap["n_focal_players"] > 1).sum()),
        "shared_unique_match_fraction": float((overlap["n_focal_players"] > 1).mean()),
        "rows_from_shared_matches": int(
            overlap.loc[overlap["n_focal_players"] > 1, "n_focal_players"].sum()
        ),
        "row_fraction_from_shared_matches": float(
            overlap.loc[overlap["n_focal_players"] > 1, "n_focal_players"].sum()
            / len(df)
        ),
        "one_focal_rows": int(len(one_idx)),
        "singleton_rows": int(singleton.sum()),
    }
    write_json(DIRS["data_audit"] / "shared_match_summary.json", summary)
    return overlap, regional_df


def build_scores(metric_frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=metric_frame.index)
    for sub in SUBCOMPONENT_ORDER:
        metrics = SUBCOMPONENT_METRICS[sub]
        if not metrics:
            continue
        out[sub] = zscore(metric_frame[metrics].mean(axis=1).to_numpy())
    for domain in DOMAIN_ORDER:
        components = DOMAIN_COMPONENTS[domain]
        out[domain] = zscore(out[components].mean(axis=1).to_numpy())
    return out


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pd.DataFrame(METRICS).to_csv(
        DIRS["tables"] / "construct_dictionary_18_metrics.csv", index=False
    )
    residuals = residualize_metrics(
        df,
        METRICS,
        categorical=["tier", "team_position", "patch"],
        numeric=["game_duration"],
        cache=DIRS["data"] / "participant_metric_residuals_minimal.parquet",
        audit_path=DIRS["tables"] / "metric_preprocessing_audit_minimal.csv",
    )
    residual_scores = build_scores(residuals)
    score_path = DIRS["data"] / "participant_process_scores_minimal.parquet"
    if not score_path.exists():
        pd.concat(
            [
                df[
                    [
                        "match_id",
                        "puuid",
                        "region",
                        "tier",
                        "tier_num",
                        "team_position",
                        "champion_id",
                        "patch",
                        "game_duration",
                        "duration_min",
                        "win",
                    ]
                ].reset_index(drop=True),
                residual_scores.reset_index(drop=True),
            ],
            axis=1,
        ).to_parquet(score_path, index=False)

    transformed_path = DIRS["data"] / "participant_metrics_transformed_z.parquet"
    if transformed_path.exists():
        transformed = pd.read_parquet(transformed_path)
    else:
        transformed = pd.DataFrame(index=df.index)
        rows = []
        for item in METRICS:
            values, qlo, qhi = transform_metric(df[item["metric"]])
            values = zscore(values)
            if item["reverse"]:
                values = -values
            transformed[item["metric"]] = values
            rows.append(
                {
                    "metric": item["metric"],
                    "winsor_log1p_p01": qlo,
                    "winsor_log1p_p99": qhi,
                }
            )
        transformed.to_parquet(transformed_path, index=False)
        pd.DataFrame(rows).to_csv(
            DIRS["tables"] / "metric_transformation_audit_direct_models.csv",
            index=False,
        )
    transformed_scores = build_scores(transformed)
    return residuals, residual_scores, transformed_scores


def boundary_analysis(
    residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dictionary = pd.DataFrame(METRICS).set_index("metric")
    corr = residuals[METRIC_NAMES].corr()
    corr.to_csv(DIRS["tables"] / "residual_correlation_matrix_18_metrics.csv")
    corr.to_csv(DIRS["source_data"] / "figure1a_residual_correlation_matrix.csv")
    rows = []
    for i, a in enumerate(METRIC_NAMES):
        for b in METRIC_NAMES[i + 1 :]:
            rows.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "label_a": dictionary.loc[a, "label"],
                    "label_b": dictionary.loc[b, "label"],
                    "domain_a": dictionary.loc[a, "domain"],
                    "domain_b": dictionary.loc[b, "domain"],
                    "subcomponent_a": dictionary.loc[a, "subcomponent"],
                    "subcomponent_b": dictionary.loc[b, "subcomponent"],
                    "primary_pair": bool(
                        dictionary.loc[a, "primary_boundary"]
                        and dictionary.loc[b, "primary_boundary"]
                    ),
                    "r": float(corr.loc[a, b]),
                    "abs_r": float(abs(corr.loc[a, b])),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(DIRS["tables"] / "residual_metric_pair_correlations.csv", index=False)

    primary = pairs[pairs["primary_pair"]].copy()
    summary = []
    for level, a_col, b_col in [
        ("domain", "domain_a", "domain_b"),
        ("subcomponent", "subcomponent_a", "subcomponent_b"),
    ]:
        same = primary[a_col].eq(primary[b_col])
        for name, mask in [("within", same), ("between", ~same)]:
            vals = primary.loc[mask, "abs_r"]
            summary.append(
                {
                    "boundary_level": level,
                    "pair_type": name,
                    "n_pairs": len(vals),
                    "mean_abs_r": vals.mean(),
                    "median_abs_r": vals.median(),
                }
            )
    # Explicit information hierarchy: persistent, event, and their between-block relation.
    info = primary[
        primary["domain_a"].eq("information_support_practices")
        & primary["domain_b"].eq("information_support_practices")
    ]
    for label, mask in [
        (
            "within_persistent",
            info["subcomponent_a"].eq("persistent_shared_information_maintenance")
            & info["subcomponent_b"].eq("persistent_shared_information_maintenance"),
        ),
        (
            "within_event_signalling",
            info["subcomponent_a"].eq("event_based_informational_signalling")
            & info["subcomponent_b"].eq("event_based_informational_signalling"),
        ),
        (
            "between_information_subcomponents",
            info["subcomponent_a"].ne(info["subcomponent_b"]),
        ),
    ]:
        vals = info.loc[mask, "abs_r"]
        summary.append(
            {
                "boundary_level": "information_hierarchy",
                "pair_type": label,
                "n_pairs": len(vals),
                "mean_abs_r": vals.mean(),
                "median_abs_r": vals.median(),
            }
        )
    vision_vals = pairs[
        (
            pairs["metric_a"].eq("vision_score_pm")
            & pairs["subcomponent_b"].eq("persistent_shared_information_maintenance")
        )
        | (
            pairs["metric_b"].eq("vision_score_pm")
            & pairs["subcomponent_a"].eq("persistent_shared_information_maintenance")
        )
    ]
    summary.append(
        {
            "boundary_level": "convergent_diagnostic",
            "pair_type": "vision_score_vs_persistent_indicators",
            "n_pairs": len(vision_vals),
            "mean_abs_r": vision_vals["abs_r"].mean(),
            "median_abs_r": vision_vals["abs_r"].median(),
        }
    )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(DIRS["tables"] / "construct_boundary_summary.csv", index=False)

    metric_order = [m for m in METRIC_NAMES if m in PRIMARY_BOUNDARY_METRICS]
    index = {m: i for i, m in enumerate(metric_order)}
    pp = primary[
        primary["metric_a"].isin(metric_order) & primary["metric_b"].isin(metric_order)
    ]
    pair_idx = [
        (index[a], index[b])
        for a, b in pp[["metric_a", "metric_b"]].itertuples(index=False)
    ]
    abs_values = pp["abs_r"].to_numpy()
    metric_info = dictionary.loc[metric_order]
    permutations = []
    null_rows = []
    for level, label_col in [("domain", "domain"), ("subcomponent", "subcomponent")]:
        labels = metric_info[label_col].to_numpy(dtype=object)
        same = np.asarray([labels[i] == labels[j] for i, j in pair_idx])
        observed = float(abs_values[same].mean() - abs_values[~same].mean())
        generator = rng(f"boundary_permutation_{level}")
        null = np.empty(N_PERMUTATIONS, dtype=np.float32)
        for idx_perm in range(N_PERMUTATIONS):
            perm = generator.permutation(labels)
            same_perm = np.asarray([perm[i] == perm[j] for i, j in pair_idx])
            null[idx_perm] = (
                abs_values[same_perm].mean() - abs_values[~same_perm].mean()
            )
        p = (1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1)
        permutations.append(
            {
                "boundary_level": level,
                "observed_gap": observed,
                "n_permutations": N_PERMUTATIONS,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std()),
                "null_ci_low": float(np.quantile(null, 0.025)),
                "null_ci_high": float(np.quantile(null, 0.975)),
                "permutation_p": float(p),
                "seed": child_seed(f"boundary_permutation_{level}"),
            }
        )
        null_rows.extend({"boundary_level": level, "gap": float(v)} for v in null)
    perm_df = pd.DataFrame(permutations)
    null_df = pd.DataFrame(null_rows)
    perm_df.to_csv(
        DIRS["tables"] / "construct_boundary_permutation_test.csv", index=False
    )
    null_df.to_csv(
        DIRS["source_data"] / "figure1c_permutation_null_distribution.csv", index=False
    )
    return corr, pairs, summary_df, perm_df


def boundary_analysis_revised(
    residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe overlap without treating formative indices as latent scales.

    The family-level permutation includes only multi-indicator families. The
    single assists indicator is retained in the full correlation matrix and
    summarized separately, but it cannot contribute a within-family pair.
    """
    dictionary = pd.DataFrame(METRICS).set_index("metric")
    corr = residuals[METRIC_NAMES].corr()
    corr.to_csv(DIRS["tables"] / "residual_correlation_matrix_18_metrics.csv")
    corr.to_csv(
        DIRS["source_data"] / "extended_data_figure1a_residual_correlation_matrix.csv"
    )

    rows = []
    for i, metric_a in enumerate(METRIC_NAMES):
        for metric_b in METRIC_NAMES[i + 1 :]:
            row_a, row_b = dictionary.loc[metric_a], dictionary.loc[metric_b]
            value = float(corr.loc[metric_a, metric_b])
            rows.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "label_a": row_a["label"],
                    "label_b": row_b["label"],
                    "family_a": row_a["family"],
                    "family_b": row_b["family"],
                    "subtype_a": row_a["subtype"],
                    "subtype_b": row_b["subtype"],
                    "role_a": row_a["role"],
                    "role_b": row_b["role"],
                    "primary_boundary_pair": bool(
                        row_a["primary_boundary"] and row_b["primary_boundary"]
                    ),
                    "r": value,
                    "abs_r": abs(value),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(DIRS["tables"] / "residual_metric_pair_correlations.csv", index=False)

    summary_rows: list[dict] = []
    primary = pairs[pairs["primary_boundary_pair"]].copy()
    family_pairs = primary[
        primary["family_a"].isin(BOUNDARY_FAMILY_ORDER)
        & primary["family_b"].isin(BOUNDARY_FAMILY_ORDER)
    ].copy()
    same_family = family_pairs["family_a"].eq(family_pairs["family_b"])
    for label, mask in [("within", same_family), ("between", ~same_family)]:
        values = family_pairs.loc[mask, "abs_r"]
        summary_rows.append(
            {
                "boundary_level": "multi_indicator_family",
                "pair_type": label,
                "n_pairs": int(len(values)),
                "mean_abs_r": float(values.mean()),
                "median_abs_r": float(values.median()),
            }
        )

    # Pairwise family cells expose overlap that a single within-between gap can
    # conceal, particularly between discrete signal functions.
    for i, family_a in enumerate(BOUNDARY_FAMILY_ORDER):
        for family_b in BOUNDARY_FAMILY_ORDER[i:]:
            if family_a == family_b:
                mask = family_pairs["family_a"].eq(family_a) & family_pairs[
                    "family_b"
                ].eq(family_b)
            else:
                mask = (
                    family_pairs["family_a"].eq(family_a)
                    & family_pairs["family_b"].eq(family_b)
                ) | (
                    family_pairs["family_a"].eq(family_b)
                    & family_pairs["family_b"].eq(family_a)
                )
            values = family_pairs.loc[mask, "abs_r"]
            summary_rows.append(
                {
                    "boundary_level": "family_pair",
                    "pair_type": f"{family_a}__{family_b}",
                    "n_pairs": int(len(values)),
                    "mean_abs_r": float(values.mean()),
                    "median_abs_r": float(values.median()),
                }
            )

    signal_pairs = pairs[
        pairs["family_a"].eq("team_signalling")
        & pairs["family_b"].eq("team_signalling")
    ].copy()
    same_subtype = signal_pairs["subtype_a"].eq(signal_pairs["subtype_b"])
    for label, mask in [("within", same_subtype), ("between", ~same_subtype)]:
        values = signal_pairs.loc[mask, "abs_r"]
        summary_rows.append(
            {
                "boundary_level": "signal_subtype",
                "pair_type": label,
                "n_pairs": int(len(values)),
                "mean_abs_r": float(values.mean()),
                "median_abs_r": float(values.median()),
            }
        )
    for subtype in SIGNAL_SUBTYPE_ORDER:
        mask = signal_pairs["subtype_a"].eq(subtype) & signal_pairs["subtype_b"].eq(
            subtype
        )
        values = signal_pairs.loc[mask, "abs_r"]
        summary_rows.append(
            {
                "boundary_level": "signal_subtype_detail",
                "pair_type": f"within_{subtype}",
                "n_pairs": int(len(values)),
                "mean_abs_r": float(values.mean()),
                "median_abs_r": float(values.median()),
            }
        )

    for family in [
        "shared_information_maintenance",
        "team_signalling",
        "combat_economy_performance",
    ]:
        mask = (pairs["metric_a"].eq("assists_pm") & pairs["family_b"].eq(family)) | (
            pairs["metric_b"].eq("assists_pm") & pairs["family_a"].eq(family)
        )
        values = pairs.loc[mask, "abs_r"]
        summary_rows.append(
            {
                "boundary_level": "single_indicator_benchmark",
                "pair_type": f"assists_vs_{family}",
                "n_pairs": int(len(values)),
                "mean_abs_r": float(values.mean()),
                "median_abs_r": float(values.median()),
            }
        )

    vision_mask = (
        pairs["metric_a"].eq("vision_score_pm")
        & pairs["family_b"].eq("shared_information_maintenance")
    ) | (
        pairs["metric_b"].eq("vision_score_pm")
        & pairs["family_a"].eq("shared_information_maintenance")
    )
    vision_values = pairs.loc[vision_mask, "abs_r"]
    summary_rows.append(
        {
            "boundary_level": "convergent_diagnostic",
            "pair_type": "vision_score_vs_maintenance_indicators",
            "n_pairs": int(len(vision_values)),
            "mean_abs_r": float(vision_values.mean()),
            "median_abs_r": float(vision_values.median()),
        }
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DIRS["tables"] / "construct_boundary_summary.csv", index=False)

    permutation_rows: list[dict] = []
    null_rows: list[dict] = []

    def permute_gap(level: str, metrics: list[str], label_column: str) -> None:
        metric_index = {metric: idx for idx, metric in enumerate(metrics)}
        selected = pairs[
            pairs["metric_a"].isin(metrics) & pairs["metric_b"].isin(metrics)
        ].copy()
        pair_indices = [
            (metric_index[a], metric_index[b])
            for a, b in selected[["metric_a", "metric_b"]].itertuples(index=False)
        ]
        values = selected["abs_r"].to_numpy(float)
        labels = dictionary.loc[metrics, label_column].to_numpy(dtype=object)
        same = np.asarray([labels[i] == labels[j] for i, j in pair_indices])
        observed = float(values[same].mean() - values[~same].mean())
        generator = rng(f"boundary_permutation_{level}")
        null = np.empty(N_PERMUTATIONS, dtype=np.float32)
        for index in range(N_PERMUTATIONS):
            permuted = generator.permutation(labels)
            permuted_same = np.asarray(
                [permuted[i] == permuted[j] for i, j in pair_indices]
            )
            null[index] = values[permuted_same].mean() - values[~permuted_same].mean()
        empirical_p = float((1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1))
        permutation_rows.append(
            {
                "boundary_level": level,
                "observed_gap": observed,
                "n_metrics": len(metrics),
                "n_pairs": len(values),
                "n_permutations": N_PERMUTATIONS,
                "null_mean": float(null.mean()),
                "null_sd": float(null.std()),
                "null_ci_low": float(np.quantile(null, 0.025)),
                "null_ci_high": float(np.quantile(null, 0.975)),
                "permutation_p": empirical_p,
                "seed": child_seed(f"boundary_permutation_{level}"),
            }
        )
        null_rows.extend(
            {"boundary_level": level, "gap": float(value)} for value in null
        )

    family_metrics = [
        metric
        for metric in PRIMARY_BOUNDARY_METRICS
        if dictionary.loc[metric, "family"] in BOUNDARY_FAMILY_ORDER
    ]
    signal_metrics = [
        item["metric"]
        for item in METRICS
        if item["family"] == "team_signalling" and item["primary_boundary"]
    ]
    permute_gap("multi_indicator_family", family_metrics, "family")
    permute_gap("signal_subtype", signal_metrics, "subtype")

    permutation = pd.DataFrame(permutation_rows)
    null_distribution = pd.DataFrame(null_rows)
    permutation.to_csv(
        DIRS["tables"] / "construct_boundary_permutation_test.csv", index=False
    )
    null_distribution.to_csv(
        DIRS["source_data"]
        / "extended_data_figure1c_permutation_null_distribution.csv",
        index=False,
    )
    return corr, pairs, summary, permutation


def player_aggregation(
    df: pd.DataFrame, residuals: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    work = pd.concat(
        [
            df[
                [
                    "match_id",
                    "puuid",
                    "region",
                    "tier",
                    "tier_num",
                    "team_position",
                    "patch",
                    "game_duration",
                    "win",
                ]
            ].reset_index(drop=True),
            residuals.reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    # The skill benchmark is both a terminal component and a top-level domain.
    # Preserve it once in aggregation outputs rather than creating a duplicate
    # parquet column with the same name.
    value_cols = list(dict.fromkeys(METRIC_NAMES + SUBCOMPONENT_ORDER + DOMAIN_ORDER))
    player = (
        work.groupby(["puuid", "region", "tier", "tier_num"], observed=True)[value_cols]
        .mean()
        .reset_index()
    )
    player.to_parquet(
        DIRS["data"] / "player_metric_and_process_scores_minimal.parquet", index=False
    )
    player_role = (
        work.groupby(
            ["puuid", "region", "tier", "tier_num", "team_position"], observed=True
        )[value_cols]
        .mean()
        .reset_index()
    )
    player_role["role_interdependence"] = player_role["team_position"].map(
        ROLE_INTERDEPENDENCE
    )
    player_role.to_parquet(
        DIRS["data"] / "player_role_metric_and_process_scores_minimal.parquet",
        index=False,
    )
    return player


def effects_table(
    player: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dictionary = pd.DataFrame(METRICS).set_index("metric")
    score_meta = [
        (measure, DOMAIN_LABELS[measure], PRIMARY_MEASURE_TYPES[measure], measure)
        for measure in DOMAIN_ORDER
    ]
    score_meta.extend(
        (subtype, SUBCOMPONENT_LABELS[subtype], "signal_subtype", "team_signalling")
        for subtype in SIGNAL_SUBTYPE_ORDER
    )
    rows = []
    for metric, label, level, domain in score_meta:
        for a, b in REGION_PAIRS:
            av = player.loc[player["region"].eq(a), metric].to_numpy()
            bv = player.loc[player["region"].eq(b), metric].to_numpy()
            d, low, high = bootstrap_d(av, bv, f"effect_{metric}_{a}_{b}", n_boot=1000)
            rows.append(
                {
                    "level": level,
                    "measure": metric,
                    "label": label,
                    "family": domain,
                    "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                    "region_a": a,
                    "region_b": b,
                    "cohens_d": d,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_method": "player bootstrap",
                    "bootstrap_n": 1000,
                }
            )
    score_effects = pd.DataFrame(rows)
    score_effects.to_csv(
        DIRS["tables"] / "regional_measure_and_signal_subtype_effects.csv", index=False
    )

    metric_rows = []
    for metric in METRIC_NAMES:
        for a, b in REGION_PAIRS:
            av = player.loc[player["region"].eq(a), metric].to_numpy()
            bv = player.loc[player["region"].eq(b), metric].to_numpy()
            d = cohen_d(av, bv)
            se = math.sqrt(
                (len(av) + len(bv)) / (len(av) * len(bv))
                + d * d / (2 * (len(av) + len(bv) - 2))
            )
            metric_rows.append(
                {
                    "metric": metric,
                    "label": dictionary.loc[metric, "label"],
                    "family": dictionary.loc[metric, "family"],
                    "subtype": dictionary.loc[metric, "subtype"],
                    "role": dictionary.loc[metric, "role"],
                    "primary_score": bool(dictionary.loc[metric, "primary_score"]),
                    "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                    "region_a": a,
                    "region_b": b,
                    "cohens_d": d,
                    "ci_low": d - 1.96 * se,
                    "ci_high": d + 1.96 * se,
                    "ci_method": "large-sample analytic",
                }
            )
    metric_effects = pd.DataFrame(metric_rows)
    metric_effects.to_csv(
        DIRS["tables"] / "regional_metric_effect_spectrum.csv", index=False
    )

    profile_cols = SCORE_ORDER
    profile = player.groupby("region", observed=True)[profile_cols].agg(
        ["mean", "std", "count"]
    )
    profile.columns = ["_".join(c) for c in profile.columns]
    profile = profile.reset_index()
    for col in profile_cols:
        profile[f"{col}_se"] = profile[f"{col}_std"] / np.sqrt(profile[f"{col}_count"])
        profile[f"{col}_ci_low"] = profile[f"{col}_mean"] - 1.96 * profile[f"{col}_se"]
        profile[f"{col}_ci_high"] = profile[f"{col}_mean"] + 1.96 * profile[f"{col}_se"]
    profile.to_csv(
        DIRS["tables"] / "adjusted_regional_process_profiles.csv", index=False
    )
    return score_effects, metric_effects, profile


def direct_domain_omnibus(
    df: pd.DataFrame, transformed_scores: pd.DataFrame
) -> pd.DataFrame:
    cache = DIRS["tables"] / "direct_region_by_measure_model.csv"
    if cache.exists():
        return pd.read_csv(cache)
    n = len(df)
    domains = DOMAIN_ORDER
    y = np.concatenate(
        [transformed_scores[d].to_numpy(dtype=np.float32) for d in domains]
    )
    domain_code = np.repeat(np.arange(len(domains), dtype=np.int16), n)
    region_code = pd.Categorical(df["region"], categories=REGION_ORDER).codes.astype(
        np.int16
    )
    tier_code = pd.Categorical(df["tier"], categories=TIER_ORDER).codes.astype(np.int16)
    role_code = pd.Categorical(df["team_position"]).codes.astype(np.int16)
    patch_code = pd.Categorical(df["patch"]).codes.astype(np.int16)
    duration = zscore(df["game_duration"]).astype(np.float32)
    repeated = {
        "region": np.tile(region_code, len(domains)),
        "domain": domain_code,
        "tier": np.tile(tier_code, len(domains)),
        "role": np.tile(role_code, len(domains)),
        "patch": np.tile(patch_code, len(domains)),
    }
    cat_full = pd.DataFrame(
        {
            **repeated,
            "region_domain": repeated["region"] * 10 + repeated["domain"],
            "tier_domain": repeated["tier"] * 10 + repeated["domain"],
            "role_domain": repeated["role"] * 10 + repeated["domain"],
            "patch_domain": repeated["patch"] * 10 + repeated["domain"],
        }
    )
    full_cols = [
        "region",
        "domain",
        "region_domain",
        "tier",
        "tier_domain",
        "role",
        "role_domain",
        "patch",
        "patch_domain",
    ]
    reduced_cols = [c for c in full_cols if c != "region_domain"]
    rows = []
    models = {}
    for label, cols in [
        ("without_region_by_domain", reduced_cols),
        ("with_region_by_domain", full_cols),
    ]:
        encoder = one_hot()
        xcat = encoder.fit_transform(cat_full[cols])
        dlong = np.tile(duration, len(domains))
        numeric = [dlong]
        for dom in range(1, len(domains)):
            numeric.append(dlong * (domain_code == dom))
        x = sparse.hstack(
            [
                sparse.csr_matrix(np.ones((len(y), 1))),
                xcat,
                sparse.csr_matrix(np.column_stack(numeric)),
            ],
            format="csr",
        )
        model = Ridge(alpha=1e-8, solver="lsqr", fit_intercept=False).fit(x, y)
        pred = model.predict(x)
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - pred) ** 2)) / sst
        rows.append(
            {
                "model": label,
                "r2": r2,
                "n_participant_domain_rows": len(y),
                "n_features": x.shape[1],
                "ridge_alpha": 1e-8,
            }
        )
        models[label] = r2
        del x, xcat, encoder, model, pred
    result = pd.DataFrame(rows)
    result["delta_r2_region_by_measure"] = np.nan
    result.loc[
        result["model"].eq("with_region_by_domain"), "delta_r2_region_by_measure"
    ] = (models["with_region_by_domain"] - models["without_region_by_domain"])
    result.to_csv(cache, index=False)
    return result


def build_team_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cache = DIRS["data"] / "unique_match_team_whole_match_telemetry.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if not SNAPSHOT_DB.exists():
        raise FileNotFoundError(SNAPSHOT_DB)
    meta = (
        df.groupby("match_id", observed=True)
        .agg(
            target_region=("region", "first"),
            target_tier=(
                "tier",
                lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0],
            ),
            target_patch=("patch", "first"),
            target_duration=("game_duration", "first"),
        )
        .reset_index()
    )
    meta.to_parquet(
        DIRS["data"] / "canonical_unique_match_metadata.parquet", index=False
    )
    con = sqlite3.connect(SNAPSHOT_DB)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("CREATE TEMP TABLE target_matches(match_id TEXT PRIMARY KEY)")
    con.executemany(
        "INSERT INTO target_matches(match_id) VALUES (?)",
        ((m,) for m in meta["match_id"].astype(str)),
    )
    query = """
        SELECT mp.match_id, mp.team_id, COUNT(*) AS team_members,
               SUM(mp.kills) AS kills, SUM(mp.deaths) AS deaths, SUM(mp.assists) AS assists,
               SUM(mp.gold_earned) AS gold_earned, SUM(mp.total_damage_dealt) AS total_damage_dealt,
               SUM(mp.vision_score) AS vision_score, SUM(mp.wards_placed) AS wards_placed,
               SUM(mp.wards_killed) AS wards_killed, SUM(mp.control_wards_bought) AS control_wards_bought,
               SUM(mp.enemy_vision_pings) AS enemy_vision_pings,
               SUM(mp.enemy_missing_pings) AS enemy_missing_pings,
               SUM(mp.need_vision_pings) AS need_vision_pings,
               SUM(mp.command_pings) AS command_pings, SUM(mp.on_my_way_pings) AS on_my_way_pings,
               SUM(mp.assist_me_pings) AS assist_me_pings, SUM(mp.push_pings) AS push_pings,
               SUM(mp.all_in_pings) AS all_in_pings, SUM(mp.get_back_pings) AS get_back_pings,
               m.blue_team_win,
               o.baron_kills, o.dragon_kills, o.rift_herald_kills, o.tower_kills, o.inhibitor_kills
        FROM match_participants mp
        JOIN target_matches tm ON tm.match_id = mp.match_id
        JOIN matches m ON m.match_id = mp.match_id
        LEFT JOIN team_objectives o ON o.match_id = mp.match_id AND o.team_id = mp.team_id
        GROUP BY mp.match_id, mp.team_id
    """
    team = pd.read_sql_query(query, con)
    con.close()
    team = team.merge(meta, on="match_id", how="inner")
    team = team[team["team_members"].eq(5) & team["team_id"].isin([100, 200])].copy()
    match_team_count = team.groupby("match_id").size()
    team = team[
        team["match_id"].isin(match_team_count[match_team_count.eq(2)].index)
    ].copy()
    duration_min = team["target_duration"].clip(lower=1) / 60.0
    for metric in METRIC_NAMES:
        raw = metric.replace("_pm", "")
        team[metric] = team[raw].fillna(0) / duration_min
    team.to_parquet(cache, index=False)
    return team


def team_functional_alignment(
    team: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transformed = pd.DataFrame(index=team.index)
    for item in METRICS:
        values, _, _ = transform_metric(team[item["metric"]])
        values = zscore(values)
        if item["reverse"]:
            values = -values
        transformed[item["metric"]] = values
    scores = build_scores(transformed)
    team = pd.concat(
        [team.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
    )
    keep = [
        "match_id",
        "team_id",
        "target_region",
        "target_tier",
        "target_patch",
        "target_duration",
        "blue_team_win",
        "baron_kills",
        "dragon_kills",
        "rift_herald_kills",
        "tower_kills",
        "inhibitor_kills",
        *DOMAIN_ORDER,
    ]
    blue = team[team["team_id"].eq(100)][keep].copy()
    red = team[team["team_id"].eq(200)][keep].copy()
    pair = blue.merge(
        red, on="match_id", suffixes=("_blue", "_red"), validate="one_to_one"
    )
    pair["region"] = pair["target_region_blue"]
    pair["tier"] = pair["target_tier_blue"]
    pair["patch"] = pair["target_patch_blue"]
    pair["game_duration"] = pair["target_duration_blue"]
    pair["blue_win"] = pair["blue_team_win_blue"].astype(int)
    for domain in DOMAIN_ORDER:
        pair[f"{domain}_diff"] = pair[f"{domain}_blue"] - pair[f"{domain}_red"]
    pair["neutral_objective_diff"] = (
        pair["baron_kills_blue"]
        + pair["dragon_kills_blue"]
        + pair["rift_herald_kills_blue"]
        - pair["baron_kills_red"]
        - pair["dragon_kills_red"]
        - pair["rift_herald_kills_red"]
    )
    pair["building_objective_diff"] = (
        pair["tower_kills_blue"]
        + pair["inhibitor_kills_blue"]
        - pair["tower_kills_red"]
        - pair["inhibitor_kills_red"]
    )
    pair.to_parquet(
        DIRS["data"] / "unique_match_team_paired_functional_dataset.parquet",
        index=False,
    )

    predictors = [f"{d}_diff" for d in DOMAIN_ORDER]
    work = (
        pair[
            [
                "region",
                "tier",
                "patch",
                "game_duration",
                "blue_win",
                "neutral_objective_diff",
                "building_objective_diff",
                *predictors,
            ]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    x_ctrl, _ = control_design(work, ["region", "tier", "patch"], ["game_duration"])
    xp = np.column_stack([zscore(work[p]) for p in predictors])
    x = sparse.hstack([x_ctrl, sparse.csr_matrix(xp)], format="csr")
    indices = list(range(x.shape[1] - len(predictors), x.shape[1]))
    rows = []
    logit = fit_sparse_logistic(x, work["blue_win"].to_numpy(), indices)
    for predictor, effect in zip(predictors, logit["effects"]):
        beta, se, low, high, odds, ame = effect
        rows.append(
            {
                "outcome": "final_win",
                "outcome_type": "binary",
                "predictor": predictor.replace("_diff", ""),
                "beta": beta,
                "se": se,
                "ci_low": low,
                "ci_high": high,
                "odds_ratio": odds,
                "average_marginal_effect": ame,
                "n_matches": len(work),
                "interpretation": "whole-match convergent association",
            }
        )
    for outcome in ["neutral_objective_diff", "building_objective_diff"]:
        linear = fit_sparse_linear(x, zscore(work[outcome]), indices)
        for predictor, effect in zip(predictors, linear["effects"]):
            beta, se, low, high = effect
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_type": "continuous_standardized",
                    "predictor": predictor.replace("_diff", ""),
                    "beta": beta,
                    "se": se,
                    "ci_low": low,
                    "ci_high": high,
                    "odds_ratio": np.nan,
                    "average_marginal_effect": np.nan,
                    "n_matches": len(work),
                    "r2": linear["r2"],
                    "interpretation": "whole-match convergent association",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DIRS["tables"] / "whole_match_functional_alignment.csv", index=False)

    # Comparable descriptive scale for the main figure. Each process difference
    # and outcome is first residualized on the same task-context controls, then
    # related using a partial Pearson correlation. The simultaneous regression
    # estimates above remain available to expose attenuation and suppression.
    context_x, _ = control_design(work, ["region", "tier", "patch"], ["game_duration"])
    residualized = {}
    for variable in [
        *predictors,
        "blue_win",
        "neutral_objective_diff",
        "building_objective_diff",
    ]:
        values = work[variable].to_numpy(dtype=float)
        fit = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False).fit(
            context_x, values
        )
        residualized[variable] = values - fit.predict(context_x)
    marginal_rows = []
    outcome_map = {
        "neutral_objective_diff": "neutral_objective_diff",
        "building_objective_diff": "building_objective_diff",
        "blue_win": "final_win",
    }
    df_fisher = max(len(work) - context_x.shape[1] - 3, 1)
    for predictor in predictors:
        for outcome_col, outcome_label in outcome_map.items():
            r = float(
                np.corrcoef(residualized[predictor], residualized[outcome_col])[0, 1]
            )
            z = np.arctanh(np.clip(r, -0.999999, 0.999999))
            se_z = 1 / math.sqrt(df_fisher)
            marginal_rows.append(
                {
                    "predictor": predictor.replace("_diff", ""),
                    "outcome": outcome_label,
                    "partial_r": r,
                    "ci_low": float(np.tanh(z - 1.96 * se_z)),
                    "ci_high": float(np.tanh(z + 1.96 * se_z)),
                    "n_matches": len(work),
                    "controls": "region+tier+patch+game_duration",
                    "interpretation": "context-adjusted whole-match convergent association",
                }
            )
    marginal = pd.DataFrame(marginal_rows)
    marginal.to_csv(
        DIRS["tables"] / "whole_match_context_adjusted_marginal_associations.csv",
        index=False,
    )
    marginal.to_csv(
        DIRS["source_data"]
        / "figure1d_whole_match_context_adjusted_partial_correlations.csv",
        index=False,
    )
    return result, marginal, pair


def overlap_sensitivity(
    df: pd.DataFrame, residuals: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    one_idx = np.load(DIRS["data"] / "one_focal_per_match_indices.npy")
    singleton = np.load(DIRS["data"] / "singleton_match_mask.npy")
    base = pd.concat(
        [
            df[["match_id", "puuid", "region"]].reset_index(drop=True),
            residuals.reset_index(drop=True),
            scores.reset_index(drop=True),
        ],
        axis=1,
    )
    rows = []
    for sample, part in [
        ("full_canonical", base),
        ("one_focal_per_match", base.iloc[one_idx]),
        ("singleton_matches_only", base.loc[singleton]),
    ]:
        player = (
            part.groupby(["puuid", "region"], observed=True)[SCORE_ORDER]
            .mean()
            .reset_index()
        )
        for measure in SCORE_ORDER:
            for a, b in REGION_PAIRS:
                d = cohen_d(
                    player.loc[player["region"].eq(a), measure],
                    player.loc[player["region"].eq(b), measure],
                )
                rows.append(
                    {
                        "sample": sample,
                        "measure": measure,
                        "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                        "cohens_d": d,
                        "n_players": player["puuid"].nunique(),
                        "n_rows": len(part),
                    }
                )
    result = pd.DataFrame(rows)
    pivot = result.pivot_table(
        index=["measure", "contrast"], columns="sample", values="cohens_d"
    ).reset_index()
    pivot["one_focal_absolute_shift"] = (
        pivot["one_focal_per_match"] - pivot["full_canonical"]
    ).abs()
    pivot["singleton_absolute_shift"] = (
        pivot["singleton_matches_only"] - pivot["full_canonical"]
    ).abs()
    pivot["one_focal_sign_reversal"] = np.sign(pivot["one_focal_per_match"]) != np.sign(
        pivot["full_canonical"]
    )
    pivot["singleton_sign_reversal"] = np.sign(
        pivot["singleton_matches_only"]
    ) != np.sign(pivot["full_canonical"])
    result.to_csv(
        DIRS["tables"] / "shared_match_sensitivity_all_estimates.csv", index=False
    )
    pivot.to_csv(DIRS["tables"] / "shared_match_sensitivity_flags.csv", index=False)
    return pivot


def duration_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    specs = [m for m in METRICS if m["primary_score"]]
    rows = []
    regimes = [
        ("no_additional_duration_control", [], df),
        (
            "duration_polynomial",
            ["duration_z", "duration_z2", "duration_z3"],
            df.assign(
                duration_z=zscore(df["game_duration"]),
                duration_z2=zscore(df["game_duration"]) ** 2,
                duration_z3=zscore(df["game_duration"]) ** 3,
            ),
        ),
    ]
    for name, numeric, frame in regimes:
        residuals = residualize_metrics(
            frame,
            specs,
            ["tier", "team_position", "patch"],
            numeric,
            DIRS["data"] / f"participant_metric_residuals_{name}.parquet",
            DIRS["tables"] / f"metric_preprocessing_audit_{name}.csv",
        )
        scores = build_scores(residuals)
        joined = pd.concat(
            [
                df[["puuid", "region"]].reset_index(drop=True),
                scores.reset_index(drop=True),
            ],
            axis=1,
        )
        player = (
            joined.groupby(["puuid", "region"], observed=True)[DOMAIN_ORDER]
            .mean()
            .reset_index()
        )
        for measure in DOMAIN_ORDER:
            for a, b in REGION_PAIRS:
                rows.append(
                    {
                        "regime": name,
                        "measure": measure,
                        "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                        "cohens_d": cohen_d(
                            player.loc[player.region.eq(a), measure],
                            player.loc[player.region.eq(b), measure],
                        ),
                    }
                )
        del residuals, scores, joined, player

    # Aggregate Poisson exposure models for two key count metrics.
    poisson_rows = []
    count_specs = [
        ("wards_killed", "Wards cleared"),
        ("control_wards_bought", "Control wards purchased"),
    ]
    for count, label in count_specs:
        cells = (
            df.groupby(["region", "tier", "team_position", "patch"], observed=True)
            .agg(
                count_sum=(count, "sum"),
                exposure_minutes=("duration_min", "sum"),
                n=(count, "size"),
            )
            .reset_index()
        )
        cells["region"] = pd.Categorical(cells["region"], categories=REGION_ORDER)
        model = smf.glm(
            "count_sum ~ C(region) + C(tier) + C(team_position) + C(patch)",
            data=cells,
            family=sm.families.Poisson(),
            offset=np.log(cells["exposure_minutes"].clip(lower=1e-6)),
        ).fit()
        for region in ["EUW1", "KR"]:
            term = f"C(region)[T.{region}]"
            beta = model.params.get(term, np.nan)
            se = model.bse.get(term, np.nan)
            poisson_rows.append(
                {
                    "metric": count,
                    "label": label,
                    "contrast": f"{REGION_LABELS[region]}-NA",
                    "log_rate_ratio": beta,
                    "ci_low": beta - 1.96 * se,
                    "ci_high": beta + 1.96 * se,
                    "incidence_rate_ratio": math.exp(beta),
                    "n_cells": len(cells),
                }
            )
    pd.DataFrame(poisson_rows).to_csv(
        DIRS["tables"] / "poisson_offset_key_metric_sensitivity.csv", index=False
    )
    result = pd.DataFrame(rows)
    result.to_csv(DIRS["tables"] / "duration_handling_sensitivity.csv", index=False)
    return result


def raw_key_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label in [
        ("wards_killed_pm", "Wards cleared"),
        ("control_wards_bought_pm", "Control wards purchased"),
        ("wards_placed_pm", "Wards placed"),
    ]:
        player = (
            df.groupby(["puuid", "region"], observed=True)[metric].mean().reset_index()
        )
        for region, part in player.groupby("region", observed=True):
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "region": region,
                    "display_region": REGION_LABELS[region],
                    "mean_per_minute": part[metric].mean(),
                    "mean_per_10_minutes": part[metric].mean() * 10,
                    "sd_per_minute": part[metric].std(),
                    "n_players": len(part),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(DIRS["tables"] / "key_metric_raw_units_by_region.csv", index=False)
    result.to_csv(
        DIRS["source_data"] / "figure2d_key_metric_raw_units.csv", index=False
    )
    return result


def draw_figure1(corr, pairs, summary, perm, marginal):
    """Draw the construct-boundary figure with three explicit primary domains."""
    set_figure_style()
    dictionary = pd.DataFrame(METRICS).set_index("metric")
    fig = plt.figure(figsize=(7.25, 6.55))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.42, 1.0],
        height_ratios=[1.18, 0.72, 1.02],
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.94,
        wspace=0.43,
        hspace=0.52,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    top_right = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.31], hspace=0.53)
    ax_b = fig.add_subplot(top_right[0, 0])
    ax_bi = fig.add_subplot(top_right[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 1])

    order = METRIC_NAMES
    labels = [
        dictionary.loc[m, "short"] + ("†" if m == "vision_score_pm" else "")
        for m in order
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "correlation_zero_white", ["#A55B52", "#FFFFFF", "#2E7772"]
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.85)
    im = ax_a.imshow(corr.loc[order, order], cmap=cmap, norm=norm, aspect="equal")
    ax_a.set_anchor("N")
    ax_a.set_xticks(range(len(order)), labels, rotation=53, ha="right", fontsize=5.0)
    ax_a.set_yticks(range(len(order)), labels, fontsize=5.1)
    ax_a.tick_params(length=0, pad=1.6)

    # Exactly three solid outlines denote the three primary formative domains.
    # Subcomponent boundaries are thin dotted separators within those outlines.
    for start, size in [(1, 6), (7, 7), (14, 4)]:
        ax_a.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                size,
                size,
                fill=False,
                ec="#202020",
                lw=1.0,
                zorder=4,
            )
        )
    for boundary, start, end in [(3.5, 0.5, 6.5), (12.5, 6.5, 13.5)]:
        ax_a.plot(
            [boundary, boundary],
            [start, end],
            color="#4B4B4B",
            lw=0.65,
            ls=(0, (2, 2)),
            zorder=4,
        )
        ax_a.plot(
            [start, end],
            [boundary, boundary],
            color="#4B4B4B",
            lw=0.65,
            ls=(0, (2, 2)),
            zorder=4,
        )
    ax_a.text(
        0.01,
        -0.19,
        "† Vision score is a convergent diagnostic and is not included in the primary scores.",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="#555555",
    )
    ax_a.text(
        0.01,
        -0.245,
        "Solid outlines: three primary domains; dotted separators: nested subcomponents.",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color="#555555",
    )
    cax = ax_a.inset_axes([0.12, 1.045, 0.76, 0.026])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([-0.5, 0, 0.4, 0.8])
    cb.ax.tick_params(labelsize=5.4, length=2, pad=1)
    cb.set_label("Residual Pearson r", fontsize=6.0, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")

    # Three-by-three summary retains the primary hierarchy and makes overlap visible.
    primary = pairs[pairs["primary_pair"].astype(str).str.lower().eq("true")].copy()
    domain_matrix = np.full((3, 3), np.nan)
    for i, d1 in enumerate(DOMAIN_ORDER):
        for j, d2 in enumerate(DOMAIN_ORDER):
            if i == j:
                mask = primary["domain_a"].eq(d1) & primary["domain_b"].eq(d1)
            else:
                mask = (primary["domain_a"].eq(d1) & primary["domain_b"].eq(d2)) | (
                    primary["domain_a"].eq(d2) & primary["domain_b"].eq(d1)
                )
            domain_matrix[i, j] = primary.loc[mask, "abs_r"].mean()
    short_domains = [
        "Information\nsupport",
        "Action\ncoordination",
        "Skill\nperformance",
    ]
    im_b = ax_b.imshow(
        domain_matrix,
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "boundary_strength", ["#FFFFFF", "#D9E9E6", "#2F7F78"]
        ),
        vmin=0,
        vmax=0.35,
        aspect="auto",
    )
    ax_b.set_xticks(range(3), short_domains, fontsize=5.6)
    ax_b.set_yticks(range(3), short_domains, fontsize=5.6)
    ax_b.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            ax_b.text(
                j,
                i,
                f"{domain_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if domain_matrix[i, j] > 0.24 else "#222222",
            )
    ax_b.set_xlabel("Mean |residual r|", fontsize=6.2, labelpad=3)
    pd.DataFrame(domain_matrix, index=DOMAIN_ORDER, columns=DOMAIN_ORDER).to_csv(
        DIRS["source_data"] / "figure1b_primary_domain_boundary_matrix.csv"
    )

    hierarchy_keys = [
        "within_persistent",
        "between_information_subcomponents",
        "within_event_signalling",
    ]
    hierarchy_labels = [
        "Within\npersistent",
        "Between\nsubcomponents",
        "Within event\nsignals",
    ]
    hierarchy_source = summary[
        summary["boundary_level"].eq("information_hierarchy")
    ].set_index("pair_type")
    hierarchy = hierarchy_source.reindex(hierarchy_keys)["mean_abs_r"].to_numpy(float)
    ax_bi.imshow(
        hierarchy.reshape(1, -1),
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "info_hierarchy", ["#FFFFFF", "#CDE1DE", "#236A64"]
        ),
        vmin=0,
        vmax=0.35,
        aspect="auto",
    )
    ax_bi.set_xticks(range(3), hierarchy_labels, fontsize=5.2)
    ax_bi.set_yticks([])
    ax_bi.tick_params(length=0)
    for j, val in enumerate(hierarchy):
        ax_bi.text(
            j,
            0,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=5.9,
            color="white" if val > 0.24 else "#222222",
        )
    ax_bi.set_ylabel(
        "Information\nhierarchy",
        rotation=0,
        ha="right",
        va="center",
        fontsize=5.4,
        labelpad=8,
    )
    pd.DataFrame({"relation": hierarchy_keys, "mean_abs_r": hierarchy}).to_csv(
        DIRS["source_data"] / "figure1b_information_hierarchy.csv", index=False
    )

    null = pd.read_csv(
        DIRS["source_data"] / "figure1c_permutation_null_distribution.csv"
    )
    top = null[null["boundary_level"].eq("domain")]
    observed = perm.loc[perm["boundary_level"].eq("domain"), "observed_gap"].iloc[0]
    ax_c.hist(top["gap"], bins=45, color="#D9DEDD", edgecolor="white", linewidth=0.35)
    ax_c.axvline(observed, color="#236A64", lw=1.8)
    p = perm.loc[perm["boundary_level"].eq("domain"), "permutation_p"].iloc[0]
    ax_c.text(
        0.98,
        0.94,
        f"Observed gap = {observed:.3f}\nPermutation P = {p:.4f}",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
    )
    ax_c.set_xlabel("Within-minus-between mean |r|")
    ax_c.set_ylabel("Label permutations")

    heat = marginal.pivot(
        index="predictor", columns="outcome", values="partial_r"
    ).reindex(
        index=DOMAIN_ORDER,
        columns=["neutral_objective_diff", "building_objective_diff", "final_win"],
    )
    im2 = ax_d.imshow(
        heat,
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "partial_r", ["#A55B52", "#FFFFFF", "#2F7F78"]
        ),
        norm=mpl.colors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.85),
        aspect="auto",
    )
    ax_d.set_xticks(
        range(3), ["Neutral\nobjectives", "Buildings", "Final win"], fontsize=5.9
    )
    ax_d.set_yticks(range(3), short_domains, fontsize=5.9)
    ax_d.tick_params(length=0)
    ax_d.set_xlabel("Whole-match functional association")
    for i in range(3):
        for j in range(3):
            val = heat.iloc[i, j]
            ax_d.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="white" if val > 0.48 else "#222222",
            )
    cax2 = ax_d.inset_axes([1.04, 0.08, 0.025, 0.84])
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_ticks([0, 0.4, 0.8])
    cb2.ax.tick_params(labelsize=5.1, length=2, pad=1)
    cb2.set_label("Partial r", fontsize=5.7, labelpad=2)

    for label, ax, xoff in [
        ("a", ax_a, -0.12),
        ("b", ax_b, -0.16),
        ("c", ax_c, -0.14),
        ("d", ax_d, -0.16),
    ]:
        ax.text(
            xoff,
            1.045,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
    save_figure(
        fig, DIRS["figures"] / "figure1_structured_behavioural_families_and_function"
    )


def draw_figure2(profile, score_effects, metric_effects, raw_units):
    set_figure_style()
    fig = plt.figure(figsize=(7.25, 6.2))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.02, 1.22, 0.72],
        height_ratios=[0.82, 1.28],
        wspace=0.48,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2])

    x = np.arange(len(DOMAIN_ORDER))
    for region in REGION_ORDER:
        row = profile[profile["region"].eq(region)].iloc[0]
        y = [row[f"{d}_mean"] for d in DOMAIN_ORDER]
        low = [row[f"{d}_ci_low"] for d in DOMAIN_ORDER]
        high = [row[f"{d}_ci_high"] for d in DOMAIN_ORDER]
        ax_a.errorbar(
            x,
            y,
            yerr=[np.asarray(y) - np.asarray(low), np.asarray(high) - np.asarray(y)],
            marker="o",
            lw=1.4,
            ms=4,
            capsize=2,
            color=REGION_COLORS[region],
            label=REGION_LABELS[region],
        )
    ax_a.axhline(0, color="#999999", lw=0.7)
    ax_a.set_xticks(x, ["Information", "Action", "Skill"])
    ax_a.set_ylabel("Adjusted player score")
    ax_a.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.19))

    subs = [
        "persistent_shared_information_maintenance",
        "event_based_informational_signalling",
    ]
    for region in REGION_ORDER:
        row = profile[profile["region"].eq(region)].iloc[0]
        y = [row[f"{s}_mean"] for s in subs]
        ax_b.plot(
            range(2),
            y,
            marker="o",
            lw=1.5,
            ms=4,
            color=REGION_COLORS[region],
            label=REGION_LABELS[region],
        )
    ax_b.axhline(0, color="#999999", lw=0.7)
    ax_b.set_xticks(
        range(2),
        [
            "Persistent shared-\ninformation maintenance",
            "Event-based\ninformational signalling",
        ],
    )
    ax_b.set_ylabel("Adjusted player score")

    dictionary = pd.DataFrame(METRICS).set_index("metric")
    order = list(reversed(METRIC_NAMES))
    y = np.arange(len(order))
    offsets = {"KR-NA": -0.20, "KR-EUW": 0, "EUW-NA": 0.20}
    pair_colors = REGION_PAIR_COLORS
    for contrast in ["KR-NA", "KR-EUW", "EUW-NA"]:
        sub = (
            metric_effects[metric_effects["contrast"].eq(contrast)]
            .set_index("metric")
            .reindex(order)
        )
        ax_c.errorbar(
            sub["cohens_d"],
            y + offsets[contrast],
            xerr=[sub["cohens_d"] - sub["ci_low"], sub["ci_high"] - sub["cohens_d"]],
            fmt="o",
            ms=2.8,
            lw=0.75,
            capsize=1.2,
            color=pair_colors[contrast],
            label=contrast,
        )
    ax_c.axvline(0, color="#777777", lw=0.7)
    ax_c.set_yticks(y, [dictionary.loc[m, "short"] for m in order])
    ax_c.set_xlabel("Signed Cohen's d")
    ax_c.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    # Subtle domain bands preserve the hierarchy without treating the two
    # information subcomponents as peer domains.
    for idx, metric in enumerate(order):
        color = DOMAIN_COLORS[dictionary.loc[metric, "domain"]]
        ax_c.axhspan(idx - 0.48, idx + 0.48, color=color, alpha=0.045, lw=0)
    ax_c.axhline(order.index("vision_score_pm") + 0.5, color="#A9C8C4", lw=0.7, ls=":")
    for idx in range(len(order) - 1):
        if (
            dictionary.loc[order[idx], "domain"]
            != dictionary.loc[order[idx + 1], "domain"]
        ):
            ax_c.axhline(idx + 0.5, color="#8C8C8C", lw=0.75)

    ru = raw_units.pivot(
        index="label", columns="display_region", values="mean_per_10_minutes"
    ).reindex(["Wards cleared", "Control wards purchased", "Wards placed"])
    xx = np.arange(len(ru))
    width = 0.22
    for i, region in enumerate(["NA", "EUW", "KR"]):
        ax_d.bar(
            xx + (i - 1) * width,
            ru[region],
            width=width,
            color={
                "NA": REGION_COLORS["NA1"],
                "EUW": REGION_COLORS["EUW1"],
                "KR": REGION_COLORS["KR"],
            }[region],
            label=region,
        )
    ax_d.set_xticks(
        xx,
        ["Cleared", "Control\npurchased", "Placed"],
        rotation=25,
        ha="right",
        fontsize=5.7,
    )
    ax_d.set_ylabel("Mean events per 10 min")
    ax_d.legend(ncol=1, loc="upper left", fontsize=5.6)

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(
            -0.12 if ax is not ax_c else -0.045,
            1.04,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
    save_figure(fig, DIRS["figures"] / "figure2_regional_shared_information_profiles")


def main():
    ensure_dirs()
    started = time.time()
    df = validate_and_load()
    overlap, regional_overlap = sample_audit(df)
    audit = json.loads(
        (DIRS["data_audit"] / "canonical_invariants.json").read_text(encoding="utf-8")
    )
    residuals, residual_scores, transformed_scores = preprocess(df)
    corr, pairs, boundary_summary, perm = boundary_analysis_revised(residuals)
    player = player_aggregation(df, residuals, residual_scores)
    score_effects, metric_effects, profile = effects_table(player)
    raw_units = raw_key_metric_table(df)
    omnibus = direct_domain_omnibus(df, transformed_scores)
    team = build_team_dataset(df)
    functional, marginal_functional, paired_team = team_functional_alignment(team)
    overlap_flags = overlap_sensitivity(df, residuals, residual_scores)
    duration = duration_sensitivity(df)
    summary = {
        "sample": audit,
        "shared_matches": json.loads(
            (DIRS["data_audit"] / "shared_match_summary.json").read_text(
                encoding="utf-8"
            )
        ),
        "boundary": perm.to_dict("records"),
        "regional_effects": score_effects.to_dict("records"),
        "direct_model": omnibus.to_dict("records"),
        "whole_match_function": functional.to_dict("records"),
        "whole_match_context_adjusted_marginal_associations": marginal_functional.to_dict(
            "records"
        ),
        "taxonomy": {
            "process_families": ["shared_information_maintenance", "team_signalling"],
            "signal_subtypes": SIGNAL_SUBTYPE_ORDER,
            "task_performance_indicators": [
                "joint_action_participation",
                "combat_economy_performance",
            ],
            "vision_score": "convergent diagnostic only",
        },
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(DIRS["reports"] / "core_machine_readable_summary.json", summary)
    print(
        json.dumps(
            {"status": "complete", "elapsed_minutes": summary["elapsed_minutes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
