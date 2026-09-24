"""Extended Data diagnostics under the 2026-07-17 taxonomy."""

from __future__ import annotations

import itertools
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from common import (
    DIRS,
    ROOT,
    child_seed,
    cohen_d,
    ensure_dirs,
    one_hot,
    rng,
    save_figure,
    set_figure_style,
    write_json,
    zscore,
)
from constructs import (
    DOMAIN_COLORS,
    DOMAIN_COMPONENTS,
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    INTERDEPENDENCE_ORDER,
    LEGACY_FOUR_CHANNELS,
    MASTER_SEED,
    METRICS,
    METRIC_NAMES,
    REGION_COLORS,
    REGION_LABELS,
    REGION_ORDER,
    REGION_PAIR_COLORS,
    ROLE_INTERDEPENDENCE,
    ROLE_ORDER,
    SUBCOMPONENT_LABELS,
    SUBCOMPONENT_METRICS,
    SUBCOMPONENT_ORDER,
    SIGNAL_SUBTYPE_ORDER,
    SCORE_ORDER,
    TIER_ORDER,
)

SCORES_PATH = DIRS["data"] / "participant_process_scores_minimal.parquet"
RESIDUALS_PATH = DIRS["data"] / "participant_metric_residuals_minimal.parquet"
TRANSFORMED_PATH = DIRS["data"] / "participant_metrics_transformed_z.parquet"
PLAYER_PATH = DIRS["data"] / "player_metric_and_process_scores_minimal.parquet"
PLAYER_ROLE_PATH = (
    DIRS["data"] / "player_role_metric_and_process_scores_minimal.parquet"
)
CANONICAL = (
    ROOT
    / "outputs"
    / "study0"
    / "data"
    / "study0_canonical_participant_match_sample.pkl"
)
REGION_PAIRS = [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]


def score_from_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for sub in SUBCOMPONENT_ORDER:
        metrics = SUBCOMPONENT_METRICS[sub]
        out[sub] = zscore(frame[metrics].mean(axis=1).to_numpy())
    for domain in DOMAIN_ORDER:
        out[domain] = zscore(out[DOMAIN_COMPONENTS[domain]].mean(axis=1).to_numpy())
    return out


def analytic_d_row(values_a, values_b) -> tuple[float, float, float]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    d = cohen_d(a, b)
    se = math.sqrt(
        (len(a) + len(b)) / (len(a) * len(b)) + d * d / (2 * (len(a) + len(b) - 2))
    )
    return d, d - 1.96 * se, d + 1.96 * se


def contextual_trajectories() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participant = pd.read_parquet(SCORES_PATH)
    player = (
        participant.groupby(["puuid", "region", "tier", "tier_num"], observed=True)[
            DOMAIN_ORDER
            + [
                "persistent_shared_information_maintenance",
                "event_based_informational_signalling",
            ]
        ]
        .mean()
        .reset_index()
    )
    tier_rows = []
    for tier in TIER_ORDER:
        part = player[player.tier.eq(tier)]
        for measure in DOMAIN_ORDER + [
            "persistent_shared_information_maintenance",
            "event_based_informational_signalling",
        ]:
            for a, b in REGION_PAIRS:
                d, lo, hi = analytic_d_row(
                    part.loc[part.region.eq(a), measure],
                    part.loc[part.region.eq(b), measure],
                )
                tier_rows.append(
                    {
                        "tier": tier,
                        "tier_num": TIER_ORDER.index(tier) + 1,
                        "measure": measure,
                        "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                        "cohens_d": d,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_a": int((part.region.eq(a)).sum()),
                        "n_b": int((part.region.eq(b)).sum()),
                    }
                )
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(
        DIRS["tables"] / "context_tier_signed_regional_trajectories.csv", index=False
    )

    pr = pd.read_parquet(PLAYER_ROLE_PATH)
    pr = pr[pr.team_position.isin(ROLE_ORDER)].copy()
    pr["role_interdependence"] = pr.team_position.map(ROLE_INTERDEPENDENCE)
    pooled = (
        pr.groupby(["puuid", "region", "role_interdependence"], observed=True)[
            DOMAIN_ORDER
            + [
                "persistent_shared_information_maintenance",
                "event_based_informational_signalling",
            ]
        ]
        .mean()
        .reset_index()
    )
    role_rows = []
    for level in INTERDEPENDENCE_ORDER:
        part = pooled[pooled.role_interdependence.eq(level)]
        for measure in DOMAIN_ORDER + [
            "persistent_shared_information_maintenance",
            "event_based_informational_signalling",
        ]:
            for a, b in REGION_PAIRS:
                d, lo, hi = analytic_d_row(
                    part.loc[part.region.eq(a), measure],
                    part.loc[part.region.eq(b), measure],
                )
                role_rows.append(
                    {
                        "role_interdependence": level,
                        "level_num": INTERDEPENDENCE_ORDER.index(level) + 1,
                        "measure": measure,
                        "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                        "cohens_d": d,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_a": int((part.region.eq(a)).sum()),
                        "n_b": int((part.region.eq(b)).sum()),
                    }
                )
    role_df = pd.DataFrame(role_rows)
    role_df.to_csv(
        DIRS["tables"] / "context_role_ecology_signed_regional_trajectories.csv",
        index=False,
    )

    endpoint = []
    for source, frame, low, high in [
        ("tier", tier_df, "IRON", "DIAMOND"),
        ("role_ecology", role_df, "Lower", "High"),
    ]:
        level_col = "tier" if source == "tier" else "role_interdependence"
        for (measure, contrast), group in frame.groupby(
            ["measure", "contrast"], observed=True
        ):
            a = group[group[level_col].eq(low)].cohens_d.iloc[0]
            b = group[group[level_col].eq(high)].cohens_d.iloc[0]
            endpoint.append(
                {
                    "context": source,
                    "measure": measure,
                    "contrast": contrast,
                    "low_endpoint": low,
                    "high_endpoint": high,
                    "signed_d_low": a,
                    "signed_d_high": b,
                    "signed_change": b - a,
                    "absolute_change": abs(b) - abs(a),
                    "direction_stable": np.sign(a) == np.sign(b),
                }
            )
    endpoint = pd.DataFrame(endpoint)
    endpoint.to_csv(DIRS["tables"] / "context_endpoint_changes.csv", index=False)
    return tier_df, role_df, endpoint


def _encode(values):
    enc = one_hot()
    return enc.fit_transform(np.asarray(values).reshape(-1, 1))


def direct_context_and_variance_models() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    participant = pd.read_parquet(SCORES_PATH)
    transformed = pd.read_parquet(TRANSFORMED_PATH)
    standard_role = participant.team_position.isin(ROLE_ORDER).to_numpy()
    participant = participant.loc[standard_role].reset_index(drop=True)
    transformed = transformed.loc[standard_role].reset_index(drop=True)
    scores = score_from_metric_frame(transformed)
    n = len(participant)
    domains = DOMAIN_ORDER
    n_domains = len(domains)
    y = np.concatenate([scores[d].to_numpy(dtype=np.float32) for d in domains])
    domain = np.repeat(np.arange(n_domains, dtype=np.int16), n)
    region = np.tile(
        pd.Categorical(participant.region, categories=REGION_ORDER).codes.astype(
            np.int16
        ),
        n_domains,
    )
    tier = np.tile(
        pd.Categorical(participant.tier, categories=TIER_ORDER).codes.astype(np.int16),
        n_domains,
    )
    role_raw = pd.Categorical(
        participant.team_position, categories=ROLE_ORDER
    ).codes.astype(np.int16)
    role = np.tile(role_raw, n_domains)
    inter_raw = (
        participant.team_position.map(ROLE_INTERDEPENDENCE)
        .map({"Lower": 0, "Medium": 1, "High": 2})
        .fillna(-1)
        .to_numpy(dtype=np.int16)
    )
    inter = np.tile(inter_raw, n_domains)
    patch = np.tile(pd.Categorical(participant.patch).codes.astype(np.int16), n_domains)
    duration = np.tile(zscore(participant.game_duration), n_domains)

    pieces = {
        "domain": _encode(domain),
        "tier": _encode(tier),
        "role": _encode(role),
        "patch": _encode(patch),
        "tier_domain": _encode(tier * 10 + domain),
        "role_domain": _encode(role * 10 + domain),
        "patch_domain": _encode(patch * 10 + domain),
        "region": _encode(region),
        "region_domain": _encode(region * 10 + domain),
        "region_tier": _encode(region * 10 + tier),
        "region_interdependence": _encode(region * 10 + inter),
        "region_tier_domain": _encode((region * 10 + tier) * 10 + domain),
        "region_interdependence_domain": _encode((region * 10 + inter) * 10 + domain),
    }
    duration_domain = np.column_stack(
        [duration, *[duration * (domain == idx) for idx in range(1, n_domains)]]
    )
    baseline = sparse.hstack(
        [
            sparse.csr_matrix(np.ones((len(y), 1))),
            pieces["domain"],
            pieces["tier"],
            pieces["role"],
            pieces["patch"],
            pieces["tier_domain"],
            pieces["role_domain"],
            pieces["patch_domain"],
            sparse.csr_matrix(duration_domain),
        ],
        format="csr",
    )
    blocks = {
        "region_main": pieces["region"],
        "region_by_domain": pieces["region_domain"],
        "region_by_context": sparse.hstack(
            [pieces["region_tier"], pieces["region_interdependence"]], format="csr"
        ),
        "region_by_context_by_domain": sparse.hstack(
            [pieces["region_tier_domain"], pieces["region_interdependence_domain"]],
            format="csr",
        ),
    }
    cache = {}

    def r2_for(subset):
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        x = sparse.hstack([baseline, *[blocks[k] for k in key]], format="csr")
        model = Ridge(alpha=1e-8, solver="lsqr", fit_intercept=False).fit(x, y)
        pred = model.predict(x)
        r2 = 1 - float(np.sum((y - pred) ** 2)) / float(np.sum((y - y.mean()) ** 2))
        cache[key] = (r2, x.shape[1])
        return cache[key]

    all_blocks = list(blocks)
    for r in range(5):
        for combo in itertools.combinations(all_blocks, r):
            r2_for(combo)
    sequence = []
    current = []
    prev = r2_for([])[0]
    sequence.append(
        {
            "step": 0,
            "block": "baseline",
            "r2": prev,
            "delta_r2": np.nan,
            "n_rows": len(y),
            "n_features": r2_for([])[1],
        }
    )
    for step, block in enumerate(all_blocks, 1):
        current.append(block)
        val, nf = r2_for(current)
        sequence.append(
            {
                "step": step,
                "block": block,
                "r2": val,
                "delta_r2": val - prev,
                "n_rows": len(y),
                "n_features": nf,
            }
        )
        prev = val
    sequence = pd.DataFrame(sequence)
    sequence.to_csv(
        DIRS["extended_data"] / "shared_task_context_variance_sequence.csv", index=False
    )
    contrib = {b: [] for b in all_blocks}
    for order in itertools.permutations(all_blocks):
        entered = []
        before = r2_for([])[0]
        for b in order:
            entered.append(b)
            after = r2_for(entered)[0]
            contrib[b].append(after - before)
            before = after
    averaged = pd.DataFrame(
        [
            {
                "block": b,
                "order_averaged_r2_contribution": np.mean(v),
                "min_marginal": np.min(v),
                "max_marginal": np.max(v),
                "n_orders": len(v),
            }
            for b, v in contrib.items()
        ]
    )
    averaged.to_csv(
        DIRS["extended_data"] / "shared_task_context_order_averaged_contributions.csv",
        index=False,
    )
    formal = pd.DataFrame(
        [
            {
                "moderator": "competitive_tier",
                "base_r2": r2_for(
                    ["region_main", "region_by_domain", "region_by_context"]
                )[0],
                "full_r2": r2_for(all_blocks)[0],
                "delta_r2": r2_for(all_blocks)[0]
                - r2_for(["region_main", "region_by_domain", "region_by_context"])[0],
                "term": "region × tier/interdependence × domain combined",
            },
            {
                "moderator": "region_by_domain",
                "base_r2": r2_for(["region_main"])[0],
                "full_r2": r2_for(["region_main", "region_by_domain"])[0],
                "delta_r2": r2_for(["region_main", "region_by_domain"])[0]
                - r2_for(["region_main"])[0],
                "term": "region × domain",
            },
        ]
    )
    formal.to_csv(
        DIRS["tables"] / "context_formal_micro_long_interactions.csv", index=False
    )
    return sequence, averaged, formal


def leave_one_role_out() -> pd.DataFrame:
    pr = pd.read_parquet(PLAYER_ROLE_PATH)
    rows = []
    for excluded in ROLE_ORDER:
        part = (
            pr[pr.team_position.ne(excluded)]
            .groupby(["puuid", "region"], observed=True)[DOMAIN_ORDER]
            .mean()
            .reset_index()
        )
        for measure in DOMAIN_ORDER:
            for a, b in REGION_PAIRS:
                rows.append(
                    {
                        "excluded_role": excluded,
                        "measure": measure,
                        "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                        "cohens_d": cohen_d(
                            part.loc[part.region.eq(a), measure],
                            part.loc[part.region.eq(b), measure],
                        ),
                        "n_players": part.puuid.nunique(),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(DIRS["tables"] / "context_leave_one_role_out.csv", index=False)
    return result


def control_regime_sensitivity() -> pd.DataFrame:
    base = ROOT / "exp_20260708" / "study2" / "data"
    rows = []
    files = {
        "champion_adjusted": "study2_player_residuals_champion_adjusted.csv",
        "post_outcome_adjusted": "study2_player_residuals_post_outcome_adjusted.csv",
        "skill_ecology_adjusted": "study2_player_residuals_skill_ecology_adjusted.csv",
    }
    for regime, name in files.items():
        path = base / name
        if not path.exists():
            continue
        old = pd.read_csv(path)
        frame = pd.DataFrame(index=old.index)
        for metric in METRIC_NAMES:
            col = metric + "_resid_z" if metric + "_resid_z" in old.columns else metric
            if col not in old.columns:
                break
            frame[metric] = old[col]
        else:
            scores = score_from_metric_frame(frame)
            meta = old[["puuid", "region"]].reset_index(drop=True)
            joined = pd.concat([meta, scores.reset_index(drop=True)], axis=1)
            for measure in DOMAIN_ORDER:
                for a, b in REGION_PAIRS:
                    rows.append(
                        {
                            "regime": regime,
                            "measure": measure,
                            "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                            "cohens_d": cohen_d(
                                joined.loc[joined.region.eq(a), measure],
                                joined.loc[joined.region.eq(b), measure],
                            ),
                            "source": "re-scored from prior residuals under new taxonomy",
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(
        DIRS["extended_data"] / "control_regime_sensitivity_new_taxonomy.csv",
        index=False,
    )
    return result


def recent_patch_sensitivity() -> pd.DataFrame:
    participant = pd.read_parquet(SCORES_PATH)
    patches = sorted(
        participant.patch.unique(), key=lambda x: tuple(map(int, str(x).split(".")))
    )[-10:]
    part = (
        participant[participant.patch.isin(patches)]
        .groupby(["puuid", "region"], observed=True)[DOMAIN_ORDER]
        .mean()
        .reset_index()
    )
    rows = []
    for measure in DOMAIN_ORDER:
        for a, b in REGION_PAIRS:
            rows.append(
                {
                    "measure": measure,
                    "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                    "cohens_d": cohen_d(
                        part.loc[part.region.eq(a), measure],
                        part.loc[part.region.eq(b), measure],
                    ),
                    "patches": ",".join(map(str, patches)),
                    "n_players": part.puuid.nunique(),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(
        DIRS["extended_data"] / "recent_patch_regional_effects.csv", index=False
    )
    return result


def classifier_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = pd.read_pickle(CANONICAL)[["match_id", "puuid", "region"]]
    residuals = pd.read_parquet(RESIDUALS_PATH)
    singleton = np.load(DIRS["data"] / "singleton_match_mask.npy")
    joined = pd.concat(
        [canonical.reset_index(drop=True), residuals.reset_index(drop=True)], axis=1
    ).loc[singleton]
    player = (
        joined.groupby(["puuid", "region"], observed=True)[METRIC_NAMES]
        .mean()
        .reset_index()
    )
    dictionary = pd.DataFrame(METRICS).set_index("metric")
    sets = {
        "all_process_metrics": [
            m for m in METRIC_NAMES if dictionary.loc[m, "role"] == "process_indicator"
        ],
        "shared_information_maintenance": SUBCOMPONENT_METRICS[
            "shared_information_maintenance"
        ],
        "team_signalling": [
            m for subtype in SIGNAL_SUBTYPE_ORDER for m in SUBCOMPONENT_METRICS[subtype]
        ],
        "task_state_attention_signals": SUBCOMPONENT_METRICS[
            "task_state_attention_signals"
        ],
        "coordination_request_signals": SUBCOMPONENT_METRICS[
            "coordination_request_signals"
        ],
        "action_oriented_signals": SUBCOMPONENT_METRICS["action_oriented_signals"],
        "joint_action_participation": SUBCOMPONENT_METRICS[
            "joint_action_participation"
        ],
        "combat_economy_performance": SUBCOMPONENT_METRICS[
            "combat_economy_performance"
        ],
    }
    train_idx, test_idx = train_test_split(
        np.arange(len(player)),
        test_size=0.25,
        random_state=MASTER_SEED,
        stratify=player.region,
    )
    y_train = player.region.iloc[train_idx].to_numpy()
    y_test = player.region.iloc[test_idx].to_numpy()
    rows = []
    importance_rows = []
    for label, features in sets.items():
        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=MASTER_SEED,
            n_jobs=-1,
        ).fit(player.iloc[train_idx][features], y_train)
        pred = model.predict(player.iloc[test_idx][features])
        ba = balanced_accuracy_score(y_test, pred)
        generator = rng(f"classifier_label_permutation_{label}")
        null = np.empty(1000, dtype=np.float32)
        for i in range(1000):
            null[i] = balanced_accuracy_score(generator.permutation(y_test), pred)
        p = (1 + np.sum(null >= ba)) / 1001
        rows.append(
            {
                "feature_set": label,
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": ba,
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "permutation_p": p,
                "n_features": len(features),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_singleton_players": len(player),
                "random_seed": MASTER_SEED,
            }
        )
        if label == "all_process_metrics":
            imp = permutation_importance(
                model,
                player.iloc[test_idx][features],
                y_test,
                scoring="balanced_accuracy",
                n_repeats=20,
                random_state=MASTER_SEED,
                n_jobs=-1,
            )
            importance_rows.extend(
                {
                    "feature": f,
                    "importance_mean": m,
                    "importance_sd": s,
                    "family": dictionary.loc[f, "family"],
                    "subtype": dictionary.loc[f, "subtype"],
                }
                for f, m, s in zip(features, imp.importances_mean, imp.importances_std)
            )
    performance = pd.DataFrame(rows)
    importance = pd.DataFrame(importance_rows).sort_values(
        "importance_mean", ascending=False
    )
    performance.to_csv(
        DIRS["extended_data"] / "singleton_classifier_performance.csv", index=False
    )
    importance.to_csv(
        DIRS["extended_data"] / "singleton_classifier_permutation_importance.csv",
        index=False,
    )
    return performance, importance


def residual_correlation_diagnostics() -> dict:
    player = pd.read_parquet(PLAYER_PATH)
    pearson = player[METRIC_NAMES].corr()
    spearman = player[METRIC_NAMES].corr(method="spearman")
    pearson.to_csv(DIRS["extended_data"] / "player_residual_correlation_pearson.csv")
    spearman.to_csv(DIRS["extended_data"] / "player_residual_correlation_spearman.csv")
    regional = {}
    for region in REGION_ORDER:
        mat = player[player.region.eq(region)][METRIC_NAMES].corr()
        mat.to_csv(
            DIRS["extended_data"]
            / f"player_residual_correlation_{REGION_LABELS[region]}.csv"
        )
        regional[region] = mat
    diffs = []
    for a, b in REGION_PAIRS:
        mat = regional[a] - regional[b]
        for i, m1 in enumerate(METRIC_NAMES):
            for m2 in METRIC_NAMES[i + 1 :]:
                diffs.append(
                    {
                        "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                        "metric_a": m1,
                        "metric_b": m2,
                        "correlation_difference": mat.loc[m1, m2],
                    }
                )
    pd.DataFrame(diffs).to_csv(
        DIRS["extended_data"] / "player_residual_correlation_region_differences.csv",
        index=False,
    )
    adjusted = player[METRIC_NAMES] - player.groupby("region", observed=True)[
        METRIC_NAMES
    ].transform("mean")
    adjusted.corr().to_csv(
        DIRS["extended_data"] / "player_residual_correlation_region_adjusted.csv"
    )
    return {
        "n_players": len(player),
        "max_abs_region_difference": float(
            pd.DataFrame(diffs).correlation_difference.abs().max()
        ),
    }


def legacy_and_general_indices() -> tuple[pd.DataFrame, pd.DataFrame]:
    player = pd.read_parquet(PLAYER_PATH)
    dictionary = pd.DataFrame(METRICS).set_index("metric")
    rows = []
    for legacy, metrics in LEGACY_FOUR_CHANNELS.items():
        score = zscore(player[metrics].mean(axis=1))
        temp = pd.DataFrame({"region": player.region, "score": score})
        for a, b in REGION_PAIRS:
            rows.append(
                {
                    "legacy_channel": legacy,
                    "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                    "cohens_d": cohen_d(
                        temp.loc[temp.region.eq(a), "score"],
                        temp.loc[temp.region.eq(b), "score"],
                    ),
                }
            )
    legacy = pd.DataFrame(rows)
    legacy.to_csv(
        DIRS["extended_data"] / "legacy_four_channel_sensitivity.csv", index=False
    )
    player["cooperation_related_process_engagement_index"] = zscore(
        player[["shared_information_maintenance", "team_signalling"]].mean(axis=1)
    )
    rows = []
    for a, b in REGION_PAIRS:
        rows.append(
            {
                "contrast": f"{REGION_LABELS[a]}-{REGION_LABELS[b]}",
                "cohens_d": cohen_d(
                    player.loc[
                        player.region.eq(a),
                        "cooperation_related_process_engagement_index",
                    ],
                    player.loc[
                        player.region.eq(b),
                        "cooperation_related_process_engagement_index",
                    ],
                ),
                "interpretation": "descriptive two-process formative index; not general cooperativeness",
            }
        )
    general = pd.DataFrame(rows)
    general.to_csv(
        DIRS["extended_data"] / "cooperation_related_process_engagement_index.csv",
        index=False,
    )
    return legacy, general


def draw_figure4(tier, role):
    set_figure_style()
    fig = plt.figure(figsize=(7.35, 5.1))
    outer = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.72, 1.72],
        height_ratios=[1, 1],
        left=0.075,
        right=0.99,
        bottom=0.105,
        top=0.80,
        wspace=0.34,
        hspace=0.52,
    )

    # Each row has three evidence groups: the parent information domain, its
    # two nested components, and two non-information boundary domains.
    axes = {}
    axes["a"] = [fig.add_subplot(outer[0, 0])]
    sub_b = outer[0, 1].subgridspec(1, 2, wspace=0.16)
    axes["b"] = [fig.add_subplot(sub_b[0, 0]), fig.add_subplot(sub_b[0, 1])]
    sub_c = outer[0, 2].subgridspec(1, 2, wspace=0.16)
    axes["c"] = [fig.add_subplot(sub_c[0, 0]), fig.add_subplot(sub_c[0, 1])]
    axes["d"] = [fig.add_subplot(outer[1, 0])]
    sub_e = outer[1, 1].subgridspec(1, 2, wspace=0.16)
    axes["e"] = [fig.add_subplot(sub_e[0, 0]), fig.add_subplot(sub_e[0, 1])]
    sub_f = outer[1, 2].subgridspec(1, 2, wspace=0.16)
    axes["f"] = [fig.add_subplot(sub_f[0, 0]), fig.add_subplot(sub_f[0, 1])]

    measure_groups = {
        "a": ["information_support_practices"],
        "b": [
            "persistent_shared_information_maintenance",
            "event_based_informational_signalling",
        ],
        "c": ["action_coordination_practices", "skill_performance_benchmark"],
        "d": ["information_support_practices"],
        "e": [
            "persistent_shared_information_maintenance",
            "event_based_informational_signalling",
        ],
        "f": ["action_coordination_practices", "skill_performance_benchmark"],
    }
    titles = {
        "information_support_practices": "Information-support\npractices",
        "persistent_shared_information_maintenance": "Persistent shared-\ninformation maintenance",
        "event_based_informational_signalling": "Event-based\ninformational signalling",
        "action_coordination_practices": "Action-coordination\npractices",
        "skill_performance_benchmark": "Skill-performance\nbenchmark",
    }

    def trajectory(ax, data, measure, xcol, xticks, xticklabels):
        for contrast, color in REGION_PAIR_COLORS.items():
            g = data[
                (data.measure.eq(measure)) & (data.contrast.eq(contrast))
            ].sort_values(xcol)
            if g.empty:
                continue
            xv = g[xcol].to_numpy(float)
            yv = g.cohens_d.to_numpy(float)
            ax.fill_between(
                xv,
                g.ci_low.to_numpy(float),
                g.ci_high.to_numpy(float),
                color=color,
                alpha=0.075,
                lw=0,
            )
            ax.plot(
                xv,
                yv,
                marker="o",
                ms=2.7,
                lw=1.15,
                color=color,
                label=contrast,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
        ax.axhline(0, color="#8F8F8F", lw=0.6)
        ax.set_xticks(xticks, xticklabels)
        ax.set_ylim(-0.22, 1.31)
        ax.set_title(titles[measure], fontsize=6.35, pad=3)
        ax.grid(axis="y", color="#EAEAEA", lw=0.45)
        ax.spines["left"].set_color("#777777")
        ax.spines["bottom"].set_color("#777777")

    for key in ["a", "b", "c"]:
        for ax, measure in zip(axes[key], measure_groups[key]):
            trajectory(
                ax,
                tier,
                measure,
                "tier_num",
                range(1, 8),
                ["I", "B", "S", "G", "P", "E", "D"],
            )
            ax.set_xlabel("Competitive tier")
    for key in ["d", "e", "f"]:
        for ax, measure in zip(axes[key], measure_groups[key]):
            trajectory(
                ax, role, measure, "level_num", [1, 2, 3], ["Lower", "Medium", "High"]
            )
            ax.set_xlabel("Role ecology")

    # Only the first axis in each row carries numeric y labels; the common scale
    # keeps the selective amplification directly comparable across domains.
    axes["a"][0].set_ylabel("Signed Cohen's d")
    axes["d"][0].set_ylabel("Signed Cohen's d")
    for key in ["b", "c", "e", "f"]:
        for ax in axes[key]:
            ax.tick_params(labelleft=False)

    def group_title(ax_list, text):
        left = ax_list[0].get_position().x0
        right = ax_list[-1].get_position().x1
        y = max(ax.get_position().y1 for ax in ax_list) + 0.070
        fig.text(
            (left + right) / 2,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="bold",
            color="#333333",
        )

    group_title(axes["a"], "Primary information domain")
    group_title(axes["b"], "Nested information-support components")
    group_title(axes["c"], "Boundary domains")

    handles, labels = axes["a"][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        handlelength=2.3,
        columnspacing=1.5,
    )
    for label in "abcdef":
        ax = axes[label][0]
        ax.text(
            -0.19,
            1.16 if label in "abc" else 1.08,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=9,
            va="bottom",
        )
    save_figure(fig, DIRS["figures"] / "figure4_contextual_variation_by_tier_and_role")


def main():
    ensure_dirs()
    started = time.time()
    for p in [
        SCORES_PATH,
        RESIDUALS_PATH,
        TRANSFORMED_PATH,
        PLAYER_PATH,
        PLAYER_ROLE_PATH,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)
    sequence, averaged, formal = direct_context_and_variance_models()
    leave = pd.DataFrame()
    control = control_regime_sensitivity()
    recent = pd.DataFrame()
    classifier, importance = classifier_diagnostic()
    corr = residual_correlation_diagnostics()
    legacy, general = legacy_and_general_indices()
    summary = {
        "master_seed": MASTER_SEED,
        "formal_models": formal.to_dict("records"),
        "variance_sequence": sequence.to_dict("records"),
        "order_averaged": averaged.to_dict("records"),
        "classifier": classifier.to_dict("records"),
        "residual_correlations": corr,
        "descriptive_process_index": general.to_dict("records"),
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(
        DIRS["reports"] / "context_extended_machine_readable_summary.json", summary
    )
    print(
        json.dumps(
            {"status": "complete", "elapsed_minutes": summary["elapsed_minutes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
