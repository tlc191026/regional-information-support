"""Study 0: CHB-style measurement validation and construct modelling.

This script reproduces the quantitative part of the CHB reference design
where the available database supports it, then extends Study 0 with an
explicit measurement-model check for signalling constructs.

Outputs are written under outputs/study0/.
"""

from __future__ import annotations

import json
import math
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from semopy import Model, calc_stats
except Exception:  # pragma: no cover - handled in runtime report
    Model = None
    calc_stats = None


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "lol_data_20260524_145052_before_matchdto_timeline.db"
OUT = ROOT / "outputs" / "study0"
DATA = OUT / "data"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

RNG_SEED = 20260601
SAMPLE_PER_REGION_TIER = 6000
MIN_ELIGIBLE_MATCHES_PER_PLAYER = 20
CANONICAL_MATCHES_PER_PLAYER = 20
SQL_BATCH_SIZE = 500
EFA_SAMPLE_N = 31500
CFA_SAMPLE_N = 31500
USE_CACHE = True
PARTICIPANT_MODEL_SAMPLE_N = 250000
PROCESSING_VERSION = "ge20_balanced6000_canonical20_v2"

REGIONS = ["NA1", "EUW1", "KR"]
REGION_DISPLAY = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}


def display_region(region: str) -> str:
    return REGION_DISPLAY.get(str(region), str(region))


TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
TIER_NUM = {tier: i for i, tier in enumerate(TIERS, start=1)}
DIVISION_NUM = {"IV": 1, "III": 2, "II": 3, "I": 4}
STANDARD_ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
SAMPLE_AUDIT_PLAYER_COUNTS = (
    ROOT
    / "outputs"
    / "sample_audit"
    / "tables"
    / "sample_audit_player_eligible_match_counts.csv"
)

CHB_PINGS = [
    "all_in_pings",
    "assist_me_pings",
    "enemy_missing_pings",
    "enemy_vision_pings",
    "get_back_pings",
    "need_vision_pings",
    "on_my_way_pings",
    "push_pings",
]

PING_MAP = {
    "strategic": [
        "on_my_way_pings",
        "enemy_vision_pings",
        "enemy_missing_pings",
        "push_pings",
        "command_pings",
    ],
    "collaborative": [
        "assist_me_pings",
        "all_in_pings",
        "need_vision_pings",
        "vision_cleared_pings",
    ],
    "reactive": [
        "danger_pings",
        "get_back_pings",
        "hold_pings",
        "bait_pings",
        "basic_pings",
    ],
}

OBJECTIVE_METRICS = [
    "first_baron",
    "first_dragon",
    "first_rift_herald",
    "first_tower",
    "first_inhibitor",
]

ACTIVITY_METRICS = [
    "kills_pm",
    "deaths_pm",
    "assists_pm",
    "wards_placed_pm",
    "wards_killed_pm",
    "control_wards_bought_pm",
    "vision_score_pm",
    "gold_earned_pm",
    "total_damage_dealt_pm",
    "total_damage_taken_pm",
]

CHB_COMMUNICATION_METRICS = [f"{x}_pm" for x in CHB_PINGS] + [
    "strategic_pings_pm",
    "total_pings_pm",
]

COMPOSITE_METRICS = [
    "vision_control",
    "combat_output",
    "strategic_signal",
    "collaborative_signal",
    "reactive_signal",
    "objective_control",
]

METRIC_LABELS = {
    "first_baron": "First Baron",
    "first_dragon": "First Dragon",
    "first_rift_herald": "First Herald",
    "first_tower": "First Tower",
    "first_inhibitor": "First Inhibitor",
    "kills_pm": "Kills/min",
    "deaths_pm": "Deaths/min",
    "assists_pm": "Assists/min",
    "wards_placed_pm": "Wards placed/min",
    "wards_killed_pm": "Wards cleared/min",
    "control_wards_bought_pm": "Control wards/min",
    "vision_score_pm": "Vision score/min",
    "gold_earned_pm": "Gold/min",
    "total_damage_dealt_pm": "Damage dealt/min",
    "total_damage_taken_pm": "Damage taken/min",
    "all_in_pings_pm": "All-in pings/min",
    "assist_me_pings_pm": "Assist-me pings/min",
    "bait_pings_pm": "Bait pings/min",
    "basic_pings_pm": "Basic pings/min",
    "command_pings_pm": "Command pings/min",
    "danger_pings_pm": "Danger pings/min",
    "enemy_missing_pings_pm": "Enemy-missing pings/min",
    "enemy_vision_pings_pm": "Enemy-vision pings/min",
    "get_back_pings_pm": "Get-back pings/min",
    "hold_pings_pm": "Hold pings/min",
    "need_vision_pings_pm": "Need-vision pings/min",
    "on_my_way_pings_pm": "On-my-way pings/min",
    "push_pings_pm": "Push pings/min",
    "vision_cleared_pings_pm": "Vision-cleared pings/min",
    "strategic_pings_pm": "Strategic pings/min",
    "total_pings_pm": "Total pings/min",
    "vision_control": "Vision control",
    "combat_output": "Combat output",
    "strategic_signal": "Strategic signal",
    "collaborative_signal": "Collaborative signal",
    "reactive_signal": "Reactive signal",
    "objective_control": "Objective control",
}

METRIC_DOMAIN = {
    **{m: "objective_coordination" for m in OBJECTIVE_METRICS},
    "kills_pm": "combat_control",
    "deaths_pm": "combat_control",
    "gold_earned_pm": "combat_control",
    "total_damage_dealt_pm": "combat_control",
    "total_damage_taken_pm": "combat_control",
    "assists_pm": "backup_behavior",
    "wards_placed_pm": "information_support",
    "wards_killed_pm": "information_support",
    "control_wards_bought_pm": "information_support",
    "vision_score_pm": "information_support",
    **{f"{m}_pm": "communication" for m in CHB_PINGS},
    "strategic_pings_pm": "strategic_communication",
    "total_pings_pm": "communication",
    "vision_control": "information_support",
    "combat_output": "combat_control",
    "strategic_signal": "strategic_communication",
    "collaborative_signal": "collaborative_signal",
    "reactive_signal": "reactive_signal",
    "objective_control": "objective_coordination",
}


@dataclass
class FitResult:
    model: str
    status: str
    n: int
    metrics: dict[str, float | str | None]


def ensure_dirs() -> None:
    for path in [DATA, TABLES, FIGURES, REPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def sqlite_connection() -> sqlite3.Connection:
    uri = f"file:{DB_PATH.resolve().as_posix()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=60)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def load_player_eligibility_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """Return one row per source player with observable eligible-match counts."""
    if SAMPLE_AUDIT_PLAYER_COUNTS.exists():
        players = pd.read_csv(SAMPLE_AUDIT_PLAYER_COUNTS)
        required = {
            "puuid",
            "region",
            "tier",
            "division",
            "wins",
            "losses",
            "collected_matches",
            "eligible_distinct_matches",
        }
        if required.issubset(players.columns):
            return players

    query = """
    WITH target_players AS (
        SELECT puuid, region, tier, division, wins, losses, collected_matches, match_collection_done
        FROM players
        WHERE region IN ('NA1', 'EUW1', 'KR')
          AND tier IN ('IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND')
    )
    SELECT
        tp.puuid,
        tp.region,
        tp.tier,
        tp.division,
        tp.wins,
        tp.losses,
        tp.collected_matches,
        tp.match_collection_done,
        COUNT(DISTINCT m.match_id) AS eligible_distinct_matches
    FROM target_players AS tp
    LEFT JOIN match_participants AS mp INDEXED BY idx_participants_puuid
      ON mp.puuid = tp.puuid
    LEFT JOIN matches AS m
      ON m.match_id = mp.match_id
     AND m.region = tp.region
     AND m.queue_id = 420
     AND m.game_duration >= 900
    GROUP BY tp.puuid, tp.region, tp.tier, tp.division, tp.wins, tp.losses, tp.collected_matches, tp.match_collection_done
    """
    return pd.read_sql_query(query, con)


def sample_design_matches(sample: pd.DataFrame) -> bool:
    required = {"puuid", "region", "tier", "eligible_distinct_matches"}
    if not required.issubset(sample.columns):
        return False
    if len(sample) != SAMPLE_PER_REGION_TIER * len(REGIONS) * len(TIERS):
        return False
    if sample["eligible_distinct_matches"].min() < MIN_ELIGIBLE_MATCHES_PER_PLAYER:
        return False
    counts = sample.groupby(["region", "tier"], observed=True).size()
    expected_index = pd.MultiIndex.from_product(
        [REGIONS, TIERS], names=["region", "tier"]
    )
    counts = counts.reindex(expected_index, fill_value=0)
    return bool((counts == SAMPLE_PER_REGION_TIER).all())


def processing_version_matches() -> bool:
    path = TABLES / "study0_processing_design.csv"
    if not path.exists():
        return False
    try:
        design = pd.read_csv(path)
    except Exception:
        return False
    if design.empty or "processing_version" not in design.columns:
        return False
    return str(design.iloc[0]["processing_version"]) == PROCESSING_VERSION


def write_processing_version(rows: pd.DataFrame, players: pd.DataFrame) -> None:
    duplicate_path = TABLES / "study0_participant_duplicate_rows_audit.csv"
    rows_removed = np.nan
    if duplicate_path.exists():
        duplicate_audit = pd.read_csv(duplicate_path)
        if not duplicate_audit.empty and "rows_removed" in duplicate_audit.columns:
            rows_removed = int(duplicate_audit.iloc[0]["rows_removed"])
    pd.DataFrame(
        [
            {
                "processing_version": PROCESSING_VERSION,
                "participant_deduplication_key": "match_id + puuid",
                "participant_rows_after_processing": int(len(rows)),
                "player_rows_after_processing": int(len(players)),
                "duplicate_rows_removed": rows_removed,
            }
        ]
    ).to_csv(TABLES / "study0_processing_design.csv", index=False)


def load_and_sample_players(con: sqlite3.Connection) -> pd.DataFrame:
    players = load_player_eligibility_frame(con)
    players["tier_num"] = players["tier"].map(TIER_NUM)
    players["division_num"] = players["division"].map(DIVISION_NUM)
    players["eligible_distinct_matches"] = (
        players["eligible_distinct_matches"].fillna(0).astype(int)
    )
    eligible = players[
        players["region"].isin(REGIONS)
        & players["tier"].isin(TIERS)
        & players["eligible_distinct_matches"].ge(MIN_ELIGIBLE_MATCHES_PER_PLAYER)
    ].copy()

    rng = np.random.default_rng(RNG_SEED)
    sampled = []
    availability_rows = []
    for (region, tier), group in eligible.groupby(["region", "tier"], sort=False):
        take = SAMPLE_PER_REGION_TIER
        availability_rows.append(
            {
                "region": region,
                "tier": tier,
                "n_available_players": len(group),
                "n_sampled_players": min(take, len(group)),
                "min_eligible_matches_per_player": MIN_ELIGIBLE_MATCHES_PER_PLAYER,
            }
        )
        if len(group) < take:
            raise ValueError(
                f"Not enough eligible players for {region} {tier}: "
                f"{len(group):,} available, {take:,} requested."
            )
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        sampled.append(group.loc[idx])
    sample = pd.concat(sampled, ignore_index=True)
    sample = sample.sort_values(
        ["region", "tier_num", "division_num", "puuid"]
    ).reset_index(drop=True)
    sample.to_csv(DATA / "study0_sampled_players.csv", index=False)

    counts = (
        sample.groupby(["region", "tier"], observed=True)
        .size()
        .reset_index(name="n_sampled_players")
        .sort_values(["region", "tier"])
    )
    availability = pd.DataFrame(availability_rows)
    counts = counts.merge(
        availability, on=["region", "tier", "n_sampled_players"], how="left"
    )
    counts.to_csv(TABLES / "study0_sampling_counts.csv", index=False)

    division_counts = (
        sample.groupby(["region", "tier", "division"], observed=True)
        .size()
        .reset_index(name="n_sampled_players")
        .sort_values(["region", "tier", "division"])
    )
    division_counts.to_csv(
        TABLES / "study0_sampling_counts_by_division.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "sampling_unit": "player",
                "strata": "region x tier",
                "n_per_region_tier": SAMPLE_PER_REGION_TIER,
                "min_eligible_matches_per_player": MIN_ELIGIBLE_MATCHES_PER_PLAYER,
                "total_sampled_players": len(sample),
                "random_seed": RNG_SEED,
            }
        ]
    ).to_csv(TABLES / "study0_sampling_design.csv", index=False)
    return sample


def batched(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def load_participant_rows(
    con: sqlite3.Connection, sample: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "match_id",
        "puuid",
        "team_id",
        "team_position",
        "champion_id",
        "champion_name",
        "kills",
        "deaths",
        "assists",
        "gold_earned",
        "total_damage_dealt",
        "total_damage_taken",
        "vision_score",
        "wards_placed",
        "wards_killed",
        "control_wards_bought",
        "win",
        "all_in_pings",
        "assist_me_pings",
        "bait_pings",
        "basic_pings",
        "command_pings",
        "danger_pings",
        "enemy_missing_pings",
        "enemy_vision_pings",
        "hold_pings",
        "need_vision_pings",
        "on_my_way_pings",
        "push_pings",
        "vision_cleared_pings",
        "get_back_pings",
        "region",
        "game_duration",
        "game_version",
        "blue_team_win",
        "first_baron",
        "first_dragon",
        "first_rift_herald",
        "first_tower",
        "first_inhibitor",
        "baron_kills",
        "dragon_kills",
        "rift_herald_kills",
        "tower_kills",
        "inhibitor_kills",
    ]
    frames = []
    sample_by_region = {
        region: group["puuid"].tolist()
        for region, group in sample.groupby("region", sort=False)
    }
    for region, puuids in sample_by_region.items():
        for batch_no, batch in enumerate(batched(puuids, SQL_BATCH_SIZE), start=1):
            placeholders = ",".join("?" for _ in batch)
            query = f"""
            SELECT
                mp.match_id, mp.puuid, mp.team_id, mp.team_position,
                mp.champion_id, mp.champion_name, mp.kills, mp.deaths, mp.assists,
                mp.gold_earned, mp.total_damage_dealt, mp.total_damage_taken,
                mp.vision_score, mp.wards_placed, mp.wards_killed,
                mp.control_wards_bought, mp.win,
                mp.all_in_pings, mp.assist_me_pings, mp.bait_pings,
                mp.basic_pings, mp.command_pings, mp.danger_pings,
                mp.enemy_missing_pings, mp.enemy_vision_pings, mp.hold_pings,
                mp.need_vision_pings, mp.on_my_way_pings, mp.push_pings,
                mp.vision_cleared_pings, mp.get_back_pings,
                m.region, m.game_duration, m.game_version, m.blue_team_win,
                obj.first_baron, obj.first_dragon, obj.first_rift_herald,
                obj.first_tower, obj.first_inhibitor,
                obj.baron_kills, obj.dragon_kills, obj.rift_herald_kills,
                obj.tower_kills, obj.inhibitor_kills
            FROM match_participants AS mp INDEXED BY idx_participants_puuid
            JOIN matches AS m
              ON m.match_id = mp.match_id
            LEFT JOIN team_objectives AS obj
              ON obj.match_id = mp.match_id AND obj.team_id = mp.team_id
            WHERE mp.puuid IN ({placeholders})
              AND m.region = ?
              AND m.queue_id = 420
              AND m.game_duration >= 900
            """
            params = [*batch, region]
            chunk = pd.read_sql_query(query, con, params=params)
            if not chunk.empty:
                frames.append(chunk)
            if batch_no % 20 == 0:
                print(
                    f"Loaded {region} batch {batch_no}/{math.ceil(len(puuids) / SQL_BATCH_SIZE)}"
                )

    if not frames:
        raise RuntimeError(
            "No participant rows were loaded. Check source database filters."
        )
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[columns]
    duplicate_mask = rows.duplicated(["match_id", "puuid"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = (
        int(rows.loc[duplicate_mask, ["match_id", "puuid"]].drop_duplicates().shape[0])
        if duplicate_rows
        else 0
    )
    if duplicate_rows:
        rows = rows.drop_duplicates(["match_id", "puuid"], keep="first").reset_index(
            drop=True
        )
    pd.DataFrame(
        [
            {
                "deduplication_key": "match_id + puuid",
                "duplicate_rows_before_dedup": duplicate_rows,
                "duplicate_player_match_keys": duplicate_keys,
                "rows_removed": duplicate_rows - duplicate_keys,
                "rows_after_dedup": int(len(rows)),
            }
        ]
    ).to_csv(TABLES / "study0_participant_duplicate_rows_audit.csv", index=False)
    return rows


def add_derived_metrics(rows: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = rows.merge(
        sample[["puuid", "tier", "division", "tier_num", "division_num"]],
        on="puuid",
        how="left",
        validate="many_to_one",
    )
    rows = rows.dropna(subset=["tier_num"]).copy()
    rows["duration_min"] = rows["game_duration"] / 60.0
    rows["patch"] = rows["game_version"].str.extract(r"^(\d+\.\d+)", expand=False)

    count_metrics = [
        "kills",
        "deaths",
        "assists",
        "gold_earned",
        "total_damage_dealt",
        "total_damage_taken",
        "vision_score",
        "wards_placed",
        "wards_killed",
        "control_wards_bought",
        *CHB_PINGS,
        "bait_pings",
        "basic_pings",
        "command_pings",
        "danger_pings",
        "hold_pings",
        "vision_cleared_pings",
        "get_back_pings",
    ]
    count_metrics = list(dict.fromkeys(count_metrics))
    for metric in count_metrics:
        rows[f"{metric}_pm"] = rows[metric] / rows["duration_min"]

    all_ping_raw = [
        "all_in_pings",
        "assist_me_pings",
        "bait_pings",
        "basic_pings",
        "command_pings",
        "danger_pings",
        "enemy_missing_pings",
        "enemy_vision_pings",
        "hold_pings",
        "need_vision_pings",
        "on_my_way_pings",
        "push_pings",
        "vision_cleared_pings",
        "get_back_pings",
    ]
    rows["total_pings"] = rows[all_ping_raw].sum(axis=1)
    rows["total_pings_pm"] = rows["total_pings"] / rows["duration_min"]
    for category, metrics in PING_MAP.items():
        rows[f"{category}_signal_raw"] = rows[metrics].sum(axis=1)
        rows[f"{category}_signal_pm"] = (
            rows[f"{category}_signal_raw"] / rows["duration_min"]
        )
    rows["strategic_pings"] = rows[
        ["on_my_way_pings", "enemy_vision_pings", "enemy_missing_pings"]
    ].sum(axis=1)
    rows["strategic_pings_pm"] = rows["strategic_pings"] / rows["duration_min"]

    objective_cols = [
        "first_baron",
        "first_dragon",
        "first_rift_herald",
        "first_tower",
        "first_inhibitor",
    ]
    for col in objective_cols:
        rows[col] = rows[col].fillna(0).astype(float)
    rows["objective_control"] = rows[objective_cols].mean(axis=1)

    rows["vision_control"] = rows[
        [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
        ]
    ].mean(axis=1)
    rows["combat_output"] = rows[
        ["kills_pm", "gold_earned_pm", "total_damage_dealt_pm"]
    ].mean(axis=1)
    rows["strategic_signal"] = rows["strategic_signal_pm"]
    rows["collaborative_signal"] = rows["collaborative_signal_pm"]
    rows["reactive_signal"] = rows["reactive_signal_pm"]
    return rows


def player_aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = (
        OBJECTIVE_METRICS
        + ACTIVITY_METRICS
        + CHB_COMMUNICATION_METRICS
        + [
            f"{c}_pm"
            for c in [
                "bait_pings",
                "basic_pings",
                "command_pings",
                "danger_pings",
                "hold_pings",
                "vision_cleared_pings",
                "get_back_pings",
            ]
        ]
        + [
            "strategic_signal_pm",
            "collaborative_signal_pm",
            "reactive_signal_pm",
            "vision_control",
            "combat_output",
            "strategic_signal",
            "collaborative_signal",
            "reactive_signal",
            "objective_control",
        ]
    )
    metric_cols = list(dict.fromkeys([m for m in metric_cols if m in rows.columns]))
    agg = rows.groupby("puuid", observed=True)[metric_cols].mean()
    meta = rows.groupby("puuid", observed=True).agg(
        region=("region", "first"),
        tier=("tier", "first"),
        division=("division", "first"),
        tier_num=("tier_num", "first"),
        division_num=("division_num", "first"),
        n_matches=("match_id", "nunique"),
        primary_role=(
            "team_position",
            lambda s: s.mode().iat[0] if not s.mode().empty else "",
        ),
        n_champions=("champion_id", "nunique"),
        n_patches=("patch", "nunique"),
    )
    out = meta.join(agg).reset_index()
    out.to_csv(DATA / "study0_player_aggregate.csv", index=False)
    return out


def canonical_participant_match_sample(
    rows: pd.DataFrame, sample: pd.DataFrame
) -> pd.DataFrame:
    """Select an equal number of focal participant-match rows per sampled player.

    The raw participant cache can contain more than the API-requested 30 matches
    for some players because matches downloaded for one sampled player can also
    contain another sampled player. The canonical sample equalizes player
    weight before all cross-regional analyses.
    """
    out_path = DATA / "study0_canonical_participant_match_sample.pkl"
    metadata_path = TABLES / "study0_canonical_participant_match_sample_metadata.csv"
    player_counts_path = TABLES / "study0_canonical_player_match_counts.csv"
    region_tier_path = TABLES / "study0_canonical_counts_by_region_tier.csv"
    role_path = TABLES / "study0_canonical_role_counts.csv"
    overlap_path = TABLES / "study0_canonical_match_overlap_distribution.csv"
    overlap_region_path = TABLES / "study0_canonical_match_overlap_by_region.csv"

    sample_puuids = set(sample["puuid"].astype(str))
    required_cols = {"match_id", "puuid", "region", "tier", "team_position"}
    if USE_CACHE and out_path.exists():
        cached = pd.read_pickle(out_path)
        cached_puuids = (
            set(cached["puuid"].astype(str).unique())
            if "puuid" in cached.columns
            else set()
        )
        per_player = (
            cached.groupby("puuid", observed=True)["match_id"].nunique()
            if {"puuid", "match_id"}.issubset(cached.columns)
            else pd.Series(dtype=int)
        )
        valid_cache = (
            required_cols.issubset(cached.columns)
            and cached_puuids == sample_puuids
            and len(cached) == len(sample_puuids) * CANONICAL_MATCHES_PER_PLAYER
            and per_player.size == len(sample_puuids)
            and per_player.eq(CANONICAL_MATCHES_PER_PLAYER).all()
            and not cached.duplicated(["match_id", "puuid"]).any()
            and out_path.stat().st_mtime
            >= (DATA / "study0_participant_rows.pkl").stat().st_mtime
        )
        if valid_cache:
            return cached

    work = rows[rows["puuid"].astype(str).isin(sample_puuids)].copy()
    work = work.drop_duplicates(["match_id", "puuid"], keep="first").reset_index(
        drop=True
    )
    work["team_position"] = (
        work["team_position"].astype(str).replace({"nan": "", "None": ""})
    )
    work["_is_standard_role"] = work["team_position"].isin(STANDARD_ROLES)

    rng = np.random.default_rng(RNG_SEED)
    parts = []
    count_records = []
    for puuid, group in work.groupby("puuid", sort=False, observed=True):
        standard = group[group["_is_standard_role"]].copy()
        nonstandard = group[~group["_is_standard_role"]].copy()
        if len(group) < CANONICAL_MATCHES_PER_PLAYER:
            raise ValueError(
                f"Player {puuid} has only {len(group)} eligible rows; expected at least {CANONICAL_MATCHES_PER_PLAYER}."
            )
        standard = (
            standard.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            if len(standard)
            else standard
        )
        nonstandard = (
            nonstandard.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            if len(nonstandard)
            else nonstandard
        )
        selected = (
            pd.concat([standard, nonstandard], ignore_index=False)
            .head(CANONICAL_MATCHES_PER_PLAYER)
            .copy()
        )
        selected["_canonical_match_rank"] = np.arange(1, len(selected) + 1)
        selected["_canonical_nonstandard_fill"] = ~selected["_is_standard_role"]
        parts.append(selected)
        count_records.append(
            {
                "puuid": puuid,
                "available_rows": int(len(group)),
                "available_standard_role_rows": int(len(standard)),
                "selected_rows": int(len(selected)),
                "selected_nonstandard_rows": int(
                    (~selected["_is_standard_role"]).sum()
                ),
            }
        )

    canonical = pd.concat(parts, ignore_index=True)
    canonical = canonical.drop(columns=["_is_standard_role"])
    canonical.to_pickle(out_path)

    counts = pd.DataFrame(count_records)
    counts = counts.merge(
        sample[["puuid", "region", "tier"]],
        on="puuid",
        how="left",
        validate="one_to_one",
    )
    counts.to_csv(player_counts_path, index=False)

    region_tier = (
        canonical.groupby(["region", "tier"], observed=True)
        .agg(
            n_participant_rows=("match_id", "size"),
            n_players=("puuid", "nunique"),
            n_unique_matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    region_tier.to_csv(region_tier_path, index=False)

    role_counts = (
        canonical.assign(
            team_position=canonical["team_position"].replace({"": "UNKNOWN_ROLE"})
        )
        .groupby(["team_position"], observed=True)
        .size()
        .reset_index(name="n_participant_rows")
    )
    role_counts.to_csv(role_path, index=False)

    match_overlap = (
        canonical.groupby(["region", "match_id"], observed=True)
        .agg(n_focal_players=("puuid", "nunique"))
        .reset_index()
    )
    overlap_dist = (
        match_overlap["n_focal_players"]
        .value_counts()
        .sort_index()
        .rename_axis("sampled_focal_players_per_match")
        .reset_index(name="n_unique_matches")
    )
    overlap_dist.to_csv(overlap_path, index=False)
    overlap_region = (
        match_overlap.groupby("region", observed=True)
        .agg(
            n_unique_matches=("match_id", "nunique"),
            mean_sampled_focal_players_per_match=("n_focal_players", "mean"),
            median_sampled_focal_players_per_match=("n_focal_players", "median"),
            max_sampled_focal_players_per_match=("n_focal_players", "max"),
        )
        .reset_index()
    )
    overlap_region.to_csv(overlap_region_path, index=False)

    metadata = pd.DataFrame(
        [
            {
                "sampling_unit": "participant-match",
                "source": "study0_participant_rows.pkl",
                "n_participant_rows": int(len(canonical)),
                "n_players": int(canonical["puuid"].nunique()),
                "n_unique_matches": int(canonical["match_id"].nunique()),
                "matches_per_player": CANONICAL_MATCHES_PER_PLAYER,
                "random_seed": RNG_SEED,
                "standard_role_rows": int(
                    canonical["team_position"].isin(STANDARD_ROLES).sum()
                ),
                "nonstandard_fill_rows": int(
                    canonical["_canonical_nonstandard_fill"].sum()
                ),
                "selection_rule": "within each sampled player, randomly select 20 rows, prioritising standard role labels and filling rare shortages with non-standard role rows",
            }
        ]
    )
    metadata.to_csv(metadata_path, index=False)
    return canonical


def participant_level_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    objective_records = []
    for region, region_df in rows.groupby("region", observed=True):
        for tier, tier_df in region_df.groupby("tier", observed=True):
            for objective in OBJECTIVE_METRICS:
                first = tier_df[objective].astype(float)
                win = tier_df["win"].astype(float)
                has_obj = first == 1
                no_obj = first == 0
                objective_records.append(
                    {
                        "region": region,
                        "tier": tier,
                        "tier_num": TIER_NUM.get(tier, np.nan),
                        "objective": objective,
                        "label": METRIC_LABELS.get(objective, objective),
                        "n_participant_rows": int(len(tier_df)),
                        "objective_rate": float(first.mean()),
                        "win_rate_when_first": (
                            float(win[has_obj].mean()) if has_obj.any() else np.nan
                        ),
                        "win_rate_without_first": (
                            float(win[no_obj].mean()) if no_obj.any() else np.nan
                        ),
                        "win_rate_delta": (
                            float(win[has_obj].mean() - win[no_obj].mean())
                            if has_obj.any() and no_obj.any()
                            else np.nan
                        ),
                    }
                )
    objective_out = pd.DataFrame(objective_records).sort_values(
        ["region", "objective", "tier_num"]
    )
    objective_out.to_csv(TABLES / "study0_objective_win_leverage.csv", index=False)
    return objective_out


def load_or_build_participant_rows(sample: pd.DataFrame) -> pd.DataFrame:
    cache_path = DATA / "study0_participant_rows.pkl"
    if USE_CACHE and cache_path.exists():
        rows = pd.read_pickle(cache_path)
        sample_puuids = set(sample["puuid"].astype(str))
        cached_puuids = (
            set(rows["puuid"].astype(str).unique())
            if "puuid" in rows.columns
            else set()
        )
        has_duplicate_player_matches = {"match_id", "puuid"}.issubset(
            rows.columns
        ) and rows.duplicated(["match_id", "puuid"]).any()
        if cached_puuids == sample_puuids and not has_duplicate_player_matches:
            return rows
        if cached_puuids != sample_puuids:
            print(
                "Cached participant rows do not match current sample design; rebuilding."
            )
        else:
            print(
                "Cached participant rows contain duplicate match_id + puuid keys; rebuilding."
            )
    with sqlite_connection() as con:
        print("Loading participant rows for Study 0 robustness checks...")
        rows = load_participant_rows(con, sample)
    rows = add_derived_metrics(rows, sample)
    rows.to_pickle(cache_path)
    participant_level_summaries(rows)
    return rows


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / pooled


def bootstrap_d_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 300
) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        aa = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        vals.append(cohen_d(aa, bb))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def standardized_slope_and_r2(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    x = df[["tier_num"]].to_numpy(dtype=float)
    y = np.log1p(df[metric].astype(float).to_numpy())
    mask = np.isfinite(y) & np.isfinite(x[:, 0])
    if mask.sum() < 10 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan
    xz = StandardScaler().fit_transform(x[mask])
    yz = StandardScaler().fit_transform(y[mask].reshape(-1, 1)).ravel()
    model = LinearRegression().fit(xz, yz)
    return float(model.coef_[0]), float(model.score(xz, yz))


def anova_eta2(df: pd.DataFrame, metric: str) -> float:
    y = np.log1p(df[metric].clip(lower=0).astype(float).to_numpy())
    tiers = df["tier"].to_numpy()
    mask = np.isfinite(y)
    y = y[mask]
    tiers = tiers[mask]
    if len(y) < 10:
        return np.nan
    grand = np.mean(y)
    ss_total = np.sum((y - grand) ** 2)
    if ss_total == 0:
        return np.nan
    ss_between = 0.0
    for tier in TIERS:
        group = y[tiers == tier]
        if len(group):
            ss_between += len(group) * (np.mean(group) - grand) ** 2
    return float(ss_between / ss_total)


def monotonicity_score(df: pd.DataFrame, metric: str) -> float:
    means = (
        df.groupby(["tier", "tier_num"], observed=True)[metric]
        .mean()
        .reset_index()
        .sort_values("tier_num")[metric]
        .to_numpy(dtype=float)
    )
    if len(means) < 2 or np.allclose(means[0], means[-1]):
        return np.nan
    direction = np.sign(means[-1] - means[0])
    steps = np.diff(means)
    non_zero = np.abs(steps) > 1e-12
    if not np.any(non_zero):
        return np.nan
    return float(np.mean(np.sign(steps[non_zero]) == direction))


def effect_table(
    players: pd.DataFrame, metrics: list[str], table_name: str
) -> pd.DataFrame:
    records = []
    for region, region_df in players.groupby("region", observed=True):
        for metric in metrics:
            if metric not in region_df.columns:
                continue
            iron = (
                region_df.loc[region_df["tier"] == "IRON", metric].dropna().to_numpy()
            )
            diamond = (
                region_df.loc[region_df["tier"] == "DIAMOND", metric]
                .dropna()
                .to_numpy()
            )
            d_raw = cohen_d(iron, diamond)
            d_log = cohen_d(np.log1p(iron), np.log1p(diamond))
            ci_low, ci_high = bootstrap_d_ci(np.log1p(iron), np.log1p(diamond))
            beta, r2 = standardized_slope_and_r2(region_df, metric)
            eta2 = anova_eta2(region_df, metric)
            mono = monotonicity_score(region_df, metric)
            iron_mean = float(np.nanmean(iron)) if len(iron) else np.nan
            diamond_mean = float(np.nanmean(diamond)) if len(diamond) else np.nan
            pct_change = (
                ((diamond_mean - iron_mean) / iron_mean * 100.0)
                if iron_mean not in [0, np.nan] and iron_mean != 0
                else np.nan
            )
            records.append(
                {
                    "region": region,
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "domain": METRIC_DOMAIN.get(metric, "other"),
                    "n_iron_players": len(iron),
                    "n_diamond_players": len(diamond),
                    "iron_mean": iron_mean,
                    "diamond_mean": diamond_mean,
                    "percent_change_raw": pct_change,
                    "cohen_d_raw": d_raw,
                    "cohen_d_log": d_log,
                    "cohen_d_log_ci_low": ci_low,
                    "cohen_d_log_ci_high": ci_high,
                    "std_beta_tier_log": beta,
                    "r2_tier_log": r2,
                    "anova_eta2_log": eta2,
                    "monotonicity_score": mono,
                }
            )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / table_name, index=False)
    return out


def consecutive_tier_effect_table(
    players: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    records = []
    for region, region_df in players.groupby("region", observed=True):
        for metric in metrics:
            if metric not in region_df.columns:
                continue
            for lower, higher in zip(TIERS[:-1], TIERS[1:]):
                low = (
                    region_df.loc[region_df["tier"] == lower, metric]
                    .dropna()
                    .to_numpy()
                )
                high = (
                    region_df.loc[region_df["tier"] == higher, metric]
                    .dropna()
                    .to_numpy()
                )
                low_log = np.log1p(np.clip(low, 0, None))
                high_log = np.log1p(np.clip(high, 0, None))
                d_log = cohen_d(low_log, high_log)
                ci_low, ci_high = bootstrap_d_ci(low_log, high_log, n_boot=200)
                try:
                    t_stat, p_value = stats.ttest_ind(
                        low_log, high_log, equal_var=False, nan_policy="omit"
                    )
                except Exception:
                    t_stat, p_value = np.nan, np.nan
                records.append(
                    {
                        "region": region,
                        "metric": metric,
                        "label": METRIC_LABELS.get(metric, metric),
                        "domain": METRIC_DOMAIN.get(metric, "other"),
                        "tier_lower": lower,
                        "tier_higher": higher,
                        "comparison": f"{lower}_to_{higher}",
                        "n_lower_players": len(low),
                        "n_higher_players": len(high),
                        "lower_mean": float(np.nanmean(low)) if len(low) else np.nan,
                        "higher_mean": float(np.nanmean(high)) if len(high) else np.nan,
                        "cohen_d_log": d_log,
                        "cohen_d_log_ci_low": ci_low,
                        "cohen_d_log_ci_high": ci_high,
                        "welch_t": float(t_stat) if pd.notna(t_stat) else np.nan,
                        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    }
                )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_consecutive_tier_effects.csv", index=False)
    return out


def domain_effect_summary(all_effects: pd.DataFrame) -> pd.DataFrame:
    out = (
        all_effects.groupby("domain", observed=True)
        .agg(
            n_effects=("cohen_d_log", "count"),
            median_iron_diamond_d=("cohen_d_log", "median"),
            mean_iron_diamond_d=("cohen_d_log", "mean"),
            median_tier_beta=("std_beta_tier_log", "median"),
            median_eta2=("anova_eta2_log", "median"),
        )
        .reset_index()
        .sort_values("median_iron_diamond_d", ascending=False)
    )
    out.to_csv(TABLES / "study0_domain_effect_summary.csv", index=False)
    return out


def omnibus_tier_tests(players: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []
    for region, region_df in players.groupby("region", observed=True):
        for metric in metrics:
            if metric not in region_df.columns:
                continue
            work = region_df[["tier", "tier_num", metric]].dropna().copy()
            work["y"] = np.log1p(work[metric].clip(lower=0).astype(float))
            work = work[np.isfinite(work["y"])]
            if len(work) < 20 or work["y"].nunique() < 2:
                continue

            groups = [work.loc[work["tier"] == tier, "y"].to_numpy() for tier in TIERS]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) > 1:
                f_stat, anova_p = stats.f_oneway(*groups)
            else:
                f_stat, anova_p = np.nan, np.nan
            grand = work["y"].mean()
            ss_total = float(np.sum((work["y"] - grand) ** 2))
            ss_between = 0.0
            for tier in TIERS:
                g = work.loc[work["tier"] == tier, "y"]
                if len(g):
                    ss_between += len(g) * float((g.mean() - grand) ** 2)
            eta2 = ss_between / ss_total if ss_total > 0 else np.nan

            lin = stats.linregress(
                work["tier_num"].astype(float), work["y"].astype(float)
            )
            r = np.corrcoef(work["tier_num"].astype(float), work["y"].astype(float))[
                0, 1
            ]
            linear_r2 = float(r**2) if np.isfinite(r) else np.nan
            y_z = StandardScaler().fit_transform(work[["y"]]).ravel()
            x_z = StandardScaler().fit_transform(work[["tier_num"]]).ravel()
            linear_beta = float(np.corrcoef(x_z, y_z)[0, 1])

            z_tier = x_z
            z_tier2 = (
                StandardScaler()
                .fit_transform(
                    (work["tier_num"].astype(float) ** 2).to_numpy().reshape(-1, 1)
                )
                .ravel()
            )
            x_linear = np.column_stack([np.ones(len(work)), z_tier])
            x_quad = np.column_stack([np.ones(len(work)), z_tier, z_tier2])
            beta_linear = np.linalg.lstsq(x_linear, y_z, rcond=None)[0]
            beta_quad = np.linalg.lstsq(x_quad, y_z, rcond=None)[0]
            resid_linear = y_z - x_linear @ beta_linear
            resid_quad = y_z - x_quad @ beta_quad
            rss_linear = float(np.sum(resid_linear**2))
            rss_quad = float(np.sum(resid_quad**2))
            df1 = 1
            df2 = max(1, len(work) - x_quad.shape[1])
            if rss_quad < rss_linear and rss_quad > 0:
                f_quad = ((rss_linear - rss_quad) / df1) / (rss_quad / df2)
                p_quad = float(stats.f.sf(f_quad, df1, df2))
            else:
                f_quad = 0.0
                p_quad = 1.0
            r2_quad = 1.0 - rss_quad / float(np.sum((y_z - y_z.mean()) ** 2))

            records.append(
                {
                    "region": region,
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "domain": METRIC_DOMAIN.get(metric, "other"),
                    "n_players": len(work),
                    "anova_df_between": len(groups) - 1,
                    "anova_df_within": len(work) - len(groups),
                    "anova_F": float(f_stat),
                    "anova_P": float(anova_p),
                    "anova_eta2": float(eta2),
                    "linear_beta_std": linear_beta,
                    "linear_R2": linear_r2,
                    "linear_P": float(lin.pvalue),
                    "quadratic_beta_std": float(beta_quad[2]),
                    "quadratic_R2": float(r2_quad),
                    "quadratic_delta_R2": float(r2_quad - linear_r2),
                    "quadratic_F_change": float(f_quad),
                    "quadratic_P": float(p_quad),
                }
            )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_omnibus_tier_tests.csv", index=False)
    return out


def chb_replication_table(
    players: pd.DataFrame, metrics: list[str], omnibus: pd.DataFrame
) -> pd.DataFrame:
    effects = effect_table(players, metrics, "study0_chb_replication_effects_tmp.csv")
    cols = [
        "region",
        "metric",
        "linear_R2",
        "linear_P",
        "anova_F",
        "anova_df_between",
        "anova_df_within",
        "anova_P",
        "anova_eta2",
    ]
    out = effects.merge(omnibus[cols], on=["region", "metric"], how="left")
    out = out[
        [
            "region",
            "metric",
            "label",
            "domain",
            "n_iron_players",
            "n_diamond_players",
            "iron_mean",
            "diamond_mean",
            "percent_change_raw",
            "cohen_d_log",
            "cohen_d_log_ci_low",
            "cohen_d_log_ci_high",
            "linear_R2",
            "linear_P",
            "anova_F",
            "anova_df_between",
            "anova_df_within",
            "anova_P",
            "anova_eta2",
            "std_beta_tier_log",
            "monotonicity_score",
        ]
    ]
    out.to_csv(TABLES / "study0_chb_replication_table.csv", index=False)
    tmp_path = TABLES / "study0_chb_replication_effects_tmp.csv"
    if tmp_path.exists():
        tmp_path.unlink()
    return out


def nonmonotonic_signal_tests(players: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "all_in_pings_pm",
        "need_vision_pings_pm",
        "push_pings_pm",
        "get_back_pings_pm",
    ]
    records = []
    for metric in metrics:
        if metric not in players.columns:
            continue
        for region, region_df in players.groupby("region", observed=True):
            work = region_df[["tier_num", metric]].dropna().copy()
            work["y"] = np.log1p(work[metric].clip(lower=0).astype(float))
            work["tier_z"] = StandardScaler().fit_transform(work[["tier_num"]]).ravel()
            work["tier2_z"] = (
                StandardScaler()
                .fit_transform(
                    (work["tier_num"].astype(float) ** 2).to_numpy().reshape(-1, 1)
                )
                .ravel()
            )
            linear = smf.ols("y ~ tier_z", data=work).fit()
            quad = smf.ols("y ~ tier_z + tier2_z", data=work).fit()
            compare = quad.compare_f_test(linear)
            records.append(
                {
                    "scope": "region_stratified",
                    "region": region,
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "n_players": len(work),
                    "linear_beta": float(linear.params.get("tier_z", np.nan)),
                    "linear_P": float(linear.pvalues.get("tier_z", np.nan)),
                    "linear_R2": float(linear.rsquared),
                    "quadratic_beta": float(quad.params.get("tier2_z", np.nan)),
                    "quadratic_P": float(quad.pvalues.get("tier2_z", np.nan)),
                    "quadratic_R2": float(quad.rsquared),
                    "quadratic_delta_R2": float(quad.rsquared - linear.rsquared),
                    "quadratic_F_change": float(compare[0]),
                    "quadratic_F_change_P": float(compare[1]),
                }
            )

        pooled = players[["region", "tier_num", metric]].dropna().copy()
        pooled["y"] = np.log1p(pooled[metric].clip(lower=0).astype(float))
        pooled["tier_z"] = StandardScaler().fit_transform(pooled[["tier_num"]]).ravel()
        pooled["tier2_z"] = (
            StandardScaler()
            .fit_transform(
                (pooled["tier_num"].astype(float) ** 2).to_numpy().reshape(-1, 1)
            )
            .ravel()
        )
        pooled["region"] = pd.Categorical(pooled["region"], categories=REGIONS)
        linear_inter = smf.ols("y ~ C(region) * tier_z", data=pooled).fit()
        quad_inter = smf.ols(
            "y ~ C(region) * tier_z + C(region) * tier2_z", data=pooled
        ).fit()
        compare = quad_inter.compare_f_test(linear_inter)
        records.append(
            {
                "scope": "region_interaction",
                "region": "ALL",
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "n_players": len(pooled),
                "linear_beta": np.nan,
                "linear_P": float(linear_inter.f_pvalue),
                "linear_R2": float(linear_inter.rsquared),
                "quadratic_beta": np.nan,
                "quadratic_P": float(quad_inter.f_pvalue),
                "quadratic_R2": float(quad_inter.rsquared),
                "quadratic_delta_R2": float(
                    quad_inter.rsquared - linear_inter.rsquared
                ),
                "quadratic_F_change": float(compare[0]),
                "quadratic_F_change_P": float(compare[1]),
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_nonmonotonic_signal_tests.csv", index=False)
    return out


def tier_summary(players: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in players.columns:
            continue
        tmp = (
            players.groupby(["region", "tier", "tier_num"], observed=True)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        tmp["metric"] = metric
        tmp["label"] = METRIC_LABELS.get(metric, metric)
        tmp["se"] = tmp["std"] / np.sqrt(tmp["count"])
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLES / "study0_metric_tier_summary.csv", index=False)
    return out


def varimax(
    phi: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6
) -> np.ndarray:
    p, k = phi.shape
    r = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        lam = phi @ r
        u, s, vh = np.linalg.svd(
            phi.T @ (lam**3 - (gamma / p) * lam @ np.diag(np.diag(lam.T @ lam)))
        )
        r = u @ vh
        d = s.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return phi @ r


def kmo_stat(corr: np.ndarray) -> float:
    inv_corr = np.linalg.pinv(corr)
    partial = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial, 0)
    corr_sq = corr.copy() ** 2
    partial_sq = partial**2
    np.fill_diagonal(corr_sq, 0)
    return float(corr_sq.sum() / (corr_sq.sum() + partial_sq.sum()))


def factor_analysis_splits(
    players: pd.DataFrame, variables: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = players[["puuid", "region", "tier", "tier_num", *variables]].dropna().copy()
    for col in variables:
        df[col] = np.log1p(df[col].clip(lower=0))
    valid_variables = [
        col
        for col in variables
        if df[col].nunique(dropna=True) > 2 and float(df[col].std(ddof=0)) > 1e-8
    ]
    dropped = sorted(set(variables) - set(valid_variables))
    if dropped:
        (REPORTS / "study0_dropped_factor_variables.txt").write_text(
            "\n".join(dropped) + "\n",
            encoding="utf-8",
        )
    df = df[["puuid", "region", "tier", "tier_num", *valid_variables]].copy()

    per_cell = max(1, int(EFA_SAMPLE_N / (len(REGIONS) * len(TIERS))))
    efa_parts = []
    cfa_parts = []
    counts = []
    for (region, tier), group in df.groupby(["region", "tier"], observed=True):
        group = group.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
        efa_take = min(per_cell, len(group))
        cfa_take = min(per_cell, max(0, len(group) - efa_take))
        efa_part = group.iloc[:efa_take].copy()
        cfa_part = group.iloc[efa_take : efa_take + cfa_take].copy()
        efa_parts.append(efa_part)
        cfa_parts.append(cfa_part)
        counts.append(
            {
                "region": region,
                "tier": tier,
                "available_players": len(group),
                "efa_players": len(efa_part),
                "cfa_players": len(cfa_part),
            }
        )

    efa = pd.concat(efa_parts, ignore_index=True)
    cfa = pd.concat(cfa_parts, ignore_index=True)
    split_counts = pd.DataFrame(counts)
    split_counts.to_csv(TABLES / "study0_factor_sample_counts.csv", index=False)
    pd.concat(
        [
            efa[["puuid", "region", "tier"]].assign(factor_split="EFA"),
            cfa[["puuid", "region", "tier"]].assign(factor_split="CFA"),
        ],
        ignore_index=True,
    ).to_csv(DATA / "study0_factor_sample_puuids.csv", index=False)

    for split_df in [efa, cfa]:
        scaler = StandardScaler()
        split_df[valid_variables] = scaler.fit_transform(split_df[valid_variables])
    return efa, cfa, valid_variables


def factor_variables(players: pd.DataFrame) -> list[str]:
    variables = [
        "on_my_way_pings_pm",
        "enemy_vision_pings_pm",
        "enemy_missing_pings_pm",
        "push_pings_pm",
        "command_pings_pm",
        "assist_me_pings_pm",
        "all_in_pings_pm",
        "need_vision_pings_pm",
        "vision_cleared_pings_pm",
        "get_back_pings_pm",
        "hold_pings_pm",
        "bait_pings_pm",
        "basic_pings_pm",
    ]
    return [v for v in variables if v in players.columns]


def run_efa(
    efa: pd.DataFrame, variables: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = efa.copy()
    x = df[variables].to_numpy()
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    eigenvalues = np.linalg.eigvalsh(corr)[::-1]
    rng = np.random.default_rng(RNG_SEED)
    random_eigs = []
    for _ in range(100):
        rand = rng.normal(size=x.shape)
        random_eigs.append(np.linalg.eigvalsh(np.corrcoef(rand, rowvar=False))[::-1])
    random_eigs = np.asarray(random_eigs)
    random_p95 = np.percentile(random_eigs, 95, axis=0)
    n_factors = int(np.sum(eigenvalues > random_p95))
    n_factors = max(1, min(n_factors, 5))

    fa = FactorAnalysis(n_components=n_factors, random_state=RNG_SEED)
    fa.fit(x)
    loadings = varimax(fa.components_.T)
    communalities = np.sum(loadings**2, axis=1)
    loadings_df = pd.DataFrame(
        loadings, index=variables, columns=[f"factor_{i + 1}" for i in range(n_factors)]
    )
    loadings_df.insert(0, "variable", loadings_df.index)
    loadings_df["communality"] = communalities
    loadings_df["expected_signal_family"] = loadings_df["variable"].map(
        {f"{v}_pm": k for k, vals in PING_MAP.items() for v in vals}
    )
    loadings_df.reset_index(drop=True).to_csv(
        TABLES / "study0_signal_efa_loadings.csv", index=False
    )

    factor_fit_records = []
    n, p = x.shape
    for k in range(1, min(6, p)):
        candidate = FactorAnalysis(n_components=k, random_state=RNG_SEED)
        candidate.fit(x)
        log_like = float(candidate.score(x) * n)
        n_params = p * k + p
        factor_fit_records.append(
            {
                "n_factors": k,
                "n": n,
                "n_variables": p,
                "log_likelihood": log_like,
                "aic": 2 * n_params - 2 * log_like,
                "bic": np.log(n) * n_params - 2 * log_like,
            }
        )
    factor_fit = pd.DataFrame(factor_fit_records)
    factor_fit.to_csv(TABLES / "study0_signal_efa_factor_fit.csv", index=False)

    eigen_df = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "observed_eigenvalue": eigenvalues,
            "parallel_random_p95": random_p95,
            "retain": eigenvalues > random_p95,
        }
    )
    eigen_df["kmo"] = kmo_stat(corr)
    eigen_df.to_csv(TABLES / "study0_signal_efa_parallel_analysis.csv", index=False)
    return loadings_df.reset_index(drop=True), eigen_df, factor_fit


def semopy_fit(
    name: str, syntax: str, df: pd.DataFrame, variables: list[str]
) -> FitResult:
    if Model is None or calc_stats is None:
        return FitResult(name, "semopy_unavailable", len(df), {})
    try:
        model = Model(syntax)
        model.fit(df[variables])
        fit = calc_stats(model).loc["Value"].to_dict()
        inspect = model.inspect()
        inspect.to_csv(TABLES / f"study0_cfa_{name}_estimates.csv", index=False)
        keep = ["DoF", "chi2", "CFI", "TLI", "RMSEA", "AIC", "BIC"]
        return FitResult(
            name,
            "ok",
            len(df),
            {k: float(fit[k]) if k in fit and pd.notna(fit[k]) else None for k in keep},
        )
    except Exception as exc:
        return FitResult(name, f"failed: {type(exc).__name__}: {exc}", len(df), {})


def run_cfa(cfa: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    df = cfa.copy()
    family_vars = {
        family: [f"{raw}_pm" for raw in raws if f"{raw}_pm" in variables]
        for family, raws in PING_MAP.items()
    }

    one_factor = "DigitalSignal =~ " + " + ".join(variables)
    factor_names = {
        "strategic": "Strategic",
        "collaborative": "Collaborative",
        "reactive": "Reactive",
    }
    usable_factors = {
        factor_names[name]: cols for name, cols in family_vars.items() if len(cols) >= 2
    }
    factor_lines = [
        f"{name} =~ {' + '.join(cols)}" for name, cols in usable_factors.items()
    ]
    cov_lines = []
    names = list(usable_factors)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            cov_lines.append(f"{left} ~~ {right}")
    three_factor = "\n".join([*factor_lines, *cov_lines])
    higher_order = "\n".join([*factor_lines, "DigitalSignal =~ " + " + ".join(names)])

    results = [semopy_fit("signal_one_factor", one_factor, df, variables)]
    if len(usable_factors) >= 2:
        results.append(
            semopy_fit("signal_correlated_factors", three_factor, df, variables)
        )
    if len(usable_factors) >= 3:
        results.append(semopy_fit("signal_higher_order", higher_order, df, variables))
    else:
        pd.DataFrame(
            [
                {
                    "model": "signal_higher_order",
                    "status": "not_estimated",
                    "reason": "Fewer than three signal families had at least two retained indicators after sparse/zero-variance items were excluded.",
                }
            ]
        ).to_csv(TABLES / "study0_cfa_signal_higher_order_estimates.csv", index=False)
    records = []
    for result in results:
        record = {"model": result.model, "status": result.status, "n": result.n}
        record.update(result.metrics)
        records.append(record)

    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_signal_cfa_fit.csv", index=False)
    return out


def run_integrative_models(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = players.copy()
    predictors = {
        "chb_combat_only": ["kills_pm", "deaths_pm"],
        "expanded_combat_controls": [
            "kills_pm",
            "deaths_pm",
            "gold_earned_pm",
            "total_damage_dealt_pm",
            "total_damage_taken_pm",
        ],
        "chb_collaborative": [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
        ],
        "collaborative_plus_signals": [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
            "strategic_signal_pm",
            "collaborative_signal_pm",
            "reactive_signal_pm",
        ],
        "collaborative_signals_objectives": [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
            "strategic_signal_pm",
            "collaborative_signal_pm",
            "reactive_signal_pm",
            "objective_control",
        ],
    }
    records = []
    coefficients = []
    for region, region_df in model_df.groupby("region", observed=True):
        y = region_df["tier_num"].to_numpy(dtype=float)
        yz = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        for step, cols in predictors.items():
            cols = [c for c in cols if c in region_df.columns]
            x = np.log1p(region_df[cols].clip(lower=0).to_numpy(dtype=float))
            xz = StandardScaler().fit_transform(x)
            model = LinearRegression().fit(xz, yz)
            records.append(
                {
                    "region": region,
                    "model": step,
                    "n_players": len(region_df),
                    "n_predictors": len(cols),
                    "r2": model.score(xz, yz),
                }
            )
            for col, coef in zip(cols, model.coef_):
                coefficients.append(
                    {
                        "region": region,
                        "model": step,
                        "predictor": col,
                        "std_beta": coef,
                    }
                )
    model_out = pd.DataFrame(records)
    coef_out = pd.DataFrame(coefficients)
    model_out.to_csv(TABLES / "study0_hierarchical_regression_r2.csv", index=False)
    coef_out.to_csv(
        TABLES / "study0_hierarchical_regression_coefficients.csv", index=False
    )

    corr_vars = [
        "wards_placed_pm",
        "wards_killed_pm",
        "assists_pm",
        "strategic_pings_pm",
        "objective_control",
        "combat_output",
    ]
    corr_frames = []
    for region, region_df in model_df.groupby("region", observed=True):
        corr = region_df[corr_vars].corr()
        corr = corr.reset_index().melt(
            id_vars="index", var_name="variable_2", value_name="r"
        )
        corr = corr.rename(columns={"index": "variable_1"})
        corr.insert(0, "region", region)
        corr_frames.append(corr)
    corr_out = pd.concat(corr_frames, ignore_index=True)
    corr_out.to_csv(TABLES / "study0_integrative_correlations.csv", index=False)
    return model_out, coef_out


def z_log_rate(series: pd.Series) -> np.ndarray:
    y = np.log1p(series.clip(lower=0).astype(float))
    sd = float(y.std(ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        return np.zeros(len(y), dtype=float)
    return ((y - float(y.mean())) / sd).to_numpy(dtype=float)


def participant_model_sample(
    rows: pd.DataFrame, max_n: int = PARTICIPANT_MODEL_SAMPLE_N
) -> pd.DataFrame:
    if len(rows) <= max_n:
        return rows.copy()
    keys = ["region", "tier", "team_position", "win"]
    n_groups = rows.groupby(keys, observed=True).ngroups
    per_group = max(50, int(math.ceil(max_n / max(1, n_groups))))
    parts = []
    for _, group in rows.groupby(keys, observed=True):
        take = min(len(group), per_group)
        parts.append(
            group.sample(n=take, random_state=RNG_SEED) if len(group) > take else group
        )
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_n:
        out = out.sample(n=max_n, random_state=RNG_SEED).reset_index(drop=True)
    return out


def role_stratified_tier_gradients(
    rows: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    records = []
    for metric in metrics:
        if metric not in rows.columns:
            continue
        for (region, role), group in rows.groupby(
            ["region", "team_position"], observed=True
        ):
            work = group[["tier_num", metric]].dropna().copy()
            if len(work) < 200 or work[metric].nunique() < 2:
                continue
            y = z_log_rate(work[metric])
            x = work["tier_num"].to_numpy(dtype=float)
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            lin = stats.linregress(x, y)
            records.append(
                {
                    "region": region,
                    "role": role,
                    "metric": metric,
                    "label": METRIC_LABELS.get(metric, metric),
                    "n_participant_rows": len(work),
                    "std_beta_tier": float(np.corrcoef(x, y)[0, 1]),
                    "r2_tier": float(np.corrcoef(x, y)[0, 1] ** 2),
                    "p_value": float(lin.pvalue),
                }
            )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_role_stratified_tier_gradients.csv", index=False)
    return out


def fit_formula_model(
    work: pd.DataFrame, formula: str, metric: str, model_name: str
) -> dict:
    try:
        model = smf.ols(formula, data=work).fit(
            cov_type="cluster", cov_kwds={"groups": work["puuid"]}
        )
        return {
            "metric": metric,
            "label": METRIC_LABELS.get(metric, metric),
            "model": model_name,
            "n_participant_rows": int(model.nobs),
            "tier_beta": float(model.params.get("tier_z", np.nan)),
            "tier_p_value": float(model.pvalues.get("tier_z", np.nan)),
            "tier_win_interaction_beta": float(
                model.params.get("tier_z:C(win)[T.1]", np.nan)
            ),
            "tier_win_interaction_p": float(
                model.pvalues.get("tier_z:C(win)[T.1]", np.nan)
            ),
            "r2": float(model.rsquared),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "metric": metric,
            "label": METRIC_LABELS.get(metric, metric),
            "model": model_name,
            "n_participant_rows": len(work),
            "tier_beta": np.nan,
            "tier_p_value": np.nan,
            "tier_win_interaction_beta": np.nan,
            "tier_win_interaction_p": np.nan,
            "r2": np.nan,
            "status": f"failed: {type(exc).__name__}: {exc}",
        }


def participant_adjusted_robustness(
    rows: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = participant_model_sample(rows)
    sample = sample.dropna(
        subset=[
            "team_position",
            "patch",
            "champion_id",
            "tier_num",
            "game_duration",
            "win",
        ]
    ).copy()
    sample["tier_z"] = StandardScaler().fit_transform(sample[["tier_num"]]).ravel()
    sample["duration_z"] = (
        StandardScaler().fit_transform(sample[["game_duration"]]).ravel()
    )
    sample["win"] = sample["win"].astype(int)
    sample["champion_id"] = sample["champion_id"].astype(str)
    records = []
    win_records = []
    formulas = {
        "participant_base_cluster_player": "metric_z ~ tier_z",
        "role_patch_duration_adjusted": "metric_z ~ tier_z + C(region) + C(team_position) + C(patch) + duration_z",
        "champion_meta_adjusted": "metric_z ~ tier_z + C(region) + C(team_position) + C(champion_id) + C(patch) + duration_z",
        "win_interaction_champion_adjusted": "metric_z ~ tier_z * C(win) + C(region) + C(team_position) + C(champion_id) + C(patch) + duration_z",
    }
    for metric in metrics:
        if metric not in sample.columns:
            continue
        work = (
            sample[
                [
                    "puuid",
                    "region",
                    "team_position",
                    "patch",
                    "champion_id",
                    "tier_z",
                    "game_duration",
                    "duration_z",
                    "win",
                    metric,
                ]
            ]
            .dropna()
            .copy()
        )
        if len(work) < 1000 or work[metric].nunique() < 2:
            continue
        work["metric_z"] = z_log_rate(work[metric])
        for model_name, formula in formulas.items():
            records.append(fit_formula_model(work, formula, metric, model_name))
        for win_value, win_df in work.groupby("win", observed=True):
            if len(win_df) < 1000 or win_df["metric_z"].nunique() < 2:
                continue
            win_records.append(
                fit_formula_model(
                    win_df,
                    "metric_z ~ tier_z + C(region) + C(team_position) + C(champion_id) + C(patch) + duration_z",
                    metric,
                    f"win_{int(win_value)}_champion_adjusted",
                )
            )
    adjusted = pd.DataFrame(records)
    win_loss = pd.DataFrame(win_records)
    adjusted.to_csv(
        TABLES / "study0_participant_match_adjusted_robustness.csv", index=False
    )
    win_loss.to_csv(TABLES / "study0_win_loss_stratified_robustness.csv", index=False)
    sample_meta = pd.DataFrame(
        [
            {
                "source_rows": len(rows),
                "model_rows": len(sample),
                "sampling": f"stratified by region x tier x team_position x win, capped at {PARTICIPANT_MODEL_SAMPLE_N}",
            }
        ]
    )
    sample_meta.to_csv(
        TABLES / "study0_participant_model_sample_metadata.csv", index=False
    )
    return adjusted, win_loss


def zero_variance_ping_audit(
    rows: pd.DataFrame, variables: list[str], valid_variables: list[str]
) -> pd.DataFrame:
    records = []
    valid = set(valid_variables)
    for var in variables:
        raw = var.removesuffix("_pm")
        if raw not in rows.columns:
            records.append(
                {
                    "ping_variable": var,
                    "raw_nonzero_count": 0,
                    "nonzero_player_percent": 0.0,
                    "patch_coverage": 0,
                    "reason_excluded": "raw field absent from participant rows",
                }
            )
            continue
        raw_series = rows[raw].fillna(0)
        player_nonzero = (
            rows.assign(_nonzero=raw_series > 0)
            .groupby("puuid", observed=True)["_nonzero"]
            .any()
        )
        patch_coverage = rows.loc[raw_series > 0, "patch"].nunique()
        if var in valid:
            reason = "retained for EFA/CFA"
        elif var == "danger_pings_pm":
            reason = "excluded from main EFA/CFA and main reactive composites because non-zero coverage is sparse"
        else:
            reason = "field present but zero or near-zero variance in eligible player aggregates"
        records.append(
            {
                "ping_variable": var,
                "raw_nonzero_count": int((raw_series > 0).sum()),
                "nonzero_player_percent": float(player_nonzero.mean() * 100),
                "patch_coverage": int(patch_coverage),
                "reason_excluded": reason,
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_zero_variance_ping_audit.csv", index=False)
    return out


def cross_validated_integrative_models(
    players: pd.DataFrame, n_splits: int = 5
) -> pd.DataFrame:
    predictors = {
        "chb_combat_only": ["kills_pm", "deaths_pm"],
        "expanded_skill_ecology_controls": [
            "kills_pm",
            "deaths_pm",
            "gold_earned_pm",
            "total_damage_dealt_pm",
            "total_damage_taken_pm",
        ],
        "chb_collaborative": [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
        ],
        "collaborative_plus_general_signalling": [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
            "total_pings_pm",
            "strategic_pings_pm",
        ],
        "collaborative_signals_plus_objective_outcomes": [
            "vision_score_pm",
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
            "total_pings_pm",
            "strategic_pings_pm",
            "objective_control",
        ],
    }
    records = []
    for region, region_df in players.groupby("region", observed=True):
        y = region_df["tier_num"].to_numpy(dtype=float)
        y_scaler = StandardScaler().fit(y.reshape(-1, 1))
        yz = y_scaler.transform(y.reshape(-1, 1)).ravel()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
        for model_name, cols in predictors.items():
            cols = [c for c in cols if c in region_df.columns]
            x = np.log1p(region_df[cols].clip(lower=0).to_numpy(dtype=float))
            for fold, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
                scaler = StandardScaler().fit(x[train_idx])
                x_train = scaler.transform(x[train_idx])
                x_test = scaler.transform(x[test_idx])
                model = LinearRegression().fit(x_train, yz[train_idx])
                records.append(
                    {
                        "region": region,
                        "model": model_name,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "train_r2": float(model.score(x_train, yz[train_idx])),
                        "test_r2": float(model.score(x_test, yz[test_idx])),
                    }
                )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_cross_validated_integrative_r2.csv", index=False)
    return out


def extract_loading_vector(estimates: pd.DataFrame, variables: list[str]) -> pd.Series:
    load = estimates[
        (estimates["op"] == "~") & (estimates["rval"] == "DigitalSignal")
    ].copy()
    load = load.set_index("lval")["Estimate"].reindex(variables)
    return load.astype(float)


def signal_measurement_invariance_audit(
    cfa_sample: pd.DataFrame, variables: list[str]
) -> pd.DataFrame:
    if Model is None or calc_stats is None or not variables:
        out = pd.DataFrame(
            [{"status": "semopy unavailable or no valid signal variables"}]
        )
        out.to_csv(
            TABLES / "study0_signal_measurement_invariance_audit.csv", index=False
        )
        return out

    syntax = "DigitalSignal =~ " + " + ".join(variables)
    records = []
    try:
        pooled_model = Model(syntax)
        pooled_model.fit(cfa_sample[variables])
        pooled_loadings = extract_loading_vector(pooled_model.inspect(), variables)
    except Exception:
        pooled_loadings = pd.Series(index=variables, dtype=float)

    for axis in ["region", "tier"]:
        for group, group_df in cfa_sample.groupby(axis, observed=True):
            if len(group_df) < 500:
                continue
            try:
                model = Model(syntax)
                model.fit(group_df[variables])
                fit = calc_stats(model).loc["Value"].to_dict()
                loadings = extract_loading_vector(model.inspect(), variables)
                joined = pd.concat(
                    [pooled_loadings.rename("pooled"), loadings.rename("group")], axis=1
                ).dropna()
                loading_corr = (
                    float(joined["pooled"].corr(joined["group"]))
                    if len(joined) > 2
                    else np.nan
                )
                max_loading_diff = (
                    float((joined["pooled"] - joined["group"]).abs().max())
                    if len(joined)
                    else np.nan
                )
                mean_abs_item_shift = float(group_df[variables].mean().abs().mean())
                records.append(
                    {
                        "group_axis": axis,
                        "group": group,
                        "n": len(group_df),
                        "status": "ok",
                        "CFI": float(fit.get("CFI", np.nan)),
                        "TLI": float(fit.get("TLI", np.nan)),
                        "RMSEA": float(fit.get("RMSEA", np.nan)),
                        "BIC": float(fit.get("BIC", np.nan)),
                        "loading_correlation_vs_pooled": loading_corr,
                        "max_abs_loading_difference_vs_pooled": max_loading_diff,
                        "mean_abs_item_mean_shift": mean_abs_item_shift,
                        "configural_flag": (
                            "acceptable"
                            if fit.get("CFI", 0) >= 0.90 and fit.get("RMSEA", 1) <= 0.08
                            else "weak"
                        ),
                        "metric_proxy_flag": (
                            "similar"
                            if pd.notna(max_loading_diff) and max_loading_diff < 0.25
                            else "inspect"
                        ),
                        "scalar_proxy_flag": (
                            "mean_shift_small"
                            if mean_abs_item_shift < 0.20
                            else "mean_shift_large"
                        ),
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "group_axis": axis,
                        "group": group,
                        "n": len(group_df),
                        "status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )
    out = pd.DataFrame(records)
    out.to_csv(TABLES / "study0_signal_measurement_invariance_audit.csv", index=False)
    return out


def create_metric_dictionary() -> pd.DataFrame:
    behavioural_inputs = {
        "kills_pm",
        "deaths_pm",
        "assists_pm",
        "wards_placed_pm",
        "wards_killed_pm",
        "control_wards_bought_pm",
        "vision_score_pm",
        *CHB_COMMUNICATION_METRICS,
        "vision_control",
        "strategic_signal",
        "collaborative_signal",
        "reactive_signal",
    }
    skill_ecology_controls = {
        "kills_pm",
        "deaths_pm",
        "gold_earned_pm",
        "total_damage_dealt_pm",
        "total_damage_taken_pm",
        "combat_output",
    }
    rows = []
    for metric, domain in METRIC_DOMAIN.items():
        if metric in OBJECTIVE_METRICS or metric == "objective_control":
            measurement_type = "team_state_outcome"
            analysis_use = "criterion validation and outcome alignment; not a core cultural fingerprint input"
        elif metric in skill_ecology_controls:
            measurement_type = "skill_ecology_control"
            analysis_use = (
                "critical control for tier, champion ecology and win-state confounding"
            )
        elif metric in behavioural_inputs:
            measurement_type = "behavioural_input"
            analysis_use = "individual observable behaviour used for CHB-guided collaboration validation"
        else:
            measurement_type = "extension"
            analysis_use = "supplementary descriptive behaviour"
        rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "study0_role": domain,
                "measurement_type": measurement_type,
                "recommended_later_use": analysis_use,
                "chb_replication_role": (
                    "team_state_objective_outcome"
                    if metric in OBJECTIVE_METRICS
                    else (
                        "collaborative_activity"
                        if metric in ACTIVITY_METRICS
                        else (
                            "communication"
                            if metric in CHB_COMMUNICATION_METRICS
                            else "extension"
                        )
                    )
                ),
                "standardization": "per minute for counts; player-level mean; log1p for slopes/EFA/CFA",
            }
        )
    out = pd.DataFrame(rows).drop_duplicates("metric")
    out.to_csv(TABLES / "study0_metric_dictionary.csv", index=False)
    return out


def write_figure_contract() -> None:
    text = """# Study 0 Figure Contract

Core conclusion: across NA1, EUW1 and KR, League of Legends telemetry can be used for CHB-guided measurement validation of digitally mediated collaboration. Within CHB's narrow combat-control frame, collaborative behaviours show stronger tier gradients than kills/deaths, but expanded skill-ecology controls such as gold and damage are also highly explanatory and must be treated as key controls.

Evidence chain:
1. CHB-style tier gradients establish conceptual replication for behavioural inputs: vision, assists and selected pings should rise with rank relative to kills/deaths.
2. Team-state objective outcomes test whether collaborative telemetry aligns with team results, but they are not treated as pure individual cooperation traits.
3. Ping-family gradients test theory-guided indicator families, while EFA/CFA determines whether a general signalling index is safer than a three-factor latent structure.
4. Robustness checks test role, champion/meta, win-state, participant-match level and cross-validated prediction sensitivity.

Archetype: quantitative grid with objective-outcome validation, behavioural-input validation and signalling panels; prediction and measurement-model diagnostics remain supplementary.

Export contract: Python/matplotlib only; white background; editable SVG/PDF; 600 dpi TIFF; source data written as CSV; statistics and sample definitions reported in accompanying tables.

Review risks:
- AFK cannot be reproduced because the current database has no AFK/timePlayed/disconnect field.
- Event-level match_pings and match_wards are empty, so no timeline or spatial-risk claims are made in Study 0.
- Server region is treated as regional server ecology, not individual culture.
"""
    (REPORTS / "study0_figure_contract.md").write_text(text, encoding="utf-8")


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save_pub(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def grouped_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics: list[str],
    colors: list[str],
    ylabel: str,
    transform: str = "raw",
    legend_loc: str = "best",
    max_players_per_box: int | None = 1600,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    base = np.arange(1, len(TIERS) + 1)
    n_metrics = len(metrics)
    width = min(0.16, 0.72 / max(1, n_metrics))
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * width * 1.2
    handles = []

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        data = []
        positions = []
        for tier_i, tier in enumerate(TIERS, start=1):
            vals = df.loc[df["tier"] == tier, metric].dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if max_players_per_box and len(vals) > max_players_per_box:
                vals = rng.choice(vals, size=max_players_per_box, replace=False)
            if transform == "log1p":
                vals = np.log1p(np.clip(vals, 0, None))
            data.append(vals)
            positions.append(tier_i + offsets[i])
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.6},
            whiskerprops={"linewidth": 0.5, "color": "#555555"},
            capprops={"linewidth": 0.5, "color": "#555555"},
            boxprops={"linewidth": 0.5, "color": "#555555"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.72)
        handles.append(
            mpl.patches.Patch(
                facecolor=colors[i % len(colors)],
                alpha=0.72,
                label=METRIC_LABELS.get(metric, metric),
            )
        )

    ax.set_xlim(0.45, len(TIERS) + 0.55)
    ax.set_xticks(base)
    ax.set_xticklabels(
        ["Iron", "Bronze", "Silver", "Gold", "Plat.", "Emer.", "Diam."], rotation=0
    )
    ax.set_xlabel("Tier")
    ax.set_ylabel(ylabel)
    if handles:
        ax.legend(
            handles=handles, loc=legend_loc, fontsize=5.4, ncol=1, handlelength=1.0
        )


def region_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    transform: str = "log1p",
    max_players_per_box: int | None = 800,
    show_legend: bool = False,
) -> None:
    rng = np.random.default_rng(RNG_SEED)
    colors = {"NA1": "#3B6EA8", "EUW1": "#6B8F71", "KR": "#B05A4A"}
    width = 0.16
    offsets = {"NA1": -width * 1.2, "EUW1": 0.0, "KR": width * 1.2}
    handles = []
    for region in REGIONS:
        data = []
        positions = []
        for tier_i, tier in enumerate(TIERS, start=1):
            vals = (
                df.loc[(df["region"] == region) & (df["tier"] == tier), metric]
                .dropna()
                .to_numpy(dtype=float)
            )
            vals = vals[np.isfinite(vals)]
            if max_players_per_box and len(vals) > max_players_per_box:
                vals = rng.choice(vals, size=max_players_per_box, replace=False)
            if transform == "log1p":
                vals = np.log1p(np.clip(vals, 0, None))
            data.append(vals)
            positions.append(tier_i + offsets[region])
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.55},
            whiskerprops={"linewidth": 0.45, "color": "#555555"},
            capprops={"linewidth": 0.45, "color": "#555555"},
            boxprops={"linewidth": 0.45, "color": "#555555"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[region])
            patch.set_alpha(0.62)
        handles.append(
            mpl.patches.Patch(
                facecolor=colors[region], alpha=0.62, label=display_region(region)
            )
        )
    ax.set_xlim(0.45, len(TIERS) + 0.55)
    ax.set_xticks(range(1, len(TIERS) + 1))
    ax.set_xticklabels(["I", "B", "S", "G", "P", "E", "D"])
    ax.set_xlabel("Tier")
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(
            handles=handles, loc="upper left", fontsize=5.2, ncol=3, handlelength=1.0
        )


def _add_iron_diamond_d(
    ax: plt.Axes,
    players: pd.DataFrame,
    metrics: list[str],
    offsets: np.ndarray,
    transform: str,
    y_frac: float = 0.93,
) -> None:
    """Annotate each metric's Iron-to-Diamond Cohen's d above the tier axis."""
    from scipy import stats as _stats

    for i, metric in enumerate(metrics):
        if metric not in players.columns:
            continue
        iron = (
            players.loc[players["tier"] == "IRON", metric]
            .dropna()
            .to_numpy(dtype=float)
        )
        diam = (
            players.loc[players["tier"] == "DIAMOND", metric]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(iron) < 10 or len(diam) < 10:
            continue
        if transform == "log1p":
            iron = np.log1p(np.clip(iron, 0, None))
            diam = np.log1p(np.clip(diam, 0, None))
        pooled_sd = np.sqrt((np.var(iron, ddof=1) + np.var(diam, ddof=1)) / 2)
        if pooled_sd < 1e-9:
            continue
        d = (np.mean(diam) - np.mean(iron)) / pooled_sd
        x_center = len(TIERS) + offsets[i]  # align with Diamond box
        ax.text(
            x_center,
            ax.get_ylim()[1] * y_frac,
            f"d={d:.2f}",
            ha="center",
            va="top",
            fontsize=4.8,
            color="#333333",
            rotation=90,
        )


def make_main_figure(
    players: pd.DataFrame,
    all_effects: pd.DataFrame,
    cfa_fit: pd.DataFrame,
    model_r2: pd.DataFrame,
) -> None:
    # Figure contract: show that cooperative telemetry recovers monotone tier
    # gradients (criterion validity) and that collaborative inputs rise faster
    # than combat controls — establishing channel separability before cross-
    # regional tests. Hero panel = signalling tier trajectories (panel c).
    set_plot_style()
    # Unified channel palette: info=#7EA6A1, exec=#C7A76C, combat=#9A9A9A
    # Objective outcomes kept green (team-state, not cooperative input)
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.08, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    grouped_boxplot(
        ax_a,
        players,
        ["first_baron", "first_rift_herald", "first_inhibitor"],
        ["#7A9E4B", "#A7B879", "#4F7E3C"],
        "First-objective rate",
        transform="raw",
        legend_loc="upper left",
    )
    add_panel_label(ax_a, "a")

    # Panel b: info-channel (teal) vs combat-control (grey) — channel colours
    grouped_boxplot(
        ax_b,
        players,
        [
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
            "kills_pm",
            "deaths_pm",
        ],
        ["#7EA6A1", "#5B8E8A", "#C7A76C", "#9A9A9A", "#C5C5C5"],
        "Rate per minute (log scale)",
        transform="log1p",
        legend_loc="upper left",
    )
    add_panel_label(ax_b, "b")

    # Panel c: signalling pings — info pings (teal) vs reactive/combat (grey)
    grouped_boxplot(
        ax_c,
        players,
        [
            "on_my_way_pings_pm",
            "enemy_vision_pings_pm",
            "enemy_missing_pings_pm",
            "get_back_pings_pm",
            "need_vision_pings_pm",
        ],
        ["#7EA6A1", "#5B8E8A", "#A8C5C2", "#9A9A9A", "#C5C5C5"],
        "Ping rate per minute (log scale)",
        transform="log1p",
        legend_loc="upper left",
    )
    add_panel_label(ax_c, "c")

    # Annotate Iron-to-Diamond Cohen's d for key metrics in panels b and c
    n_b = 5
    offsets_b = (np.arange(n_b) - (n_b - 1) / 2.0) * min(0.16, 0.72 / n_b) * 1.2
    ax_b.set_ylim(ax_b.get_ylim()[0], ax_b.get_ylim()[1] * 1.08)
    _add_iron_diamond_d(
        ax_b,
        players,
        [
            "wards_killed_pm",
            "control_wards_bought_pm",
            "assists_pm",
            "kills_pm",
            "deaths_pm",
        ],
        offsets_b,
        "log1p",
        y_frac=0.97,
    )

    n_c = 5
    offsets_c = (np.arange(n_c) - (n_c - 1) / 2.0) * min(0.16, 0.72 / n_c) * 1.2
    ax_c.set_ylim(ax_c.get_ylim()[0], ax_c.get_ylim()[1] * 1.08)
    _add_iron_diamond_d(
        ax_c,
        players,
        [
            "on_my_way_pings_pm",
            "enemy_vision_pings_pm",
            "enemy_missing_pings_pm",
            "get_back_pings_pm",
            "need_vision_pings_pm",
        ],
        offsets_c,
        "log1p",
        y_frac=0.97,
    )

    # Caption-level note on error bars for figure legend
    fig.text(
        0.01,
        0.005,
        "Boxes: IQR; whiskers: 1.5×IQR; d = Iron-to-Diamond Cohen's d (log-transformed values).",
        fontsize=4.8,
        color="#666666",
        va="bottom",
    )

    save_pub(fig, FIGURES / "study0_measurement_validation")
    plt.close(fig)


def pooled_standardized_tier(tier_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []
    for metric in metrics:
        tmp = tier_df[tier_df["metric"] == metric].copy()
        if tmp.empty:
            continue
        pooled = (
            tmp.groupby(["tier", "tier_num"], observed=True)["mean"]
            .mean()
            .reset_index()
            .sort_values("tier_num")
        )
        vals = pooled["mean"].to_numpy(dtype=float)
        sd = np.nanstd(vals)
        pooled["mean_z"] = (vals - np.nanmean(vals)) / sd if sd > 0 else 0
        pooled["metric"] = metric
        pooled["label"] = METRIC_LABELS.get(metric, metric)
        records.append(pooled)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def make_supplement_figures(
    players: pd.DataFrame,
    tier_df: pd.DataFrame,
    all_effects: pd.DataFrame,
    consecutive_effects: pd.DataFrame,
    objective_summary: pd.DataFrame,
    efa_loadings: pd.DataFrame,
    efa_eigs: pd.DataFrame,
    efa_factor_fit: pd.DataFrame,
    cfa_fit: pd.DataFrame,
) -> None:
    set_plot_style()

    # Supplementary Fig. S1: participant-match objective control and win leverage.
    if not objective_summary.empty:
        obj = (
            objective_summary.groupby(
                ["objective", "label", "tier", "tier_num"], observed=True
            )
            .agg(
                objective_rate=("objective_rate", "mean"),
                win_rate_delta=("win_rate_delta", "mean"),
            )
            .reset_index()
            .sort_values(["objective", "tier_num"])
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
        colors = ["#7A9E4B", "#A7B879", "#4F7E3C", "#C17C3A", "#6A70A8"]
        handles = []
        labels = []
        for i, (label, group) in enumerate(obj.groupby("label", observed=True)):
            color = colors[i % len(colors)]
            (handle,) = axes[0].plot(
                group["tier_num"],
                group["objective_rate"],
                marker="o",
                lw=1.0,
                ms=2.5,
                label=label,
                color=color,
            )
            axes[1].plot(
                group["tier_num"],
                group["win_rate_delta"],
                marker="o",
                lw=1.0,
                ms=2.5,
                label=label,
                color=color,
            )
            handles.append(handle)
            labels.append(label)
        for ax in axes:
            ax.set_xticks(range(1, 8))
            ax.set_xticklabels(["I", "B", "S", "G", "P", "E", "D"])
            ax.set_xlabel("Tier")
        axes[0].set_ylabel("Probability of first objective")
        axes[1].set_ylabel("Win-rate difference")
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=min(5, len(labels)),
            fontsize=5.5,
            handlelength=1.4,
            columnspacing=0.9,
        )
        add_panel_label(axes[0], "a")
        add_panel_label(axes[1], "b")
        save_pub(fig, FIGURES / "study0_supp_objective_control")
        plt.close(fig)

    # Supplementary Fig. S2: player-level collaborative and individual distributions.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    grouped_boxplot(
        axes[0, 0],
        players,
        [
            "wards_placed_pm",
            "wards_killed_pm",
            "control_wards_bought_pm",
            "vision_score_pm",
        ],
        ["#8BA6A9", "#2F6E73", "#4B7F83", "#6B8F71"],
        "log1p player rate/min",
        transform="log1p",
        legend_loc="upper left",
    )
    grouped_boxplot(
        axes[0, 1],
        players,
        ["assists_pm", "kills_pm", "deaths_pm"],
        ["#6A70A8", "#9A9A9A", "#C5C5C5"],
        "log1p player rate/min",
        transform="log1p",
        legend_loc="upper left",
    )
    grouped_boxplot(
        axes[1, 0],
        players,
        ["gold_earned_pm", "total_damage_dealt_pm", "total_damage_taken_pm"],
        ["#C17C3A", "#B96045", "#D59A58"],
        "log1p player rate/min",
        transform="log1p",
        legend_loc="upper left",
    )
    grouped_boxplot(
        axes[1, 1],
        players,
        ["first_dragon", "first_tower", "first_baron", "first_inhibitor"],
        ["#A7B879", "#7A9E4B", "#4F7E3C", "#6B8F71"],
        "Player mean first-objective rate",
        transform="raw",
        legend_loc="upper left",
    )
    for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
        add_panel_label(ax, label)
    save_pub(fig, FIGURES / "study0_supp_collaborative_vs_individual")
    plt.close(fig)

    # Supplementary Fig. S3: player-level ping distributions by tier and region.
    fig, axes = plt.subplots(
        2, 4, figsize=(7.2, 5.0), constrained_layout=True, sharex=True
    )
    ping_metrics = [
        "on_my_way_pings_pm",
        "enemy_vision_pings_pm",
        "enemy_missing_pings_pm",
        "assist_me_pings_pm",
        "get_back_pings_pm",
        "need_vision_pings_pm",
        "push_pings_pm",
        "all_in_pings_pm",
    ]
    for i, (ax, metric) in enumerate(zip(axes.ravel(), ping_metrics)):
        region_boxplot(
            ax,
            players,
            metric,
            "log1p ping rate/min" if i % 4 == 0 else "",
            transform="log1p",
            max_players_per_box=800,
            show_legend=(i == 0),
        )
        ax.text(
            0.02,
            0.82 if i == 0 else 0.93,
            METRIC_LABELS.get(metric, metric),
            transform=ax.transAxes,
            fontsize=5.8,
            va="top",
            ha="left",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
        add_panel_label(ax, chr(ord("a") + i))
    save_pub(fig, FIGURES / "study0_supp_ping_trajectories")
    plt.close(fig)

    # Supplementary Fig. S4: consecutive tier effect sizes.
    selected = [
        "wards_killed_pm",
        "control_wards_bought_pm",
        "assists_pm",
        "strategic_pings_pm",
        "on_my_way_pings_pm",
        "enemy_vision_pings_pm",
        "get_back_pings_pm",
        "need_vision_pings_pm",
        "all_in_pings_pm",
        "kills_pm",
        "deaths_pm",
        "total_damage_dealt_pm",
        "gold_earned_pm",
    ]

    def fisher_combined_p(values: pd.Series) -> float:
        vals = values.dropna().astype(float)
        vals = vals[(vals > 0) & np.isfinite(vals)]
        if vals.empty:
            return np.nan
        vals = np.clip(vals.to_numpy(), np.finfo(float).tiny, 1.0)
        return float(stats.combine_pvalues(vals, method="fisher").pvalue)

    heat = (
        consecutive_effects[consecutive_effects["metric"].isin(selected)]
        .groupby(["metric", "label", "comparison"], observed=True)
        .agg(
            cohen_d_log=("cohen_d_log", "mean"),
            fisher_p=("p_value", fisher_combined_p),
        )
        .reset_index()
    )
    heat["q_value"] = np.nan
    finite_p = heat["fisher_p"].notna()
    if finite_p.any():
        heat.loc[finite_p, "q_value"] = multipletests(
            heat.loc[finite_p, "fisher_p"], method="fdr_bh"
        )[1]

    def sig_star(q_value: float) -> str:
        if pd.isna(q_value):
            return ""
        if q_value < 0.001:
            return "***"
        if q_value < 0.01:
            return "**"
        if q_value < 0.05:
            return "*"
        return ""

    heat["significance"] = heat["q_value"].map(sig_star)
    heat.to_csv(
        TABLES / "study0_consecutive_tier_effects_display_stats.csv", index=False
    )
    comparisons = [f"{a}_to_{b}" for a, b in zip(TIERS[:-1], TIERS[1:])]
    pivot = heat.pivot(index="label", columns="comparison", values="cohen_d_log")
    pivot = pivot.reindex([METRIC_LABELS.get(m, m) for m in selected])
    pivot = pivot.reindex(columns=comparisons)
    sig_pivot = heat.pivot(index="label", columns="comparison", values="significance")
    sig_pivot = sig_pivot.reindex(pivot.index).reindex(columns=comparisons)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    im = ax.imshow(
        pivot.to_numpy(), cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto"
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(
        [c.replace("_to_", " to\n").title() for c in comparisons], fontsize=5.2
    )
    for row_i, label in enumerate(pivot.index):
        for col_i, comparison in enumerate(comparisons):
            star = (
                sig_pivot.loc[label, comparison]
                if label in sig_pivot.index and comparison in sig_pivot.columns
                else ""
            )
            if not star:
                continue
            val = pivot.loc[label, comparison]
            color = "white" if pd.notna(val) and abs(float(val)) > 0.28 else "black"
            ax.text(
                col_i,
                row_i,
                star,
                ha="center",
                va="center",
                fontsize=5.5,
                color=color,
                fontweight="bold",
            )
    add_panel_label(ax, "a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean log Cohen's d")
    save_pub(fig, FIGURES / "study0_supp_consecutive_tier_effects")
    plt.close(fig)

    # Supplementary Fig. S5: EFA/CFA measurement model details.
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.2, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1.plot(
        efa_eigs["component"],
        efa_eigs["observed_eigenvalue"],
        marker="o",
        lw=1.0,
        label="Observed",
    )
    ax1.plot(
        efa_eigs["component"],
        efa_eigs["parallel_random_p95"],
        marker="o",
        lw=1.0,
        label="Parallel p95",
    )
    # Mark retained factors: where observed > parallel p95
    if (
        not efa_eigs.empty
        and "observed_eigenvalue" in efa_eigs.columns
        and "parallel_random_p95" in efa_eigs.columns
    ):
        retained = efa_eigs[
            efa_eigs["observed_eigenvalue"] > efa_eigs["parallel_random_p95"]
        ]
        n_retained = len(retained)
        if n_retained > 0:
            cutoff = float(retained["component"].max()) + 0.5
            ax1.axvline(cutoff, color="#B05A4A", lw=0.8, ls="--")
            ax1.text(
                cutoff + 0.1,
                ax1.get_ylim()[1] * 0.95,
                f"Retain {n_retained}",
                fontsize=5.2,
                color="#B05A4A",
                va="top",
            )
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue")
    ax1.legend(fontsize=5.5)
    add_panel_label(ax1, "a")

    load = efa_loadings.set_index("variable")[
        [c for c in efa_loadings.columns if c.startswith("factor_")]
    ]
    im2 = ax2.imshow(load.to_numpy(), cmap="RdBu_r", vmin=-0.9, vmax=0.9, aspect="auto")
    ax2.set_yticks(range(len(load.index)))
    ax2.set_yticklabels([METRIC_LABELS.get(v, v) for v in load.index], fontsize=5.4)
    ax2.set_xticks(range(len(load.columns)))
    ax2.set_xticklabels([c.replace("_", " ").title() for c in load.columns])
    add_panel_label(ax2, "b")
    fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02)

    if not cfa_fit.empty and "CFI" in cfa_fit.columns:
        cfa_plot = cfa_fit.copy()
        cfa_plot["short"] = (
            cfa_plot["model"]
            .str.replace("signal_", "", regex=False)
            .str.replace("_", "\n")
        )
        cfi_vals = cfa_plot["CFI"].astype(float)
        ax3.bar(np.arange(len(cfa_plot)), cfi_vals, color="#4B7F83", width=0.55)
        # Reference lines: CFI=0.95 (good fit) and CFI=0.90 (acceptable)
        ax3.axhline(0.95, color="#333333", lw=0.8, ls="-", label="0.95 (good)")
        ax3.axhline(0.90, color="#777777", lw=0.6, ls="--", label="0.90 (acceptable)")
        ax3.legend(fontsize=4.8, loc="lower right")
        lower = max(0.0, float(np.nanmin(cfi_vals)) - 0.04)
        upper = min(1.0, float(np.nanmax(cfi_vals)) + 0.06)
        if lower >= upper:
            lower, upper = 0.0, 1.0
        ax3.set_ylim(lower, upper)
        ax3.set_xticks(np.arange(len(cfa_plot)))
        ax3.set_xticklabels(cfa_plot["short"], fontsize=5.4)
    else:
        ax3.text(
            0.5,
            0.5,
            "CFA unavailable",
            transform=ax3.transAxes,
            ha="center",
            va="center",
        )
        ax3.set_xticks([])
    ax3.set_ylabel("CFI")
    add_panel_label(ax3, "c")
    save_pub(fig, FIGURES / "study0_supp_signal_measurement_model")
    plt.close(fig)


def metric_mean_effect(
    all_effects: pd.DataFrame, metric: str, field: str = "cohen_d_log"
) -> float:
    vals = all_effects.loc[all_effects["metric"] == metric, field].astype(float)
    return float(vals.mean()) if len(vals) else np.nan


def objective_change(
    objective_summary: pd.DataFrame, objective: str, field: str
) -> tuple[float, float]:
    if objective_summary.empty:
        return np.nan, np.nan
    pooled = (
        objective_summary[objective_summary["objective"] == objective]
        .groupby(["tier", "tier_num"], observed=True)[field]
        .mean()
        .reset_index()
        .sort_values("tier_num")
    )
    if pooled.empty:
        return np.nan, np.nan
    iron = pooled.loc[pooled["tier"] == "IRON", field]
    diamond = pooled.loc[pooled["tier"] == "DIAMOND", field]
    return (
        float(iron.iloc[0]) if len(iron) else np.nan,
        float(diamond.iloc[0]) if len(diamond) else np.nan,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    write_figure_contract()
    create_metric_dictionary()

    sample_path = DATA / "study0_sampled_players.csv"
    player_path = DATA / "study0_player_aggregate.csv"
    canonical_path = DATA / "study0_canonical_participant_match_sample.pkl"
    objective_path = TABLES / "study0_objective_win_leverage.csv"
    need_participant_outputs = not objective_path.exists()
    use_cached_player_outputs = False
    if USE_CACHE and sample_path.exists() and player_path.exists():
        cached_sample = pd.read_csv(sample_path)
        cached_players = pd.read_csv(player_path)
        sample_puuids = (
            set(cached_sample["puuid"].astype(str))
            if "puuid" in cached_sample.columns
            else set()
        )
        player_puuids = (
            set(cached_players["puuid"].astype(str))
            if "puuid" in cached_players.columns
            else set()
        )
        use_cached_player_outputs = (
            sample_design_matches(cached_sample)
            and sample_puuids == player_puuids
            and canonical_path.exists()
            and processing_version_matches()
        )

    if use_cached_player_outputs:
        print("Using cached sampled players and player-level aggregate.")
        sample = pd.read_csv(sample_path)
        players = pd.read_csv(player_path)
        if need_participant_outputs:
            rows = canonical_participant_match_sample(
                load_or_build_participant_rows(sample), sample
            )
            sample_row_counts = (
                rows.groupby(["region", "tier"], observed=True)
                .agg(
                    n_rows=("match_id", "size"),
                    n_players=("puuid", "nunique"),
                    n_matches=("match_id", "nunique"),
                )
                .reset_index()
            )
            sample_row_counts.to_csv(
                TABLES / "study0_loaded_row_counts.csv", index=False
            )
            participant_level_summaries(rows)
    else:
        if sample_path.exists():
            cached_sample = pd.read_csv(sample_path)
            if sample_design_matches(cached_sample):
                print(
                    "Using existing sampled-player file to preserve the canonical 126,000-player sample."
                )
                sample = cached_sample
            else:
                with sqlite_connection() as con:
                    print("Loading players and sampling final balanced strata...")
                    sample = load_and_sample_players(con)
        else:
            with sqlite_connection() as con:
                print("Loading players and sampling final balanced strata...")
                sample = load_and_sample_players(con)
        print(f"Sampled {len(sample):,} players.")
        rows = load_or_build_participant_rows(sample)
        print(f"Loaded {len(rows):,} available focal participant rows.")
        rows = canonical_participant_match_sample(rows, sample)
        print(f"Canonical participant-match rows: {len(rows):,}")
        sample_row_counts = (
            rows.groupby(["region", "tier"], observed=True)
            .agg(
                n_rows=("match_id", "size"),
                n_players=("puuid", "nunique"),
                n_matches=("match_id", "nunique"),
            )
            .reset_index()
        )
        sample_row_counts.to_csv(TABLES / "study0_loaded_row_counts.csv", index=False)

        print("Aggregating to player level...")
        players = player_aggregate(rows)
        write_processing_version(rows, players)
        participant_level_summaries(rows)

    print("Computing CHB-style effect tables...")
    objective_effects = effect_table(
        players, OBJECTIVE_METRICS, "study0_objective_effects.csv"
    )
    activity_effects = effect_table(
        players, ACTIVITY_METRICS, "study0_activity_effects.csv"
    )
    communication_effects = effect_table(
        players, CHB_COMMUNICATION_METRICS, "study0_communication_effects.csv"
    )
    composite_effects = effect_table(
        players, COMPOSITE_METRICS, "study0_composite_effects.csv"
    )
    all_effects = pd.concat(
        [objective_effects, activity_effects, communication_effects, composite_effects],
        ignore_index=True,
    )
    all_effects.to_csv(TABLES / "study0_all_effects.csv", index=False)
    all_metrics = (
        OBJECTIVE_METRICS
        + ACTIVITY_METRICS
        + CHB_COMMUNICATION_METRICS
        + COMPOSITE_METRICS
    )
    omnibus = omnibus_tier_tests(players, all_metrics)
    chb_table = chb_replication_table(players, all_metrics, omnibus)
    nonmono = nonmonotonic_signal_tests(players)
    consecutive_effects = consecutive_tier_effect_table(players, all_metrics)
    domain_summary = domain_effect_summary(all_effects)

    tier_df = tier_summary(
        players,
        OBJECTIVE_METRICS
        + ACTIVITY_METRICS
        + CHB_COMMUNICATION_METRICS
        + COMPOSITE_METRICS,
    )

    print("Running EFA/CFA measurement-model checks...")
    all_factor_vars = factor_variables(players)
    efa_sample, cfa_sample, valid_factor_vars = factor_analysis_splits(
        players, all_factor_vars
    )
    efa_loadings, efa_eigs, efa_factor_fit = run_efa(efa_sample, valid_factor_vars)
    cfa_fit = run_cfa(cfa_sample, valid_factor_vars)
    invariance_audit = signal_measurement_invariance_audit(
        cfa_sample, valid_factor_vars
    )

    print("Running integrative regression checks...")
    model_r2, coef_out = run_integrative_models(players)
    cv_r2 = cross_validated_integrative_models(players)

    print("Running Study 0 supplemental robustness checks...")
    participant_rows = canonical_participant_match_sample(
        load_or_build_participant_rows(sample), sample
    )
    robustness_metrics = [
        "wards_killed_pm",
        "control_wards_bought_pm",
        "vision_score_pm",
        "assists_pm",
        "on_my_way_pings_pm",
        "enemy_vision_pings_pm",
        "enemy_missing_pings_pm",
        "get_back_pings_pm",
        "kills_pm",
        "deaths_pm",
        "gold_earned_pm",
        "total_damage_dealt_pm",
    ]
    role_robustness = role_stratified_tier_gradients(
        participant_rows, robustness_metrics
    )
    adjusted_robustness, win_loss_robustness = participant_adjusted_robustness(
        participant_rows, robustness_metrics
    )
    ping_audit = zero_variance_ping_audit(
        participant_rows, all_factor_vars, valid_factor_vars
    )

    print("Creating Study 0 figure...")
    make_main_figure(players, all_effects, cfa_fit, model_r2)
    objective_summary = (
        pd.read_csv(objective_path) if objective_path.exists() else pd.DataFrame()
    )
    make_supplement_figures(
        players,
        tier_df,
        all_effects,
        consecutive_effects,
        objective_summary,
        efa_loadings,
        efa_eigs,
        efa_factor_fit,
        cfa_fit,
    )

    print("Writing report...")
    print("Study 0 complete.")


if __name__ == "__main__":
    main()
