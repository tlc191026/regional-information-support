"""Focal-moderator-safe context rerun under the 2026-07-17 taxonomy.

The script never reads participant_process_scores_minimal.parquet or any
metric residual cache. It reconstructs transformed formative scores from the
canonical 2,520,000 participant-match records, fits hierarchy-complete sparse
linear models, and uses player-cluster inference. Bootstrap intervals use an
exact stratified resampling of players applied to cluster score contributions;
all of a sampled player's records and outcome-domain rows remain together.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse, stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exp_20260717"
BASE_EXP = ROOT / "exp_20260717"
CANONICAL = (
    ROOT
    / "outputs"
    / "study0"
    / "data"
    / "study0_canonical_participant_match_sample.pkl"
)
TIMELINE_TEAM = BASE_EXP / "data" / "timeline_match_team_early_infrastructure.parquet"
WHOLE_TEAM = BASE_EXP / "data" / "unique_match_team_whole_match_telemetry.parquet"
ONE_FOCAL_INDICES = BASE_EXP / "data" / "one_focal_per_match_indices.npy"

DIRS = {
    name: OUT / name
    for name in [
        "code",
        "data",
        "tables",
        "figures",
        "source_data",
        "reports",
        "manifest",
    ]
}
for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 191026
BOOTSTRAP_REPS = 1_000
ONE_FOCAL_BOOTSTRAP_REPS = 1_000
RELEASE_ID = "context-rerun-20260717-signal-taxonomy-seed191026"
GIT_COMMIT = "MISSING"
REGION_ORDER = ["NA1", "EUW1", "KR"]
REGION_LABELS = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}
REGION_PAIRS = [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]
TIER_ORDER = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POSITION_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
GROUP_ORDER = ["Lower", "Medium", "High"]
POSITION_GROUP = {
    "TOP": "Lower",
    "MIDDLE": "Medium",
    "BOTTOM": "Medium",
    "JUNGLE": "High",
    "UTILITY": "High",
}

DOMAIN_ORDER = [
    "shared_information_maintenance",
    "team_signalling",
    "joint_action_participation",
    "combat_economy_performance",
]
SIGNAL_SUBTYPE_ORDER = [
    "task_state_attention_signals",
    "coordination_request_signals",
    "action_oriented_signals",
]
# Backward-compatible internal variable name; all exported labels and reports
# identify these outcomes as team-signal subtypes.
INFO_SUBCOMPONENT_ORDER = SIGNAL_SUBTYPE_ORDER
DOMAIN_LABELS = {
    "shared_information_maintenance": "Shared-information maintenance",
    "team_signalling": "Team signalling",
    "joint_action_participation": "Joint-action participation",
    "combat_economy_performance": "Combat-economy performance",
    "task_state_attention_signals": "Task-state and attention signals",
    "coordination_request_signals": "Coordination requests",
    "action_oriented_signals": "Action-oriented signals",
}
REGION_PAIR_COLORS = {"KR-NA": "#B05A4A", "KR-EUW": "#6B8F71", "EUW-NA": "#3B6EA8"}
DOMAIN_COLORS = {
    "shared_information_maintenance": "#236A64",
    "team_signalling": "#5E8FA3",
    "joint_action_participation": "#B88A3B",
    "combat_economy_performance": "#777777",
    "task_state_attention_signals": "#8CB7C7",
    "coordination_request_signals": "#5E8FA3",
    "action_oriented_signals": "#315F73",
}

METRICS = [
    dict(
        metric="vision_score_pm",
        label="Vision score",
        subcomponent="vision_score_diagnostic",
        reverse=False,
        primary=False,
    ),
    dict(
        metric="wards_placed_pm",
        label="Wards placed",
        subcomponent="shared_information_maintenance",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="wards_killed_pm",
        label="Wards cleared",
        subcomponent="shared_information_maintenance",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="control_wards_bought_pm",
        label="Control wards purchased",
        subcomponent="shared_information_maintenance",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="enemy_missing_pings_pm",
        label="Enemy-missing pings",
        subcomponent="task_state_attention_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="enemy_vision_pings_pm",
        label="Enemy-vision pings",
        subcomponent="task_state_attention_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="command_pings_pm",
        label="Command/attention pings",
        subcomponent="task_state_attention_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="need_vision_pings_pm",
        label="Need-vision pings",
        subcomponent="coordination_request_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="assist_me_pings_pm",
        label="Assist-me pings",
        subcomponent="coordination_request_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="on_my_way_pings_pm",
        label="On-my-way pings",
        subcomponent="action_oriented_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="push_pings_pm",
        label="Push pings",
        subcomponent="action_oriented_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="all_in_pings_pm",
        label="All-in pings",
        subcomponent="action_oriented_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="get_back_pings_pm",
        label="Get-back pings",
        subcomponent="action_oriented_signals",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="assists_pm",
        label="Assists",
        subcomponent="joint_action_participation",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="kills_pm",
        label="Kills",
        subcomponent="combat_economy_performance",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="deaths_pm",
        label="Fewer deaths",
        subcomponent="combat_economy_performance",
        reverse=True,
        primary=True,
    ),
    dict(
        metric="gold_earned_pm",
        label="Gold earned",
        subcomponent="combat_economy_performance",
        reverse=False,
        primary=True,
    ),
    dict(
        metric="total_damage_dealt_pm",
        label="Damage dealt",
        subcomponent="combat_economy_performance",
        reverse=False,
        primary=True,
    ),
]
METRIC_NAMES = [item["metric"] for item in METRICS]
PRIMARY_METRICS = [item["metric"] for item in METRICS if item["primary"]]
SUBCOMPONENT_METRICS = {
    subcomponent: [
        item["metric"]
        for item in METRICS
        if item["primary"] and item["subcomponent"] == subcomponent
    ]
    for subcomponent in [
        "shared_information_maintenance",
        *SIGNAL_SUBTYPE_ORDER,
        "joint_action_participation",
        "combat_economy_performance",
    ]
}
DOMAIN_COMPONENTS = {
    "shared_information_maintenance": ["shared_information_maintenance"],
    "team_signalling": SIGNAL_SUBTYPE_ORDER,
    "joint_action_participation": ["joint_action_participation"],
    "combat_economy_performance": ["combat_economy_performance"],
}

TRACE_COLUMNS = [
    "analysis_id",
    "estimate",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
    "n_players",
    "n_records",
    "n_unique_matches",
    "bootstrap_unit",
    "bootstrap_reps",
    "seed",
    "source_script",
    "source_object",
    "git_release_identifier",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def child_seed(namespace: str) -> int:
    digest = hashlib.blake2b(
        f"{MASTER_SEED}:{namespace}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sd = float(np.nanstd(arr, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return np.zeros(len(arr), dtype=np.float32)
    return ((arr - np.nanmean(arr)) / sd).astype(np.float32)


def transform_metric(
    values: pd.Series | np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    raw = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    logged = np.log1p(np.clip(raw, 0, None))
    qlo, qhi = np.nanquantile(logged, [0.01, 0.99])
    clipped = np.clip(logged, qlo, qhi)
    mean, sd = float(clipped.mean()), float(clipped.std(ddof=0))
    return ((clipped - mean) / sd).astype(np.float32), float(qlo), float(qhi), mean, sd


def bh(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    out = np.full(len(arr), np.nan)
    valid = np.isfinite(arr)
    if valid.any():
        out[valid] = multipletests(arr[valid], method="fdr_bh")[1]
    return out


def safe_git_state() -> dict[str, Any]:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if top.returncode:
            return {
                "commit": "MISSING",
                "working_tree": "MISSING",
                "detail": top.stderr.strip(),
            }
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        return {
            "commit": commit,
            "working_tree": "dirty" if status else "clean",
            "detail": status,
        }
    except Exception as exc:
        return {"commit": "MISSING", "working_tree": "MISSING", "detail": str(exc)}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_trace(
    frame: pd.DataFrame, *, bootstrap_reps: int, source_object: str
) -> pd.DataFrame:
    out = frame.copy()
    defaults = {
        "p_value": np.nan,
        "q_value": np.nan,
        "n_players": np.nan,
        "n_records": np.nan,
        "n_unique_matches": np.nan,
        "bootstrap_unit": "region × sampling-tier stratified player cluster",
        "bootstrap_reps": bootstrap_reps,
        "seed": MASTER_SEED,
        "source_script": "exp_20260717/code/run_context_rerun.py",
        "source_object": source_object,
        "git_release_identifier": RELEASE_ID,
    }
    for key, value in defaults.items():
        if key not in out:
            out[key] = value
        else:
            out[key] = (
                out[key].fillna(value)
                if not isinstance(value, str)
                else out[key].fillna(value)
            )
    for column in TRACE_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = math.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "savefig.dpi": 400,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ["png", "pdf", "svg", "tiff"]:
        fig.savefig(
            DIRS["figures"] / f"{stem}.{suffix}",
            bbox_inches="tight",
            dpi=600 if suffix in {"png", "tiff"} else None,
        )


def build_unresidualized_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct all scores without reading any residualized score cache."""
    score_path = DIRS["data"] / "context_unresidualized_participant_scores.parquet"
    audit_path = DIRS["tables"] / "context_score_transformation_audit.csv"
    if score_path.exists() and audit_path.exists():
        return pd.read_parquet(score_path), pd.read_csv(audit_path)

    canonical = pd.read_pickle(CANONICAL)
    required_meta = [
        "match_id",
        "puuid",
        "region",
        "tier",
        "tier_num",
        "team_position",
        "patch",
        "game_duration",
        "duration_min",
        "win",
    ]
    missing = sorted(set(required_meta + METRIC_NAMES) - set(canonical.columns))
    if missing:
        raise KeyError(f"Canonical input is missing columns: {missing}")
    if len(canonical) != 2_520_000 or canonical["puuid"].nunique() != 126_000:
        raise AssertionError("Canonical sample invariants failed")

    transformed = pd.DataFrame(index=canonical.index)
    audits = []
    for item in METRICS:
        values, qlo, qhi, mean, sd = transform_metric(canonical[item["metric"]])
        if item["reverse"]:
            values = -values
        transformed[item["metric"]] = values
        audits.append(
            {
                "metric": item["metric"],
                "primary_metric": item["primary"],
                "reverse_coded": item["reverse"],
                "winsor_log1p_p01": qlo,
                "winsor_log1p_p99": qhi,
                "pre_z_mean": mean,
                "pre_z_sd": sd,
                "post_z_mean": float(values.mean()),
                "post_z_sd": float(values.std(ddof=0)),
                "residualization_controls": "none",
            }
        )

    scores = pd.DataFrame(index=canonical.index)
    for subcomponent, metrics in SUBCOMPONENT_METRICS.items():
        scores[subcomponent] = zscore(transformed[metrics].mean(axis=1).to_numpy())
    for domain, components in DOMAIN_COMPONENTS.items():
        scores[domain] = zscore(scores[components].mean(axis=1).to_numpy())

    keep_raw = ["wards_placed_pm", "wards_killed_pm", "control_wards_bought_pm"]
    compact = pd.concat(
        [
            canonical[required_meta].reset_index(drop=True),
            transformed[["vision_score_pm"]].reset_index(drop=True),
            scores.reset_index(drop=True),
            canonical[keep_raw].reset_index(drop=True),
        ],
        axis=1,
    )
    for column in ["region", "tier", "team_position", "patch"]:
        compact[column] = compact[column].astype("category")
    float_cols = [
        column for column in compact.columns if compact[column].dtype == "float64"
    ]
    compact[float_cols] = compact[float_cols].astype("float32")
    compact.to_parquet(score_path, index=False)
    pd.DataFrame(audits).to_csv(audit_path, index=False)
    del canonical, transformed, scores
    gc.collect()
    return compact, pd.DataFrame(audits)


@dataclass
class DesignSchema:
    name: str
    terms: list[tuple[str, ...]]
    levels: dict[str, list[str]]

    def __post_init__(self) -> None:
        self.offsets: dict[tuple[str, ...], tuple[int, int]] = {}
        self.feature_names: list[str] = []
        offset = 0
        for term in self.terms:
            names = self._term_names(term)
            self.offsets[term] = (offset, offset + len(names))
            self.feature_names.extend(names)
            offset += len(names)
        self.n_features = offset

    def _term_names(self, term: tuple[str, ...]) -> list[str]:
        if not term:
            return ["Intercept"]
        categorical = [factor for factor in term if factor != "duration"]
        level_names = [self.levels[factor][1:] for factor in categorical]
        if not level_names:
            combos = [()]
        else:
            combos = (
                np.array(np.meshgrid(*level_names, indexing="ij"), dtype=object)
                .reshape(len(level_names), -1)
                .T.tolist()
            )
        names = []
        for combo in combos:
            parts = list(combo) if isinstance(combo, list) else list(combo)
            if "duration" in term:
                parts.append("duration_z")
            names.append(":".join(parts))
        return names

    def columns_for_term(self, term: tuple[str, ...]) -> np.ndarray:
        start, stop = self.offsets[term]
        return np.arange(start, stop, dtype=int)

    def build(
        self,
        codes: dict[str, np.ndarray],
        duration: np.ndarray,
        domain_code: int,
        indices: np.ndarray,
    ) -> sparse.csr_matrix:
        n = len(indices)
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        local_codes = {
            factor: values[indices]
            for factor, values in codes.items()
            if factor != "domain"
        }
        for term in self.terms:
            start, _ = self.offsets[term]
            if not term:
                rows_parts.append(np.arange(n, dtype=np.int32))
                cols_parts.append(np.full(n, start, dtype=np.int32))
                data_parts.append(np.ones(n, dtype=np.float32))
                continue
            mask = np.ones(n, dtype=bool)
            active: list[np.ndarray] = []
            sizes: list[int] = []
            numeric = np.ones(n, dtype=np.float32)
            for factor in term:
                if factor == "duration":
                    numeric *= duration[indices].astype(np.float32, copy=False)
                    continue
                values = (
                    np.full(n, domain_code, dtype=np.int16)
                    if factor == "domain"
                    else local_codes[factor]
                )
                current = values.astype(np.int32) - 1
                mask &= current >= 0
                active.append(current)
                sizes.append(len(self.levels[factor]) - 1)
            if not mask.any():
                continue
            interaction_index = np.zeros(mask.sum(), dtype=np.int32)
            for current, size in zip(active, sizes):
                interaction_index = interaction_index * size + current[mask]
            row_index = np.flatnonzero(mask).astype(np.int32)
            rows_parts.append(row_index)
            cols_parts.append((start + interaction_index).astype(np.int32))
            data_parts.append(numeric[mask])
        return sparse.coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(rows_parts), np.concatenate(cols_parts)),
            ),
            shape=(n, self.n_features),
            dtype=np.float32,
        ).tocsr()

    def row_vector(
        self,
        region: str,
        tier: str,
        position: str,
        patch: str,
        domain: str,
        duration_z: float,
    ) -> np.ndarray:
        codes = {
            "region": np.array([self.levels["region"].index(region)], dtype=np.int16),
            "tier": np.array([self.levels["tier"].index(tier)], dtype=np.int16),
            "position": np.array(
                [self.levels["position"].index(position)], dtype=np.int16
            ),
            "patch": np.array([self.levels["patch"].index(patch)], dtype=np.int16),
        }
        domain_code = self.levels["domain"].index(domain)
        return (
            self.build(codes, np.array([duration_z]), domain_code, np.array([0]))
            .toarray()
            .ravel()
        )


def schema_for(
    model: str, outcomes: list[str], patches: list[str]
) -> tuple[DesignSchema, tuple[str, ...]]:
    levels = {
        "region": REGION_ORDER,
        "tier": TIER_ORDER,
        "position": POSITION_ORDER,
        "patch": patches,
        "domain": outcomes,
    }
    if model == "tier":
        terms = [
            (),
            ("region",),
            ("tier",),
            ("domain",),
            ("region", "tier"),
            ("region", "domain"),
            ("tier", "domain"),
            ("region", "tier", "domain"),
            ("position",),
            ("position", "domain"),
            ("patch",),
            ("patch", "domain"),
            ("duration",),
            ("duration", "domain"),
        ]
        tested = ("region", "tier", "domain")
    elif model == "position":
        terms = [
            (),
            ("region",),
            ("position",),
            ("domain",),
            ("region", "position"),
            ("region", "domain"),
            ("position", "domain"),
            ("region", "position", "domain"),
            ("tier",),
            ("tier", "domain"),
            ("patch",),
            ("patch", "domain"),
            ("duration",),
            ("duration", "domain"),
        ]
        tested = ("region", "position", "domain")
    elif model == "joint":
        terms = [
            (),
            ("region",),
            ("tier",),
            ("position",),
            ("domain",),
            ("region", "tier"),
            ("region", "position"),
            ("region", "domain"),
            ("tier", "domain"),
            ("position", "domain"),
            ("region", "tier", "domain"),
            ("region", "position", "domain"),
            ("patch",),
            ("patch", "domain"),
            ("duration",),
            ("duration", "domain"),
        ]
        tested = ("region", "tier", "domain")
    elif model == "region_domain":
        terms = [
            (),
            ("region",),
            ("domain",),
            ("region", "domain"),
            ("tier",),
            ("tier", "domain"),
            ("position",),
            ("position", "domain"),
            ("patch",),
            ("patch", "domain"),
            ("duration",),
            ("duration", "domain"),
        ]
        tested = ("region", "domain")
    else:
        raise ValueError(model)
    return DesignSchema(model, terms, levels), tested


@dataclass
class FitResult:
    schema: DesignSchema
    outcomes: list[str]
    indices: np.ndarray
    beta: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    yty: float
    ysum: float
    n_long: int
    r2: float
    sse: float
    reduced_columns: np.ndarray
    beta_reduced: np.ndarray
    gram_reduced: np.ndarray
    cross_reduced: np.ndarray
    r2_reduced: float
    sse_reduced: float
    tested_term: tuple[str, ...]
    rank: int
    condition_number: float


def fit_model(
    frame: pd.DataFrame,
    outcomes: list[str],
    model: str,
    indices: np.ndarray,
    omit_term: tuple[str, ...] | None = None,
) -> FitResult:
    patches = sorted(
        frame["patch"].astype(str).unique(),
        key=lambda value: tuple(int(x) for x in value.split(".")),
    )
    schema, tested = schema_for(model, outcomes, patches)
    if omit_term is None:
        omit_term = tested
    position_values = frame["team_position"].astype(str)
    position_values = position_values.where(
        position_values.isin(schema.levels["position"]), schema.levels["position"][0]
    )
    codes = {
        "region": pd.Categorical(
            frame["region"].astype(str), categories=REGION_ORDER
        ).codes.astype(np.int16),
        "tier": pd.Categorical(
            frame["tier"].astype(str), categories=TIER_ORDER
        ).codes.astype(np.int16),
        "position": pd.Categorical(
            position_values, categories=schema.levels["position"]
        ).codes.astype(np.int16),
        "patch": pd.Categorical(
            frame["patch"].astype(str), categories=patches
        ).codes.astype(np.int16),
    }
    duration_all = frame["game_duration"].to_numpy(dtype=np.float64)
    duration_mean = float(duration_all[indices].mean())
    duration_sd = float(duration_all[indices].std(ddof=0))
    duration = ((duration_all - duration_mean) / duration_sd).astype(np.float32)
    gram = np.zeros((schema.n_features, schema.n_features), dtype=np.float64)
    cross = np.zeros(schema.n_features, dtype=np.float64)
    yty = 0.0
    ysum = 0.0
    for domain_code, outcome in enumerate(outcomes):
        y = frame[outcome].to_numpy(dtype=np.float64)[indices]
        x = schema.build(codes, duration, domain_code, indices)
        gram += (x.T @ x).toarray()
        cross += np.asarray(x.T @ y).ravel()
        yty += float(y @ y)
        ysum += float(y.sum())
        del x, y
    n_long = len(indices) * len(outcomes)
    gram_inv = np.linalg.pinv(gram, rcond=1e-11, hermitian=True)
    beta = gram_inv @ cross
    sse = float(yty - 2 * beta @ cross + beta @ gram @ beta)
    sst = float(yty - ysum * ysum / n_long)
    tested_cols = schema.columns_for_term(omit_term)
    reduced_columns = np.setdiff1d(np.arange(schema.n_features), tested_cols)
    gram_reduced = gram[np.ix_(reduced_columns, reduced_columns)]
    cross_reduced = cross[reduced_columns]
    beta_reduced = (
        np.linalg.pinv(gram_reduced, rcond=1e-11, hermitian=True) @ cross_reduced
    )
    sse_reduced = float(
        yty
        - 2 * beta_reduced @ cross_reduced
        + beta_reduced @ gram_reduced @ beta_reduced
    )
    rank = int(np.linalg.matrix_rank(gram, tol=np.linalg.eigvalsh(gram).max() * 1e-10))
    return FitResult(
        schema=schema,
        outcomes=outcomes,
        indices=indices,
        beta=beta,
        gram=gram,
        cross=cross,
        yty=yty,
        ysum=ysum,
        n_long=n_long,
        r2=1 - sse / sst,
        sse=sse,
        reduced_columns=reduced_columns,
        beta_reduced=beta_reduced,
        gram_reduced=gram_reduced,
        cross_reduced=cross_reduced,
        r2_reduced=1 - sse_reduced / sst,
        sse_reduced=sse_reduced,
        tested_term=omit_term,
        rank=rank,
        condition_number=float(np.linalg.cond(gram)),
    )


def model_arrays(
    frame: pd.DataFrame, fit: FitResult
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    schema = fit.schema
    position_values = frame["team_position"].astype(str)
    position_values = position_values.where(
        position_values.isin(schema.levels["position"]), schema.levels["position"][0]
    )
    codes = {
        "region": pd.Categorical(
            frame["region"].astype(str), categories=schema.levels["region"]
        ).codes.astype(np.int16),
        "tier": pd.Categorical(
            frame["tier"].astype(str), categories=schema.levels["tier"]
        ).codes.astype(np.int16),
        "position": pd.Categorical(
            position_values, categories=schema.levels["position"]
        ).codes.astype(np.int16),
        "patch": pd.Categorical(
            frame["patch"].astype(str), categories=schema.levels["patch"]
        ).codes.astype(np.int16),
    }
    raw = frame["game_duration"].to_numpy(dtype=np.float64)
    selected = raw[fit.indices]
    duration = ((raw - selected.mean()) / selected.std(ddof=0)).astype(np.float32)
    return codes, duration


@dataclass
class ClusterComponents:
    scores_full: np.ndarray
    scores_reduced: np.ndarray
    sse_full: np.ndarray
    sse_reduced: np.ndarray
    ysum: np.ndarray
    y2sum: np.ndarray
    nobs: np.ndarray
    player_codes: np.ndarray
    player_table: pd.DataFrame


def cluster_components(frame: pd.DataFrame, fit: FitResult) -> ClusterComponents:
    indices = fit.indices
    selected_puuid = frame["puuid"].astype(str).to_numpy()[indices]
    player_labels, selected_codes = np.unique(selected_puuid, return_inverse=True)
    selected_codes = selected_codes.astype(np.int32)
    n_players = len(player_labels)
    player_first = (
        frame.iloc[indices][["puuid", "region", "tier"]]
        .drop_duplicates("puuid")
        .set_index("puuid")
        .loc[player_labels]
        .reset_index()
    )
    player_first["player_code"] = np.arange(n_players)

    scores_full = np.zeros((n_players, fit.schema.n_features), dtype=np.float32)
    scores_reduced = np.zeros((n_players, len(fit.reduced_columns)), dtype=np.float32)
    sse_full = np.zeros(n_players, dtype=np.float64)
    sse_reduced = np.zeros(n_players, dtype=np.float64)
    ysum = np.zeros(n_players, dtype=np.float64)
    y2sum = np.zeros(n_players, dtype=np.float64)
    nobs = np.zeros(n_players, dtype=np.float64)
    grouping = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float32),
            (selected_codes, np.arange(len(indices), dtype=np.int32)),
        ),
        shape=(n_players, len(indices)),
    )
    codes, duration = model_arrays(frame, fit)
    for domain_code, outcome in enumerate(fit.outcomes):
        y = frame[outcome].to_numpy(dtype=np.float64)[indices]
        x_full = fit.schema.build(codes, duration, domain_code, indices)
        x_reduced = x_full[:, fit.reduced_columns]
        residual_full = y - x_full @ fit.beta
        residual_reduced = y - x_reduced @ fit.beta_reduced
        score_full_block = grouping @ x_full.multiply(residual_full[:, None])
        score_reduced_block = grouping @ x_reduced.multiply(residual_reduced[:, None])
        scores_full += score_full_block.toarray().astype(np.float32)
        scores_reduced += score_reduced_block.toarray().astype(np.float32)
        sse_full += np.bincount(
            selected_codes, weights=residual_full**2, minlength=n_players
        )
        sse_reduced += np.bincount(
            selected_codes, weights=residual_reduced**2, minlength=n_players
        )
        ysum += np.bincount(selected_codes, weights=y, minlength=n_players)
        y2sum += np.bincount(selected_codes, weights=y**2, minlength=n_players)
        nobs += np.bincount(selected_codes, minlength=n_players)
        del (
            x_full,
            x_reduced,
            residual_full,
            residual_reduced,
            score_full_block,
            score_reduced_block,
            y,
        )
        gc.collect()
    return ClusterComponents(
        scores_full=scores_full,
        scores_reduced=scores_reduced,
        sse_full=sse_full,
        sse_reduced=sse_reduced,
        ysum=ysum,
        y2sum=y2sum,
        nobs=nobs,
        player_codes=selected_codes,
        player_table=player_first,
    )


def generate_bootstrap_weights(
    player_table: pd.DataFrame, reps: int, namespace: str
) -> np.ndarray:
    path = DIRS["data"] / f"bootstrap_weights_{namespace}_{reps}.npy"
    if path.exists():
        weights = np.load(path, mmap_mode="r")
        if weights.shape == (reps, len(player_table)):
            return weights
    weights = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.int16, shape=(reps, len(player_table))
    )
    weights[:] = 0
    generator = np.random.default_rng(child_seed(namespace))
    strata = player_table.groupby(["region", "tier"], observed=True).indices
    for _, members in strata.items():
        members = np.asarray(members, dtype=np.int32)
        n = len(members)
        for start in range(0, reps, 25):
            stop = min(start + 25, reps)
            for replicate in range(start, stop):
                sampled = generator.integers(0, n, size=n)
                weights[replicate, members] = np.bincount(sampled, minlength=n).astype(
                    np.int16
                )
    weights.flush()
    return np.load(path, mmap_mode="r")


def empirical_weights(
    frame: pd.DataFrame, column: str, levels: list[str], indices: np.ndarray
) -> dict[str, float]:
    counts = (
        frame.iloc[indices][column]
        .astype(str)
        .value_counts()
        .reindex(levels, fill_value=0)
        .astype(float)
    )
    return (counts / counts.sum()).to_dict()


def emm_vectors(
    frame: pd.DataFrame, fit: FitResult, moderator: str
) -> tuple[pd.DataFrame, np.ndarray]:
    schema = fit.schema
    patch_weights = empirical_weights(
        frame, "patch", schema.levels["patch"], fit.indices
    )
    vectors = []
    rows = []
    duration_z = 0.0
    if moderator == "tier":
        moderator_levels = TIER_ORDER
        for region in REGION_ORDER:
            for tier in moderator_levels:
                for outcome in fit.outcomes:
                    vector = np.zeros(schema.n_features)
                    for position in POSITION_ORDER:
                        for patch, patch_weight in patch_weights.items():
                            vector += (
                                (1 / len(POSITION_ORDER))
                                * patch_weight
                                * schema.row_vector(
                                    region, tier, position, patch, outcome, duration_z
                                )
                            )
                    rows.append(
                        {
                            "region": REGION_LABELS[region],
                            "moderator": "tier",
                            "moderator_level": tier,
                            "measure": outcome,
                        }
                    )
                    vectors.append(vector)
    elif moderator == "position":
        for region in REGION_ORDER:
            for position in POSITION_ORDER:
                for outcome in fit.outcomes:
                    vector = np.zeros(schema.n_features)
                    for tier in TIER_ORDER:
                        for patch, patch_weight in patch_weights.items():
                            vector += (
                                (1 / len(TIER_ORDER))
                                * patch_weight
                                * schema.row_vector(
                                    region, tier, position, patch, outcome, duration_z
                                )
                            )
                    rows.append(
                        {
                            "region": REGION_LABELS[region],
                            "moderator": "position",
                            "moderator_level": position,
                            "measure": outcome,
                        }
                    )
                    vectors.append(vector)
    elif moderator == "region_domain":
        for region in REGION_ORDER:
            for outcome in fit.outcomes:
                vector = np.zeros(schema.n_features)
                for tier in TIER_ORDER:
                    for position in POSITION_ORDER:
                        for patch, patch_weight in patch_weights.items():
                            vector += (
                                (1 / (len(TIER_ORDER) * len(POSITION_ORDER)))
                                * patch_weight
                                * schema.row_vector(
                                    region, tier, position, patch, outcome, duration_z
                                )
                            )
                rows.append(
                    {
                        "region": REGION_LABELS[region],
                        "moderator": "none",
                        "moderator_level": "all",
                        "measure": outcome,
                    }
                )
                vectors.append(vector)
    else:
        raise ValueError(moderator)
    return pd.DataFrame(rows), np.vstack(vectors)


def contrast_matrix(
    emm: pd.DataFrame, vectors: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    contrasts = []
    lookup = {
        (row.region, row.moderator_level, row.measure): index
        for index, row in emm.reset_index(drop=True).iterrows()
    }
    levels = list(dict.fromkeys(emm["moderator_level"].tolist()))
    measures = list(dict.fromkeys(emm["measure"].tolist()))
    for region_a, region_b in REGION_PAIRS:
        display_a, display_b = REGION_LABELS[region_a], REGION_LABELS[region_b]
        for level in levels:
            for measure in measures:
                ia = lookup[(display_a, level, measure)]
                ib = lookup[(display_b, level, measure)]
                rows.append(
                    {
                        "contrast": f"{display_a}-{display_b}",
                        "moderator_level": level,
                        "measure": measure,
                    }
                )
                contrasts.append(vectors[ia] - vectors[ib])
    return pd.DataFrame(rows), np.vstack(contrasts)


def bootstrap_inference(
    fit: FitResult,
    components: ClusterComponents,
    weights: np.ndarray,
    vectors: np.ndarray,
    batch_size: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return point estimates, percentile limits and bootstrap draws.

    The coefficient update is a one-step cluster score update around the full
    OLS solution. Player resampling itself is exact within region × tier strata;
    all records and all domains enter each player's cluster score.
    """
    bread = np.linalg.pinv(fit.gram, rcond=1e-11, hermitian=True)
    estimates = vectors @ fit.beta
    influence = (components.scores_full.astype(np.float64) @ bread @ vectors.T).astype(
        np.float32
    )
    draws = np.empty((len(weights), len(vectors)), dtype=np.float32)
    for start in range(0, len(weights), batch_size):
        stop = min(start + batch_size, len(weights))
        centered = np.asarray(weights[start:stop], dtype=np.float32) - 1.0
        draws[start:stop] = estimates[None, :] + centered @ influence
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return estimates, low, high, draws


def bootstrap_delta_r2(
    fit: FitResult,
    components: ClusterComponents,
    weights: np.ndarray,
    batch_size: int = 50,
) -> tuple[float, float, float, np.ndarray]:
    """Cluster-pairs bootstrap of fitted residual improvement.

    Each replicate reweights player-level residual sums from the full and
    reduced hierarchy-complete models. This conditions on the fitted nuisance
    coefficients; it is reported as a large-sample cluster bootstrap of model
    fit, not as an exact 1,000-refit bootstrap.
    """
    point = fit.r2 - fit.r2_reduced
    draws = np.empty(len(weights), dtype=np.float64)
    for start in range(0, len(weights), batch_size):
        stop = min(start + batch_size, len(weights))
        current = np.asarray(weights[start:stop], dtype=np.float64)
        sse_full = current @ components.sse_full
        sse_reduced = current @ components.sse_reduced
        ysum = current @ components.ysum
        y2sum = current @ components.y2sum
        nobs = current @ components.nobs
        sst = y2sum - ysum**2 / nobs
        draws[start:stop] = (sse_reduced - sse_full) / sst
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high), draws


def cluster_wald(
    fit: FitResult, components: ClusterComponents
) -> tuple[float, int, float]:
    focal = fit.schema.columns_for_term(fit.tested_term)
    bread = np.linalg.pinv(fit.gram, rcond=1e-11, hermitian=True)
    influence = components.scores_full.astype(np.float64) @ bread[:, focal]
    correction = (
        len(influence)
        / max(len(influence) - 1, 1)
        * (fit.n_long - 1)
        / max(fit.n_long - fit.schema.n_features, 1)
    )
    covariance = influence.T @ influence * correction
    beta = fit.beta[focal]
    statistic = float(
        beta @ np.linalg.pinv(covariance, rcond=1e-11, hermitian=True) @ beta
    )
    df = int(np.linalg.matrix_rank(covariance))
    return statistic, df, float(stats.chi2.sf(statistic, df))


def inference_tables(
    frame: pd.DataFrame,
    fit: FitResult,
    moderator: str,
    components: ClusterComponents,
    weights: np.ndarray,
    analysis_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    emm_meta, emm_c = emm_vectors(frame, fit, moderator)
    emm_est, emm_lo, emm_hi, emm_draws = bootstrap_inference(
        fit, components, weights, emm_c
    )
    emm = emm_meta.copy()
    emm["estimate"] = emm_est
    emm["ci_low"] = emm_lo
    emm["ci_high"] = emm_hi
    emm["analysis_id"] = [f"{analysis_prefix}_EMM_{i+1:03d}" for i in range(len(emm))]
    emm["p_value"] = np.nan
    emm["q_value"] = np.nan

    contrast_meta, contrast_c = contrast_matrix(emm_meta, emm_c)
    contrast_est = contrast_c @ fit.beta
    # Derive contrast draws from corresponding linear combinations directly.
    contrast_est2, contrast_lo, contrast_hi, contrast_draws = bootstrap_inference(
        fit, components, weights, contrast_c
    )
    if not np.allclose(contrast_est, contrast_est2, atol=1e-8):
        raise AssertionError("Contrast estimate mismatch")
    p_values = np.array(
        [
            min(
                1.0,
                2 * min((draw <= 0).mean(), (draw >= 0).mean()) + 1 / (len(draw) + 1),
            )
            for draw in contrast_draws.T
        ]
    )
    contrasts = contrast_meta.copy()
    contrasts["estimate"] = contrast_est
    contrasts["ci_low"] = contrast_lo
    contrasts["ci_high"] = contrast_hi
    contrasts["p_value"] = p_values
    contrasts["q_value"] = bh(p_values)
    contrasts["analysis_id"] = [
        f"{analysis_prefix}_CONTRAST_{i+1:03d}" for i in range(len(contrasts))
    ]

    delta, delta_lo, delta_hi, delta_draws = bootstrap_delta_r2(
        fit, components, weights
    )
    wald, df, p_value = cluster_wald(fit, components)
    omnibus = pd.DataFrame(
        [
            {
                "analysis_id": f"{analysis_prefix}_OMNIBUS",
                "model": fit.schema.name,
                "outcome_set": "+".join(fit.outcomes),
                "tested_term": ":".join(fit.tested_term),
                "estimate_type": "delta_R2",
                "estimate": delta,
                "ci_low": delta_lo,
                "ci_high": delta_hi,
                "p_value": p_value,
                "q_value": np.nan,
                "wald_chi2": wald,
                "wald_df": df,
                "full_r2": fit.r2,
                "reduced_r2": fit.r2_reduced,
                "rank": fit.rank,
                "n_features": fit.schema.n_features,
                "condition_number": fit.condition_number,
            }
        ]
    )
    diagnostics = {
        "delta_r2_draw_min": float(delta_draws.min()),
        "delta_r2_draw_max": float(delta_draws.max()),
        "emm_draw_shape": list(emm_draws.shape),
        "contrast_draw_shape": list(contrast_draws.shape),
    }
    return omnibus, emm, contrasts, diagnostics


def finalize_table(
    frame: pd.DataFrame,
    source_object: str,
    *,
    n_players: int,
    n_records: int,
    n_unique_matches: int,
    reps: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["n_players"] = n_players
    out["n_records"] = n_records
    out["n_unique_matches"] = n_unique_matches
    return add_trace(out, bootstrap_reps=reps, source_object=source_object)


def grouped_position_results(
    position_emm: pd.DataFrame,
    position_fit: FitResult,
    position_components: ClusterComponents,
    weights: np.ndarray,
    frame: pd.DataFrame,
    outcome_label: str,
) -> pd.DataFrame:
    emm_meta, emm_vectors_matrix = emm_vectors(frame, position_fit, "position")
    lookup = {
        (row.region, row.moderator_level, row.measure): index
        for index, row in emm_meta.reset_index(drop=True).iterrows()
    }
    observed = (
        frame.iloc[position_fit.indices]["team_position"]
        .astype(str)
        .value_counts()
        .reindex(POSITION_ORDER, fill_value=0)
        .astype(float)
    )
    group_members = {
        "Lower": ["TOP"],
        "Medium": ["MIDDLE", "BOTTOM"],
        "High": ["JUNGLE", "UTILITY"],
    }
    rows = []
    vectors = []
    for weighting in ["equal_weight", "empirical_frequency_weighted"]:
        for region_a, region_b in REGION_PAIRS:
            display_a, display_b = REGION_LABELS[region_a], REGION_LABELS[region_b]
            for group in GROUP_ORDER:
                members = group_members[group]
                if weighting == "equal_weight":
                    member_weights = np.repeat(1 / len(members), len(members))
                else:
                    member_weights = observed[members].to_numpy(dtype=float)
                    member_weights = member_weights / member_weights.sum()
                for measure in position_fit.outcomes:
                    vector_a = sum(
                        weight
                        * emm_vectors_matrix[lookup[(display_a, position, measure)]]
                        for weight, position in zip(member_weights, members)
                    )
                    vector_b = sum(
                        weight
                        * emm_vectors_matrix[lookup[(display_b, position, measure)]]
                        for weight, position in zip(member_weights, members)
                    )
                    rows.append(
                        {
                            "weighting": weighting,
                            "contrast": f"{display_a}-{display_b}",
                            "role_group": group,
                            "measure": measure,
                        }
                    )
                    vectors.append(vector_a - vector_b)
    vectors_array = np.vstack(vectors)
    estimate, low, high, draws = bootstrap_inference(
        position_fit, position_components, weights, vectors_array
    )
    p_values = np.array(
        [
            min(
                1.0,
                2 * min((draw <= 0).mean(), (draw >= 0).mean()) + 1 / (len(draw) + 1),
            )
            for draw in draws.T
        ]
    )
    result = pd.DataFrame(rows)
    result["estimate"] = estimate
    result["ci_low"] = low
    result["ci_high"] = high
    result["p_value"] = p_values
    result["q_value"] = bh(p_values)
    result["analysis_id"] = [
        f"POSITION_GROUP_{outcome_label}_{i+1:03d}" for i in range(len(result))
    ]
    return result


def fit_primary_context_models(frame: pd.DataFrame) -> dict[str, Any]:
    standard = frame["team_position"].astype(str).isin(POSITION_ORDER).to_numpy()
    indices = np.flatnonzero(standard)
    if (~standard).sum() != 6:
        raise AssertionError(
            f"Expected six non-standard position rows, found {(~standard).sum()}"
        )
    n_players = frame.iloc[indices]["puuid"].nunique()
    n_matches = frame.iloc[indices]["match_id"].nunique()
    outputs: dict[str, Any] = {"diagnostics": {}}
    all_omnibus = []

    model_specs = [
        ("tier", DOMAIN_ORDER, "TIER_DOMAIN", "tier"),
        ("tier", INFO_SUBCOMPONENT_ORDER, "TIER_SIGNAL", "tier"),
        ("position", DOMAIN_ORDER, "POSITION_DOMAIN", "position"),
        ("position", INFO_SUBCOMPONENT_ORDER, "POSITION_SIGNAL", "position"),
    ]
    for model, outcomes, prefix, moderator in model_specs:
        print(f"[{utc_now()}] fitting {prefix}", flush=True)
        fit = fit_model(frame, outcomes, model, indices)
        components = cluster_components(frame, fit)
        weights = generate_bootstrap_weights(
            components.player_table, BOOTSTRAP_REPS, "primary_region_tier"
        )
        omnibus, emm, contrasts, diagnostics = inference_tables(
            frame,
            fit,
            moderator,
            components,
            weights,
            prefix,
        )
        all_omnibus.append(omnibus)
        suffix = "measure" if outcomes == DOMAIN_ORDER else "signal_subtype"
        if model == "tier":
            outputs[f"tier_emm_{suffix}"] = finalize_table(
                emm,
                f"{prefix}.adjusted_means",
                n_players=n_players,
                n_records=len(indices),
                n_unique_matches=n_matches,
                reps=BOOTSTRAP_REPS,
            )
            outputs[f"tier_contrast_{suffix}"] = finalize_table(
                contrasts,
                f"{prefix}.planned_contrasts",
                n_players=n_players,
                n_records=len(indices),
                n_unique_matches=n_matches,
                reps=BOOTSTRAP_REPS,
            )
        else:
            outputs[f"position_emm_{suffix}"] = finalize_table(
                emm,
                f"{prefix}.adjusted_means",
                n_players=n_players,
                n_records=len(indices),
                n_unique_matches=n_matches,
                reps=BOOTSTRAP_REPS,
            )
            grouped = grouped_position_results(
                emm, fit, components, weights, frame, suffix
            )
            outputs[f"position_grouped_{suffix}"] = finalize_table(
                grouped,
                f"{prefix}.grouped_position_contrasts",
                n_players=n_players,
                n_records=len(indices),
                n_unique_matches=n_matches,
                reps=BOOTSTRAP_REPS,
            )
        outputs["diagnostics"][prefix] = diagnostics | {
            "rank": fit.rank,
            "n_features": fit.schema.n_features,
            "condition_number": fit.condition_number,
        }
        del components
        gc.collect()

    omnibus = pd.concat(all_omnibus, ignore_index=True)
    omnibus["q_value"] = bh(omnibus["p_value"])
    outputs["tier_omnibus"] = finalize_table(
        omnibus[omnibus["model"].eq("tier")].copy(),
        "tier_omnibus",
        n_players=n_players,
        n_records=len(indices),
        n_unique_matches=n_matches,
        reps=BOOTSTRAP_REPS,
    )
    outputs["position_omnibus"] = finalize_table(
        omnibus[omnibus["model"].eq("position")].copy(),
        "position_omnibus",
        n_players=n_players,
        n_records=len(indices),
        n_unique_matches=n_matches,
        reps=BOOTSTRAP_REPS,
    )
    outputs["tier_adjusted_means"] = pd.concat(
        [outputs["tier_emm_measure"], outputs["tier_emm_signal_subtype"]],
        ignore_index=True,
    )
    outputs["tier_planned_contrasts"] = pd.concat(
        [outputs["tier_contrast_measure"], outputs["tier_contrast_signal_subtype"]],
        ignore_index=True,
    )
    outputs["position_adjusted_means"] = pd.concat(
        [outputs["position_emm_measure"], outputs["position_emm_signal_subtype"]],
        ignore_index=True,
    )
    outputs["grouped_position_contrasts"] = pd.concat(
        [
            outputs["position_grouped_measure"],
            outputs["position_grouped_signal_subtype"],
        ],
        ignore_index=True,
    )
    return outputs


def fit_joint_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    standard = frame["team_position"].astype(str).isin(POSITION_ORDER).to_numpy()
    indices = np.flatnonzero(standard)
    rows = []
    for outcomes, label in [
        (DOMAIN_ORDER, "measure"),
        (INFO_SUBCOMPONENT_ORDER, "signal_subtype"),
    ]:
        # The full joint hierarchy contains both three-way blocks. Each row
        # compares it with a model deleting only the named three-way term.
        for omitted, term_label in [
            (("region", "tier", "domain"), "region_by_tier_by_measure"),
            (("region", "position", "domain"), "region_by_position_by_measure"),
        ]:
            fit = fit_model(frame, outcomes, "joint", indices, omit_term=omitted)
            components = cluster_components(frame, fit)
            weights = generate_bootstrap_weights(
                components.player_table, BOOTSTRAP_REPS, "primary_region_tier"
            )
            delta, low, high, draws = bootstrap_delta_r2(fit, components, weights)
            wald, df, p = cluster_wald(fit, components)
            rows.append(
                {
                    "analysis_id": f"JOINT_{label.upper()}_{term_label.upper()}",
                    "outcome_set": label,
                    "deleted_term": term_label,
                    "estimate_type": "delta_R2",
                    "estimate": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p,
                    "q_value": np.nan,
                    "wald_chi2": wald,
                    "wald_df": df,
                    "full_r2": fit.r2,
                    "reduced_r2": fit.r2_reduced,
                    "rank": fit.rank,
                    "n_features": fit.schema.n_features,
                    "bootstrap_draw_min": float(draws.min()),
                    "bootstrap_draw_max": float(draws.max()),
                }
            )
            del components, fit
            gc.collect()
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result["p_value"])
    return finalize_table(
        result,
        "joint_model_sensitivity",
        n_players=126_000,
        n_records=int(standard.sum()),
        n_unique_matches=frame.iloc[np.flatnonzero(standard)]["match_id"].nunique(),
        reps=BOOTSTRAP_REPS,
    )


def fit_region_domain_delta_r2(frame: pd.DataFrame) -> pd.DataFrame:
    standard = frame["team_position"].astype(str).isin(POSITION_ORDER).to_numpy()
    indices = np.flatnonzero(standard)
    fit = fit_model(frame, DOMAIN_ORDER, "region_domain", indices)
    components = cluster_components(frame, fit)
    weights = generate_bootstrap_weights(
        components.player_table, BOOTSTRAP_REPS, "primary_region_tier"
    )
    delta, low, high, draws = bootstrap_delta_r2(fit, components, weights)
    wald, df, p = cluster_wald(fit, components)
    result = pd.DataFrame(
        [
            {
                "analysis_id": "REGION_BY_DOMAIN_DELTA_R2_BOOTSTRAP",
                "estimate_type": "delta_R2",
                "estimate": delta,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
                "q_value": p,
                "wald_chi2": wald,
                "wald_df": df,
                "full_r2": fit.r2,
                "reduced_r2": fit.r2_reduced,
                "rank": fit.rank,
                "n_features": fit.schema.n_features,
                "bootstrap_draw_min": float(draws.min()),
                "bootstrap_draw_max": float(draws.max()),
            }
        ]
    )
    return finalize_table(
        result,
        "region_by_domain_delta_r2",
        n_players=126_000,
        n_records=int(standard.sum()),
        n_unique_matches=frame.iloc[indices]["match_id"].nunique(),
        reps=BOOTSTRAP_REPS,
    )


def contrast_of_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    standard = frame["team_position"].astype(str).isin(POSITION_ORDER).to_numpy()
    indices = np.flatnonzero(standard)
    outcomes = [
        "shared_information_maintenance",
        "team_signalling",
        *SIGNAL_SUBTYPE_ORDER,
    ]
    fit = fit_model(frame, outcomes, "region_domain", indices)
    components = cluster_components(frame, fit)
    weights = generate_bootstrap_weights(
        components.player_table, BOOTSTRAP_REPS, "primary_region_tier"
    )
    emm, vectors = emm_vectors(frame, fit, "region_domain")
    lookup = {(row.region, row.measure): i for i, row in emm.iterrows()}
    rows, contrast_vectors = [], []
    for region_a, region_b in REGION_PAIRS:
        a, b = REGION_LABELS[region_a], REGION_LABELS[region_b]
        maintenance = (
            vectors[lookup[(a, outcomes[0])]] - vectors[lookup[(b, outcomes[0])]]
        )
        for comparator in outcomes[1:]:
            comparison = (
                vectors[lookup[(a, comparator)]] - vectors[lookup[(b, comparator)]]
            )
            vector = maintenance - comparison
            rows.append(
                {
                    "contrast": f"{a}-{b}",
                    "focal_measure": outcomes[0],
                    "comparator": comparator,
                    "estimate_type": "adjusted contrast_of_contrasts",
                }
            )
            contrast_vectors.append(vector)
    matrix = np.vstack(contrast_vectors)
    estimate, low, high, draws = bootstrap_inference(fit, components, weights, matrix)
    p_values = np.array(
        [
            min(
                1.0,
                2 * min((draw <= 0).mean(), (draw >= 0).mean()) + 1 / (len(draw) + 1),
            )
            for draw in draws.T
        ]
    )
    result = pd.DataFrame(rows)
    result["estimate"] = estimate
    result["ci_low"] = low
    result["ci_high"] = high
    result["p_value"] = p_values
    result["q_value"] = bh(p_values)
    result["analysis_id"] = [
        f"CONTRAST_OF_CONTRASTS_{i+1:02d}" for i in range(len(result))
    ]
    return finalize_table(
        result,
        "maintenance_versus_signalling_contrast_of_contrasts",
        n_players=components.player_table.shape[0],
        n_records=len(indices),
        n_unique_matches=frame.iloc[indices]["match_id"].nunique(),
        reps=BOOTSTRAP_REPS,
    )


def metric_level_fdr() -> pd.DataFrame:
    path = BASE_EXP / "data" / "player_metric_and_process_scores_minimal.parquet"
    player = pd.read_parquet(path, columns=["puuid", "region", *METRIC_NAMES])
    source = pd.read_csv(BASE_EXP / "tables" / "regional_metric_effect_spectrum.csv")
    rows = []
    for _, item in source.iterrows():
        metric = str(item.metric)
        region_a, region_b = str(item.region_a), str(item.region_b)
        a = player.loc[player.region.astype(str).eq(region_a), metric].to_numpy(
            dtype=float
        )
        b = player.loc[player.region.astype(str).eq(region_b), metric].to_numpy(
            dtype=float
        )
        test = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "analysis_id": f"METRIC_FDR_{metric.upper()}_{item.contrast.replace('-', '_')}",
                "metric": metric,
                "metric_label": item.label,
                "family": item.family,
                "subtype": item.subtype,
                "contrast": item.contrast,
                "estimate_type": "Cohen_d",
                "estimate": item.cohens_d,
                "ci_low": item.ci_low,
                "ci_high": item.ci_high,
                "p_value": float(test.pvalue),
                "q_value": np.nan,
                "n_a": len(a),
                "n_b": len(b),
                "vision_score_diagnostic": metric == "vision_score_pm",
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result["p_value"])
    result["n_players"] = 84_000
    result["n_records"] = 1_680_000
    result["n_unique_matches"] = np.nan
    return add_trace(
        result,
        bootstrap_reps=0,
        source_object="metric_level_FDR_from_final_adjusted_player_scores",
    )


def one_focal_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    one_focal = np.load(ONE_FOCAL_INDICES)
    standard = (
        frame.iloc[one_focal]["team_position"]
        .astype(str)
        .isin(POSITION_ORDER)
        .to_numpy()
    )
    indices = one_focal[standard]
    rows = []
    for model, outcomes, label in [
        ("tier", DOMAIN_ORDER, "tier_domain"),
        ("tier", INFO_SUBCOMPONENT_ORDER, "tier_signal_subtype"),
        ("position", DOMAIN_ORDER, "position_domain"),
        ("position", INFO_SUBCOMPONENT_ORDER, "position_signal_subtype"),
    ]:
        print(f"[{utc_now()}] one-focal sensitivity {label}", flush=True)
        fit = fit_model(frame, outcomes, model, indices)
        components = cluster_components(frame, fit)
        weights = generate_bootstrap_weights(
            components.player_table, ONE_FOCAL_BOOTSTRAP_REPS, "one_focal_region_tier"
        )
        delta, low, high, _ = bootstrap_delta_r2(fit, components, weights)
        wald, df, p = cluster_wald(fit, components)
        rows.append(
            {
                "analysis_id": f"ONE_FOCAL_{label.upper()}",
                "model": model,
                "outcome_set": label,
                "tested_term": ":".join(fit.tested_term),
                "estimate_type": "delta_R2",
                "estimate": delta,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
                "q_value": np.nan,
                "wald_chi2": wald,
                "wald_df": df,
                "full_r2": fit.r2,
                "reduced_r2": fit.r2_reduced,
                "rank": fit.rank,
                "n_features": fit.schema.n_features,
            }
        )
        del components, fit
        gc.collect()
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result["p_value"])
    return finalize_table(
        result,
        "one_focal_per_match_sensitivity",
        n_players=frame.iloc[indices]["puuid"].nunique(),
        n_records=len(indices),
        n_unique_matches=frame.iloc[indices]["match_id"].nunique(),
        reps=ONE_FOCAL_BOOTSTRAP_REPS,
    )


def key_metric_raw_units(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["wards_placed_pm", "wards_killed_pm", "control_wards_bought_pm"]
    for context, group_columns in [
        ("tier", ["puuid", "region", "tier"]),
        ("position", ["puuid", "region", "team_position"]),
    ]:
        grouped = (
            frame.groupby(group_columns, observed=True)[metrics].mean().reset_index()
        )
        record_counts = (
            frame.groupby(group_columns, observed=True)
            .size()
            .rename("n_focal_records")
            .reset_index()
        )
        grouped = grouped.merge(record_counts, on=group_columns)
        level_column = "tier" if context == "tier" else "team_position"
        valid_levels = TIER_ORDER if context == "tier" else POSITION_ORDER
        grouped = grouped[grouped[level_column].astype(str).isin(valid_levels)]
        for (region, level), part in grouped.groupby(
            ["region", level_column], observed=True
        ):
            for metric in metrics:
                values = part[metric].to_numpy(dtype=float) * 10
                se = values.std(ddof=1) / math.sqrt(len(values))
                rows.append(
                    {
                        "analysis_id": f"RAW_{context.upper()}_{REGION_LABELS[str(region)]}_{str(level)}_{metric.upper()}",
                        "context": context,
                        "region": REGION_LABELS[str(region)],
                        "moderator_level": str(level),
                        "metric": metric,
                        "estimate_type": "mean_per_10_minutes",
                        "estimate": values.mean(),
                        "ci_low": values.mean() - 1.96 * se,
                        "ci_high": values.mean() + 1.96 * se,
                        "p_value": np.nan,
                        "q_value": np.nan,
                        "n_players": len(values),
                        "n_records": int(part.n_focal_records.sum()),
                        "n_unique_matches": np.nan,
                    }
                )
    return add_trace(
        pd.DataFrame(rows),
        bootstrap_reps=0,
        source_object="key_metric_player_clustered_raw_unit_summary",
    )


def timeline_match_cluster_bootstrap() -> pd.DataFrame:
    team = pd.read_parquet(
        TIMELINE_TEAM,
        columns=[
            "match_id",
            "region",
            "tier",
            "patch",
            "game_duration",
            "team_side",
            "primary_three_component",
        ],
    )
    team = team[team["tier"].astype(str).isin(TIER_ORDER[:4])].copy()
    if len(team) != 2 * 1_282_764 or team["match_id"].nunique() != 1_282_764:
        raise AssertionError("Timeline common-support sample mismatch")
    design_frame = team[["tier", "patch", "team_side", "game_duration"]].copy()
    for column in ["tier", "patch", "team_side"]:
        design_frame[column] = design_frame[column].astype(str)
    transformer = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=True
                ),
                ["tier", "patch", "team_side"],
            ),
            ("duration", StandardScaler(), ["game_duration"]),
        ],
        sparse_threshold=1.0,
    )
    x = transformer.fit_transform(design_frame)
    x = sparse.hstack([sparse.csr_matrix(np.ones((len(team), 1))), x], format="csr")
    y = team["primary_three_component"].to_numpy(dtype=float)
    adjustment = Ridge(alpha=1e-8, solver="lsqr", fit_intercept=False).fit(x, y)
    team["adjusted_infrastructure"] = y - adjustment.predict(x)
    del x, design_frame, y, adjustment
    gc.collect()
    match = (
        team.groupby(["match_id", "region", "tier"], observed=True)[
            "adjusted_infrastructure"
        ]
        .agg(["count", "sum", lambda x: float(np.square(x).sum())])
        .reset_index()
    )
    match.columns = ["match_id", "region", "tier", "n", "sum", "sumsq"]
    reps = BOOTSTRAP_REPS
    sum_draw = np.zeros((reps, len(REGION_ORDER)), dtype=np.float64)
    sumsq_draw = np.zeros_like(sum_draw)
    n_draw = np.zeros_like(sum_draw)
    generator = np.random.default_rng(child_seed("timeline_match_cluster_bootstrap"))
    for (region, tier), part in match.groupby(["region", "tier"], observed=True):
        region_index = REGION_ORDER.index(str(region))
        values_sum = part["sum"].to_numpy(dtype=float)
        values_sumsq = part["sumsq"].to_numpy(dtype=float)
        values_n = part["n"].to_numpy(dtype=float)
        n_match = len(part)
        for start in range(0, reps, 10):
            stop = min(start + 10, reps)
            sampled = generator.integers(0, n_match, size=(stop - start, n_match))
            sum_draw[start:stop, region_index] += values_sum[sampled].sum(axis=1)
            sumsq_draw[start:stop, region_index] += values_sumsq[sampled].sum(axis=1)
            n_draw[start:stop, region_index] += values_n[sampled].sum(axis=1)
    means = sum_draw / n_draw
    variances = (sumsq_draw - sum_draw**2 / n_draw) / (n_draw - 1)
    point_stats = team.groupby("region", observed=True)["adjusted_infrastructure"].agg(
        ["count", "mean", "var"]
    )
    rows = []
    for region_a, region_b in REGION_PAIRS:
        ia, ib = REGION_ORDER.index(region_a), REGION_ORDER.index(region_b)
        pooled_draw = np.sqrt(
            (
                (n_draw[:, ia] - 1) * variances[:, ia]
                + (n_draw[:, ib] - 1) * variances[:, ib]
            )
            / (n_draw[:, ia] + n_draw[:, ib] - 2)
        )
        d_draw = (means[:, ia] - means[:, ib]) / pooled_draw
        a, b = point_stats.loc[region_a], point_stats.loc[region_b]
        pooled = math.sqrt(
            ((a["count"] - 1) * a["var"] + (b["count"] - 1) * b["var"])
            / (a["count"] + b["count"] - 2)
        )
        d = (a["mean"] - b["mean"]) / pooled
        p = min(
            1.0, 2 * min((d_draw <= 0).mean(), (d_draw >= 0).mean()) + 1 / (reps + 1)
        )
        rows.append(
            {
                "analysis_id": f"TIMELINE_MATCH_CLUSTER_{REGION_LABELS[region_a]}_{REGION_LABELS[region_b]}",
                "contrast": f"{REGION_LABELS[region_a]}-{REGION_LABELS[region_b]}",
                "estimate_type": "Cohen_d",
                "estimate": d,
                "ci_low": float(np.quantile(d_draw, 0.025)),
                "ci_high": float(np.quantile(d_draw, 0.975)),
                "p_value": p,
                "q_value": np.nan,
                "n_players": np.nan,
                "n_records": int(a["count"] + b["count"]),
                "n_unique_matches": int(
                    match[
                        match.region.astype(str).isin([region_a, region_b])
                    ].match_id.nunique()
                ),
                "bootstrap_unit": "match cluster within region × tier",
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result["p_value"])
    return add_trace(
        result,
        bootstrap_reps=reps,
        source_object="timeline_regional_d_match_cluster_bootstrap",
    )


def wards_cleared_opportunity_adjustment() -> pd.DataFrame:
    team = pd.read_parquet(
        WHOLE_TEAM,
        columns=[
            "match_id",
            "team_id",
            "wards_killed",
            "wards_placed",
            "target_region",
            "target_tier",
            "target_patch",
            "target_duration",
        ],
    )
    opponent = team[["match_id", "team_id", "wards_placed"]].copy()
    opponent["team_id"] = opponent["team_id"].map({100: 200, 200: 100})
    opponent = opponent.rename(columns={"wards_placed": "opponent_wards_placed"})
    team = team.merge(opponent, on=["match_id", "team_id"], validate="one_to_one")
    team = team[team["opponent_wards_placed"].gt(0)].copy()
    team["duration_bin"] = pd.qcut(
        team["target_duration"], q=10, duplicates="drop"
    ).astype(str)
    cells = (
        team.groupby(
            ["target_region", "target_tier", "target_patch", "team_id", "duration_bin"],
            observed=True,
        )
        .agg(
            wards_killed=("wards_killed", "sum"),
            opponent_wards_placed=("opponent_wards_placed", "sum"),
            n_matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    design = cells[
        ["target_region", "target_tier", "target_patch", "team_id", "duration_bin"]
    ].copy()
    design["target_region"] = pd.Categorical(
        design["target_region"].astype(str), categories=REGION_ORDER, ordered=True
    )
    design["target_tier"] = pd.Categorical(
        design["target_tier"].astype(str), categories=TIER_ORDER, ordered=True
    )
    predictors = pd.get_dummies(design, drop_first=True, dtype=float)
    predictors.insert(0, "Intercept", 1.0)
    model = sm.GLM(
        cells["wards_killed"].to_numpy(dtype=float),
        predictors.to_numpy(dtype=float),
        family=sm.families.Poisson(),
        offset=np.log(cells["opponent_wards_placed"].to_numpy(dtype=float)),
    ).fit(cov_type="HC0")
    names = list(predictors.columns)
    beta = pd.Series(model.params, index=names)
    covariance = pd.DataFrame(model.cov_params(), index=names, columns=names)
    euw_name = next(name for name in names if name == "target_region_EUW1")
    kr_name = next(name for name in names if name == "target_region_KR")
    contrasts = [
        ("KR-NA", {kr_name: 1.0}),
        ("KR-EUW", {kr_name: 1.0, euw_name: -1.0}),
        ("EUW-NA", {euw_name: 1.0}),
    ]
    rows = []
    for label, weights in contrasts:
        vector = np.zeros(len(names))
        for name, value in weights.items():
            vector[names.index(name)] = value
        log_irr = float(vector @ beta.to_numpy())
        se = math.sqrt(float(vector @ covariance.to_numpy() @ vector))
        z = log_irr / se
        rows.append(
            {
                "analysis_id": f"WARDS_CLEARED_OPPORTUNITY_{label.replace('-', '_')}",
                "contrast": label,
                "estimate_type": "opponent_deployment_adjusted_IRR",
                "estimate": math.exp(log_irr),
                "ci_low": math.exp(log_irr - 1.96 * se),
                "ci_high": math.exp(log_irr + 1.96 * se),
                "p_value": 2 * stats.norm.sf(abs(z)),
                "q_value": np.nan,
                "log_IRR": log_irr,
                "robust_se_log_IRR": se,
                "n_players": np.nan,
                "n_records": len(team),
                "n_unique_matches": team["match_id"].nunique(),
                "bootstrap_unit": "HC0 cell-aggregated Poisson; opportunity offset=opponent wards placed",
                "bootstrap_reps": 0,
            }
        )
    result = pd.DataFrame(rows)
    result["q_value"] = bh(result["p_value"])
    return add_trace(
        result,
        bootstrap_reps=0,
        source_object="wards_cleared_opponent_deployment_offset_model",
    )


def figure_source_and_plot(
    tier: pd.DataFrame, grouped_position: pd.DataFrame
) -> pd.DataFrame:
    tier_source = tier.copy()
    tier_source["context"] = "Competitive tier"
    tier_source["context_order"] = tier_source["moderator_level"].map(
        {level: i for i, level in enumerate(TIER_ORDER)}
    )
    role_source = grouped_position[
        grouped_position["weighting"].eq("equal_weight")
    ].copy()
    role_source["context"] = "Position group"
    role_source["moderator_level"] = role_source["role_group"]
    role_source["context_order"] = role_source["moderator_level"].map(
        {level: i for i, level in enumerate(GROUP_ORDER)}
    )
    keep = [
        "analysis_id",
        "context",
        "context_order",
        "moderator_level",
        "measure",
        "contrast",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "n_players",
        "n_records",
        "n_unique_matches",
        "bootstrap_unit",
        "bootstrap_reps",
        "seed",
        "source_script",
        "source_object",
        "git_release_identifier",
    ]
    source = pd.concat([tier_source[keep], role_source[keep]], ignore_index=True)

    set_figure_style()
    measures = DOMAIN_ORDER + INFO_SUBCOMPONENT_ORDER
    fig, axes = plt.subplots(2, 5, figsize=(12.2, 5.5), sharey="row")
    for row_index, context in enumerate(["Competitive tier", "Position group"]):
        levels = TIER_ORDER if row_index == 0 else GROUP_ORDER
        labels = (
            ["Iron", "Bronze", "Silver", "Gold", "Platinum", "Emerald", "Diamond"]
            if row_index == 0
            else GROUP_ORDER
        )
        for column_index, measure in enumerate(measures):
            ax = axes[row_index, column_index]
            part = source[source.context.eq(context) & source.measure.eq(measure)]
            for contrast in ["KR-NA", "KR-EUW", "EUW-NA"]:
                line = part[part.contrast.eq(contrast)].sort_values("context_order")
                x = np.arange(len(levels))
                ax.plot(
                    x,
                    line.estimate,
                    marker="o",
                    ms=3.1,
                    color=REGION_PAIR_COLORS[contrast],
                    label=contrast,
                    zorder=3,
                )
                ax.fill_between(
                    x,
                    line.ci_low,
                    line.ci_high,
                    color=REGION_PAIR_COLORS[contrast],
                    alpha=0.12,
                    linewidth=0,
                )
            ax.axhline(0, color="#B8B8B8", lw=0.65, zorder=0)
            ax.set_xticks(
                np.arange(len(levels)),
                labels,
                rotation=38 if row_index == 0 else 0,
                ha="right" if row_index == 0 else "center",
            )
            ax.set_xlabel(
                "Competitive tier" if row_index == 0 else "Pre-specified position group"
            )
            if column_index == 0:
                ax.set_ylabel("Adjusted regional contrast")
            if row_index == 0:
                ax.text(
                    0.5,
                    1.08,
                    DOMAIN_LABELS[measure],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7.1,
                    fontweight="bold",
                )
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E7E7E7", lw=0.45)
            ax.tick_params(length=2.5)
            ax.text(
                -0.16,
                1.04,
                chr(ord("a") + row_index * 5 + column_index),
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=8.5,
            )
            if column_index >= 3:
                ax.set_facecolor("#F2F8F7")
    fig.text(
        0.79,
        0.985,
        "Information-support decomposition",
        ha="center",
        va="top",
        color="#236A64",
        fontsize=7.2,
        fontweight="bold",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.subplots_adjust(
        left=0.065, right=0.995, top=0.89, bottom=0.18, wspace=0.22, hspace=0.45
    )
    save_figure(fig, "context_rerun_figure3")
    plt.close(fig)
    return source


def context_figure_source_only(
    tier: pd.DataFrame, grouped_position: pd.DataFrame
) -> pd.DataFrame:
    """Create final-figure source data; drawing is centralized elsewhere."""
    tier_source = tier.copy()
    tier_source["context"] = "Competitive tier"
    tier_source["context_order"] = tier_source["moderator_level"].map(
        {level: index for index, level in enumerate(TIER_ORDER)}
    )
    position_source = grouped_position[
        grouped_position["weighting"].eq("equal_weight")
    ].copy()
    position_source["context"] = "Position group"
    position_source["moderator_level"] = position_source["role_group"]
    position_source["context_order"] = position_source["moderator_level"].map(
        {level: index for index, level in enumerate(GROUP_ORDER)}
    )
    keep = [
        "analysis_id",
        "context",
        "context_order",
        "moderator_level",
        "measure",
        "contrast",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "q_value",
        "n_players",
        "n_records",
        "n_unique_matches",
        "bootstrap_unit",
        "bootstrap_reps",
        "seed",
        "source_script",
        "source_object",
        "git_release_identifier",
    ]
    return pd.concat([tier_source[keep], position_source[keep]], ignore_index=True)


def write_manifest(
    frame: pd.DataFrame, outputs: list[Path], started: float
) -> dict[str, Any]:
    git = safe_git_state()
    input_paths = [CANONICAL, TIMELINE_TEAM, WHOLE_TEAM, ONE_FOCAL_INDICES]
    manifest = {
        "release_identifier": RELEASE_ID,
        "status": "frozen" if all(path.exists() for path in outputs) else "incomplete",
        "created_utc": utc_now(),
        "elapsed_minutes": (time.time() - started) / 60,
        "master_seed": MASTER_SEED,
        "seed_derivation": "blake2b(f'{MASTER_SEED}:{namespace}', digest_size=8), little-endian modulo 2^32−1",
        "bootstrap": {
            "primary_reps": BOOTSTRAP_REPS,
            "one_focal_reps": ONE_FOCAL_BOOTSTRAP_REPS,
            "unit": "player cluster, stratified by region × sampling-time tier",
            "implementation": "exact player resampling applied to full cluster score contributions; delta-R2 bootstrap reweights fitted player-cluster residual sums",
        },
        "git": git,
        "input_files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in input_paths
        ],
        "prohibited_primary_inputs": [
            "exp_20260717/data/participant_process_scores_minimal.parquet",
            "exp_20260717/data/participant_metric_residuals_minimal.parquet",
        ],
        "sample": {
            "players": int(frame.puuid.nunique()),
            "participant_match_records": len(frame),
            "unique_matches": int(frame.match_id.nunique()),
            "nonstandard_position_rows": int(
                (~frame.team_position.astype(str).isin(POSITION_ORDER)).sum()
            ),
            "tier_definition": "rank at player sampling, not historical match-time rank",
        },
        "construct": {
            "displayed_metrics": 18,
            "primary_metrics": 17,
            "vision_score": "diagnostic only",
            "damage_taken": "excluded",
            "deaths": "reverse-coded after transformation",
            "process_families": ["shared_information_maintenance", "team_signalling"],
            "signal_subtypes": SIGNAL_SUBTYPE_ORDER,
            "task_performance_indicators": [
                "joint_action_participation",
                "combat_economy_performance",
            ],
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "outputs": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
            if path.exists()
        ],
    }
    path = DIRS["manifest"] / "context_rerun_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_qa(
    frame: pd.DataFrame, outputs: dict[str, Path], manifest: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "canonical_players_126000": frame.puuid.nunique() == 126_000,
        "canonical_records_2520000": len(frame) == 2_520_000,
        "canonical_unique_matches_2240354": frame.match_id.nunique() == 2_240_354,
        "six_nonstandard_position_rows": (
            ~frame.team_position.astype(str).isin(POSITION_ORDER)
        ).sum()
        == 6,
        "seventeen_primary_metrics": len(PRIMARY_METRICS) == 17,
        "vision_diagnostic_only": not next(
            item for item in METRICS if item["metric"] == "vision_score_pm"
        )["primary"],
        "damage_taken_absent": all(
            item["metric"] != "total_damage_taken_pm" for item in METRICS
        ),
        "residual_cache_prohibited": all(
            "residual" not in str(path) for path in manifest["input_files"]
        ),
        "bootstrap_reps_1000": manifest["bootstrap"]["primary_reps"] == 1_000,
        "one_focal_bootstrap_reps_1000": manifest["bootstrap"]["one_focal_reps"]
        == 1_000,
        "all_required_outputs_exist": all(
            path.exists() and path.stat().st_size > 0 for path in outputs.values()
        ),
    }
    table_checks = {}
    for name, path in outputs.items():
        if path.suffix.lower() != ".csv":
            continue
        table = pd.read_csv(path)
        required = set(TRACE_COLUMNS)
        table_checks[f"{name}_trace_columns"] = required.issubset(table.columns)
        if {"estimate", "ci_low", "ci_high"}.issubset(table.columns):
            valid = table[["estimate", "ci_low", "ci_high"]].dropna()
            table_checks[f"{name}_ci_order"] = bool(
                (
                    (valid.ci_low <= valid.estimate) & (valid.estimate <= valid.ci_high)
                ).all()
            )
    checks.update(table_checks)
    checks = {name: bool(value) for name, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    qa = {"status": status, "created_utc": utc_now(), "checks": checks}
    (DIRS["reports"] / "context_rerun_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Context rerun QA",
        "",
        f"**{status}** — {sum(checks.values())}/{len(checks)} checks passed.",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Primary context models do not read residualized process-score caches.",
            "- Bootstrap is a stratified player-cluster score bootstrap; it is not an ordinary Ridge standard error.",
            "- Position-group estimates are derived from five-position EMMs and are not estimated while controlling the deterministic source position.",
            "- Git remains MISSING if the workspace has no repository metadata.",
        ]
    )
    (DIRS["reports"] / "context_rerun_QA.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if status != "PASS":
        raise AssertionError("QA failed")
    return qa


def main() -> None:
    started = time.time()
    print(f"[{utc_now()}] reconstructing unresidualized scores", flush=True)
    frame, transformation_audit = build_unresidualized_scores()
    if frame[DOMAIN_ORDER + INFO_SUBCOMPONENT_ORDER].isna().any().any():
        raise AssertionError("Missing context scores")

    primary = fit_primary_context_models(frame)
    table_paths: dict[str, Path] = {
        "tier_omnibus": DIRS["tables"] / "context_tier_omnibus.csv",
        "tier_adjusted_means": DIRS["tables"] / "context_tier_adjusted_means.csv",
        "tier_planned_contrasts": DIRS["tables"] / "context_tier_planned_contrasts.csv",
        "position_omnibus": DIRS["tables"] / "context_position_omnibus.csv",
        "position_adjusted_means": DIRS["tables"]
        / "context_position_adjusted_means.csv",
        "grouped_position_contrasts": DIRS["tables"]
        / "context_grouped_position_contrasts.csv",
    }
    for key, path in table_paths.items():
        primary[key].to_csv(path, index=False)

    print(f"[{utc_now()}] fitting joint sensitivity", flush=True)
    joint = fit_joint_sensitivity(frame)
    joint_path = DIRS["tables"] / "context_joint_model_sensitivity.csv"
    joint.to_csv(joint_path, index=False)
    table_paths["joint_model_sensitivity"] = joint_path

    print(f"[{utc_now()}] fitting one-focal sensitivity", flush=True)
    one_focal = one_focal_sensitivity(frame)
    one_focal_path = DIRS["tables"] / "context_one_focal_sensitivity.csv"
    one_focal.to_csv(one_focal_path, index=False)
    table_paths["one_focal_sensitivity"] = one_focal_path

    raw = key_metric_raw_units(frame)
    raw_path = DIRS["tables"] / "context_key_metrics_raw_units.csv"
    raw.to_csv(raw_path, index=False)
    table_paths["key_metrics_raw_units"] = raw_path

    source = context_figure_source_only(
        primary["tier_planned_contrasts"], primary["grouped_position_contrasts"]
    )
    source_path = DIRS["source_data"] / "context_figure_source_data.csv"
    source.to_csv(source_path, index=False)
    table_paths["context_figure_source_data"] = source_path

    print(f"[{utc_now()}] supplemental inferential gaps", flush=True)
    additions = {
        "contrast_of_contrasts": (
            contrast_of_contrasts(frame),
            DIRS["tables"]
            / "context_maintenance_vs_signalling_contrast_of_contrasts.csv",
        ),
        "metric_fdr": (
            metric_level_fdr(),
            DIRS["tables"] / "context_metric_level_fdr.csv",
        ),
        "region_domain_delta_r2": (
            fit_region_domain_delta_r2(frame),
            DIRS["tables"] / "context_region_domain_delta_r2_bootstrap.csv",
        ),
        "timeline_match_cluster": (
            timeline_match_cluster_bootstrap(),
            DIRS["tables"] / "timeline_regional_d_match_cluster_bootstrap.csv",
        ),
        "wards_opportunity": (
            wards_cleared_opportunity_adjustment(),
            DIRS["tables"] / "wards_cleared_opponent_deployment_adjustment.csv",
        ),
    }
    for key, (table, path) in additions.items():
        table.to_csv(path, index=False)
        table_paths[key] = path

    diagnostics_path = DIRS["reports"] / "context_model_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(primary["diagnostics"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    required_outputs = list(table_paths.values()) + [diagnostics_path]
    manifest = write_manifest(frame, required_outputs, started)
    # Manifest must be included in the required output list after it exists.
    manifest_path = DIRS["manifest"] / "context_rerun_manifest.json"
    table_paths["manifest"] = manifest_path
    report_tables = (
        primary
        | {"joint": joint, "one_focal": one_focal, "raw": raw}
        | {key: value[0] for key, value in additions.items()}
    )
    qa_outputs = table_paths
    qa = run_qa(frame, qa_outputs, manifest)
    summary = {
        "release_identifier": RELEASE_ID,
        "status": qa["status"],
        "elapsed_minutes": (time.time() - started) / 60,
        "tier_omnibus": primary["tier_omnibus"].to_dict("records"),
        "position_omnibus": primary["position_omnibus"].to_dict("records"),
        "joint_sensitivity": joint.to_dict("records"),
        "one_focal_sensitivity": one_focal.to_dict("records"),
        "contrast_of_contrasts": additions["contrast_of_contrasts"][0].to_dict(
            "records"
        ),
        "region_domain_delta_r2": additions["region_domain_delta_r2"][0].to_dict(
            "records"
        ),
        "timeline_match_cluster": additions["timeline_match_cluster"][0].to_dict(
            "records"
        ),
        "wards_opportunity": additions["wards_opportunity"][0].to_dict("records"),
    }
    (DIRS["reports"] / "context_rerun_machine_readable_summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                float(value) if isinstance(value, np.floating) else int(value)
            ),
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": qa["status"],
                "elapsed_minutes": summary["elapsed_minutes"],
                "output": str(OUT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
