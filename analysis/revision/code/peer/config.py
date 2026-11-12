"""Paths, constants and helpers shared by the IJIM supplementary analyses."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "revision"
CODE = WORK / "code"
DATA = ROOT / "baseline" / "data"
TABLES = WORK / "tables" / "peer"
FIGURES = WORK / "figures"
PLAN = WORK / "plan"
LOGS = WORK / "logs"

DB = Path(os.environ.get("RIS_DATABASE", ROOT / "private/lol_data.db"))
TIMELINE_ROOT = Path(os.environ.get("RIS_TIMELINES", ROOT / "private/timelines"))
MATCH_DETAIL_ROOT = Path(os.environ.get("RIS_MATCH_DETAILS", ROOT / "private/match_details"))

REPRO = ROOT / "timeline"
CANONICAL_LINKS = REPRO / "data" / "canonical_focal_links.parquet"
CANONICAL_SAMPLE = ROOT / "inputs/discovery" / "data" / "study0_canonical_participant_match_sample.pkl"
CONTEXT_SCORES = ROOT / "inputs/context" / "data" / "context_unresidualized_participant_scores.parquet"
MATCH_METADATA = ROOT / "inputs/context" / "data" / "canonical_unique_match_metadata.parquet"
BOOTSTRAP_WEIGHTS = ROOT / "inputs/context" / "data" / "bootstrap_weights_primary_region_tier_1000.npy"
TIMELINE_FEATURES_10 = REPRO / "data" / "timeline_features_0_10.parquet"
TIMELINE_TARGETS = REPRO / "data" / "aligned" / "timeline_targets_all.parquet"
TIMELINE_INVENTORY = REPRO / "data" / "aligned" / "timeline_source_file_inventory.json"
TEAM_MINUTE_SHARDS = ROOT / "minute_timeline" / "data" / "minute_shards"
SAMPLING_COUNTS = ROOT / "inputs/discovery" / "tables" / "study0_sampling_counts.csv"

SEED = 191026
REGIONS = ["NA1", "EUW1", "KR"]
REGION_LABELS = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
VALID_WARDS = {"YELLOW_TRINKET", "CONTROL_WARD", "SIGHT_WARD", "BLUE_TRINKET"}
CONTROL_WARD_ITEM = 2055
WINDOW_MS = 600000
MIN_DURATION_S = 900
QUEUE_ID = 420

PING_FIELDS = [
    "all_in_pings", "assist_me_pings", "bait_pings", "basic_pings", "command_pings",
    "danger_pings", "enemy_missing_pings", "enemy_vision_pings", "get_back_pings",
    "hold_pings", "need_vision_pings", "on_my_way_pings", "push_pings", "vision_cleared_pings",
]
# Fixed on 2026-10-22, before any statistic on the five newly added ping fields was computed.
PING_GROUPS = {
    "caution": ["get_back_pings", "danger_pings"],
    "initiation": ["push_pings", "all_in_pings"],
    "intent": ["on_my_way_pings"],
    "request": ["assist_me_pings", "need_vision_pings"],
    "enemy_information": ["enemy_missing_pings", "enemy_vision_pings", "vision_cleared_pings"],
    "tactical": ["command_pings", "hold_pings", "bait_pings"],
    "generic": ["basic_pings"],
}
VISION_FIELDS = ["wards_placed", "wards_killed", "control_wards_bought", "vision_score"]
PERFORMANCE_FIELDS = ["kills", "deaths", "assists", "gold_earned", "total_damage_dealt"]

TYPE_MIN_MATCHES = 10
TYPE_MIN_MATCHES_ROBUST = 5
DUO_MIN_COTEAM = 2
SESOI = {"beta_sd": 0.02, "relative_difference": 0.10, "occurrence_pp": 0.03}

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_name, "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ensure_dirs() -> None:
    for path in [TABLES, TABLES / "qa"]:
        path.mkdir(parents=True, exist_ok=True)


def child_seed(name: str) -> int:
    digest = hashlib.blake2b(f"{SEED}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def connect_db() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=60)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def log(message: str) -> None:
    print(datetime.datetime.now().isoformat(timespec="seconds"), message, flush=True)
