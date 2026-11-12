"""Read-only bulk extraction helpers for the source SQLite database."""
from __future__ import annotations

import concurrent.futures as cf
import sqlite3

import pandas as pd

from config import DB, MIN_DURATION_S, QUEUE_ID, REGIONS

PARTICIPANT_COLUMNS = [
    "match_id", "puuid", "team_id", "team_position", "champion_id", "champion_name",
    "win", "kills", "deaths", "assists", "gold_earned", "total_damage_dealt", "total_damage_taken",
    "vision_score", "wards_placed", "wards_killed", "control_wards_bought",
    "all_in_pings", "assist_me_pings", "bait_pings", "basic_pings", "command_pings", "danger_pings",
    "enemy_missing_pings", "enemy_vision_pings", "get_back_pings", "hold_pings", "need_vision_pings",
    "on_my_way_pings", "push_pings", "vision_cleared_pings",
]
MATCH_COLUMNS = ["match_id", "region", "game_duration", "game_version", "queue_id", "blue_team_win"]


def connect_scratch() -> sqlite3.Connection:
    """Open the source read-only and keep ID lists in an attached in-memory scratch schema."""
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=60)
    con.execute("ATTACH DATABASE ':memory:' AS scratch")
    return con


def load_ids(con: sqlite3.Connection, name: str, values, column: str = "id") -> None:
    con.execute(f"DROP TABLE IF EXISTS scratch.{name}")
    con.execute(f"CREATE TABLE scratch.{name} ({column} TEXT PRIMARY KEY)")
    con.executemany(f"INSERT OR IGNORE INTO scratch.{name} VALUES (?)", ((str(v),) for v in values))


def participants_for_matches(con: sqlite3.Connection, match_ids, columns=PARTICIPANT_COLUMNS) -> pd.DataFrame:
    load_ids(con, "ids", match_ids, "match_id")
    cols = ", ".join(f"mp.{c}" for c in columns)
    # CROSS JOIN fixes the loop order so the small ID list drives indexed lookups.
    sql = (
        f"SELECT {cols} FROM scratch.ids AS i "
        "CROSS JOIN match_participants AS mp INDEXED BY idx_participants_match ON mp.match_id = i.match_id"
    )
    return pd.read_sql_query(sql, con)


def matches_for_ids(con: sqlite3.Connection, match_ids) -> pd.DataFrame:
    load_ids(con, "mids", match_ids, "match_id")
    cols = ", ".join(f"m.{c}" for c in MATCH_COLUMNS)
    sql = f"SELECT {cols} FROM scratch.mids AS i CROSS JOIN matches AS m ON m.match_id = i.match_id"
    return pd.read_sql_query(sql, con)


def valid_matches_for_players(con: sqlite3.Connection, players: pd.DataFrame, extra_columns=()) -> pd.DataFrame:
    """Eligible ranked solo/duo rows of the given players on their own listing platform."""
    load_ids(con, "pids", players["puuid"].unique(), "puuid")
    extra = "".join(f", mp.{c}" for c in extra_columns)
    sql = (
        "SELECT mp.puuid, mp.match_id, m.region AS match_region" + extra + " FROM scratch.pids AS p "
        "CROSS JOIN match_participants AS mp INDEXED BY idx_participants_puuid ON mp.puuid = p.puuid "
        "CROSS JOIN matches AS m ON m.match_id = mp.match_id "
        f"WHERE m.queue_id = {QUEUE_ID} AND m.game_duration >= {MIN_DURATION_S} "
        f"AND m.region IN ({', '.join(repr(r) for r in REGIONS)})"
    )
    rows = pd.read_sql_query(sql, con)
    rows = rows.merge(players[["puuid", "region"]], on="puuid", how="inner")
    rows = rows[rows["match_region"].eq(rows["region"])].drop(columns="match_region")
    return rows.drop_duplicates(["puuid", "match_id"]).reset_index(drop=True)


def add_patch(frame: pd.DataFrame) -> pd.DataFrame:
    frame["patch"] = frame["game_version"].astype(str).str.extract(r"^(\d+\.\d+)", expand=False)
    return frame


def _participant_chunk(args) -> pd.DataFrame:
    match_ids, columns = args
    con = connect_scratch()
    try:
        return participants_for_matches(con, match_ids, columns)
    finally:
        con.close()


def parallel_participants(match_ids, columns=PARTICIPANT_COLUMNS, workers: int = 8, chunk: int = 40000) -> pd.DataFrame:
    """Fetch all participant rows for many matches with independent read-only connections."""
    ids = sorted(set(map(str, match_ids)))
    jobs = [(ids[i:i + chunk], list(columns)) for i in range(0, len(ids), chunk)]
    parts = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for frame in pool.map(_participant_chunk, jobs):
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(columns))
