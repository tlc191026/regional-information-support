"""Assemble the focal player-match rows used by the Study 1 re-analysis (all 14 ping fields)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import CANONICAL_LINKS, CANONICAL_SAMPLE, DATA, PING_FIELDS, TABLES, ensure_dirs, log, write_json

CHECK = ["wards_placed", "wards_killed", "control_wards_bought", "push_pings", "get_back_pings", "all_in_pings",
         "on_my_way_pings", "kills", "assists"]


def build_holdout(variant: str) -> None:
    """Study 1 replication rows: the 20 assigned matches of each held-out player."""
    chosen = pd.read_parquet(DATA / "holdout" / variant / "holdout_player_matches.parquet",
                             columns=["puuid", "match_id", "region", "tier", "in_s1"])
    chosen = chosen[chosen["in_s1"]].drop(columns="in_s1")
    parts = pd.read_parquet(DATA / "participants" / "holdout")
    rows = parts.drop(columns=["region"]).merge(chosen, on=["match_id", "puuid"], how="inner", validate="one_to_one")
    assert len(rows) == len(chosen)
    rows["duration_min"] = rows["game_duration"] / 60.0
    rows["total_pings"] = rows[PING_FIELDS].sum(axis=1)
    keep = ["match_id", "puuid", "region", "tier", "team_id", "team_position", "champion_id", "champion_name", "win",
            "game_duration", "duration_min", "patch", "kills", "deaths", "assists", "gold_earned",
            "total_damage_dealt", "vision_score", "wards_placed", "wards_killed", "control_wards_bought",
            *PING_FIELDS, "total_pings"]
    rows[keep].to_parquet(DATA / f"study1_holdout_rows_{variant}.parquet", index=False)
    write_json(TABLES / "qa" / f"study1_holdout_rows_{variant}.json", {
        "rows": int(len(rows)), "players": int(rows["puuid"].nunique()), "matches": int(rows["match_id"].nunique()),
        "players_by_region_tier": rows.groupby(["region", "tier"])["puuid"].nunique().reset_index().to_dict("records"),
    })
    log(f"holdout {variant} study1 rows: {len(rows):,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["discovery", "holdout"], default="discovery")
    ap.add_argument("--variant", choices=["primary", "strict"], default="primary")
    args = ap.parse_args()
    ensure_dirs()
    if args.sample == "holdout":
        build_holdout(args.variant)
        return
    links = pd.read_parquet(CANONICAL_LINKS)
    rows = pd.read_parquet(DATA / "participants" / "fixed")
    rows = rows.merge(links.rename(columns={"region": "list_region", "tier": "list_tier"}),
                      on=["match_id", "puuid"], how="inner", validate="one_to_one")
    assert len(rows) == len(links) == 2520000
    rows["region"] = rows["list_region"]
    rows["tier"] = rows["list_tier"]
    rows["duration_min"] = rows["game_duration"] / 60.0
    rows["total_pings"] = rows[PING_FIELDS].sum(axis=1)
    keep = ["match_id", "puuid", "region", "tier", "team_id", "team_position", "champion_id", "champion_name", "win",
            "game_duration", "duration_min", "patch", "kills", "deaths", "assists", "gold_earned",
            "total_damage_dealt", "vision_score", "wards_placed", "wards_killed", "control_wards_bought",
            *PING_FIELDS, "total_pings"]
    rows = rows[keep]

    old = pd.read_pickle(CANONICAL_SAMPLE)
    old = old[["match_id", "puuid", "team_position", "patch", "game_duration"] +
              [c for c in CHECK if c in old.columns]].drop_duplicates(["match_id", "puuid"])
    both = rows.merge(old, on=["match_id", "puuid"], suffixes=("", "_old"))
    diffs = {c: int((both[c] != both[f"{c}_old"]).sum()) for c in CHECK + ["team_position", "patch", "game_duration"]
             if f"{c}_old" in both.columns}
    rows.to_parquet(DATA / "study1_focal_rows.parquet", index=False)
    write_json(TABLES / "qa" / "study1_focal_rows.json", {
        "rows": int(len(rows)), "players": int(rows["puuid"].nunique()), "matches": int(rows["match_id"].nunique()),
        "compared_with_canonical_sample": int(len(both)), "value_differences": diffs,
        "position_counts": rows["team_position"].value_counts().to_dict(),
    })
    log(f"study1 rows: {len(rows):,}; differences vs canonical sample: {diffs}")


if __name__ == "__main__":
    main()
