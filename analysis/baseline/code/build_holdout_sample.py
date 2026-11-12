"""Build held-out player samples from listed players never drawn into the fixed sample.

Only identifiers, listing strata and position labels are read; no behavioural fields.

Primary variant: excludes every fixed-sample match. Strict variant (sensitivity): additionally
excludes matches containing any discovery focal player. The strict rule removes many dense
high-tier NA matches, so it is retained as a sensitivity check rather than the primary rule.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from config import (CANONICAL_LINKS, DATA, POSITIONS, REGIONS, TABLES, TIERS, child_seed, ensure_dirs,
                    log, write_json)
from db_extract import connect_scratch, parallel_participants, valid_matches_for_players

OUT = DATA / "holdout"
MIN_MATCHES = 20
S1_MATCHES = 20
S2_OUTCOME_CAP = 20


def order_key(salt: int, puuid: str, match_id: str) -> int:
    digest = hashlib.blake2b(f"{salt}:{puuid}:{match_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def select(rows: pd.DataFrame, players: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = rows.groupby("puuid").size().rename("n_remaining")
    eligible = players.merge(counts, left_on="puuid", right_index=True, how="inner")
    eligible = eligible[eligible["n_remaining"].ge(MIN_MATCHES)].sort_values(["region", "tier", "puuid"])
    rows = rows[rows["puuid"].isin(eligible["puuid"])].copy()

    rows = rows.sort_values(["puuid", "standard_position", "order_key"], ascending=[True, False, True])
    rows["in_s1"] = rows.groupby("puuid").cumcount().lt(S1_MATCHES)
    rows = rows.sort_values(["puuid", "order_key"])
    rank = rows.groupby("puuid").cumcount()
    n = rows["puuid"].map(eligible.set_index("puuid")["n_remaining"])
    rows["s2_set"] = np.where(rank < np.minimum(S2_OUTCOME_CAP, n // 2), "outcome", "type")

    folder = OUT / variant
    folder.mkdir(parents=True, exist_ok=True)
    keep = ["puuid", "match_id", "region", "tier", "team_position", "contains_focal", "in_s1", "s2_set"]
    rows[keep].to_parquet(folder / "holdout_player_matches.parquet", index=False)
    eligible.to_parquet(folder / "holdout_players.parquet", index=False)
    by_stratum = rows.groupby(["region", "tier"]).agg(
        players=("puuid", "nunique"),
        s1_rows=("in_s1", "sum"),
        s2_outcome_rows=("s2_set", lambda s: int((s == "outcome").sum())),
        s2_type_rows=("s2_set", lambda s: int((s == "type").sum())),
    ).reset_index()
    by_stratum.to_csv(TABLES / f"holdout_counts_by_stratum_{variant}.csv", index=False)
    return eligible, rows


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    links = pd.read_parquet(CANONICAL_LINKS, columns=["match_id", "puuid"])
    focal = set(links["puuid"])
    fixed_matches = set(links["match_id"])

    con = connect_scratch()
    players = pd.read_sql_query(
        "SELECT puuid, region, tier, division, league_points, wins, losses FROM players "
        f"WHERE region IN ({', '.join(repr(r) for r in REGIONS)})",
        con,
    )
    players = players[players["tier"].isin(TIERS) & ~players["puuid"].isin(focal)].drop_duplicates("puuid")
    log(f"listed non-focal players: {len(players):,}")
    rows = valid_matches_for_players(con, players, extra_columns=["team_position"])
    con.close()
    rows = rows.merge(players[["puuid", "tier"]], on="puuid", how="left")
    stage = {"valid_rows": len(rows)}
    rows = rows[~rows["match_id"].isin(fixed_matches)].copy()
    stage["rows_outside_fixed_sample"] = len(rows)

    candidates = rows["match_id"].unique()
    log(f"checking {len(candidates):,} candidate matches")
    parts = parallel_participants(candidates, columns=["match_id", "puuid"], workers=8)
    complete = parts.groupby("match_id").size()
    incomplete = set(complete.index[complete.ne(10)])
    focal_matches = set(parts.loc[parts["puuid"].isin(focal), "match_id"])
    rows = rows[~rows["match_id"].isin(incomplete)].copy()
    rows["contains_focal"] = rows["match_id"].isin(focal_matches)
    stage.update(candidate_matches=len(candidates), matches_without_ten_participants=len(incomplete),
                 matches_with_focal_player=len(focal_matches))

    salt = child_seed("holdout_match_order")
    rows["order_key"] = [order_key(salt, p, m) for p, m in zip(rows["puuid"], rows["match_id"])]
    rows["standard_position"] = rows["team_position"].isin(POSITIONS)

    summary = {**stage, "variants": {}}
    union = []
    for variant, subset in [("primary", rows), ("strict", rows[~rows["contains_focal"]])]:
        eligible, chosen = select(subset, players, variant)
        union.append(chosen[["match_id"]])
        overlap_players = len(set(eligible["puuid"]) & focal)
        overlap_matches = len(set(chosen["match_id"]) & fixed_matches)
        assert overlap_players == 0 and overlap_matches == 0
        summary["variants"][variant] = {
            "eligible_players": int(len(eligible)),
            "eligible_players_by_region": eligible.groupby("region").size().to_dict(),
            "rows": int(len(chosen)),
            "unique_matches": int(chosen["match_id"].nunique()),
            "s1_unique_matches": int(chosen.loc[chosen["in_s1"], "match_id"].nunique()),
            "rows_in_matches_with_focal_player": int(chosen["contains_focal"].sum()),
            "overlap_with_focal_players": overlap_players,
            "overlap_with_fixed_sample_matches": overlap_matches,
        }
        log(f"{variant}: {len(eligible):,} players; {chosen['match_id'].nunique():,} matches")

    pd.concat(union).drop_duplicates().to_parquet(OUT / "holdout_parse_matches.parquet", index=False)
    summary["rules"] = {
        "primary": "exclude fixed-sample matches",
        "strict": "also exclude matches containing a discovery focal player (sensitivity)",
        "min_remaining_matches": MIN_MATCHES,
        "s1": "20 matches, standard positions first, hash order",
        "s2": "outcome set = first min(20, n // 2) in hash order; remaining matches estimate habits",
        "selection_seed_namespace": "holdout_match_order",
    }
    write_json(TABLES / "qa" / "holdout_construction.json", summary)


if __name__ == "__main__":
    main()
