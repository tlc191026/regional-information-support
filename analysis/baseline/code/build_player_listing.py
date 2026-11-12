"""Listing-time ladder information (division, LP, season wins and losses) for focal and holdout players."""
from __future__ import annotations

import pandas as pd

from config import CANONICAL_LINKS, DATA, REGIONS, TABLES, ensure_dirs, log, write_json
from db_extract import connect_scratch

DIVISION_NUM = {"I": 1, "II": 2, "III": 3, "IV": 4}


def main() -> None:
    ensure_dirs()
    focal = pd.read_parquet(CANONICAL_LINKS, columns=["puuid", "region", "tier"]).drop_duplicates("puuid")
    focal["sample"] = "discovery"
    holdout = pd.read_parquet(DATA / "holdout" / "primary" / "holdout_players.parquet", columns=["puuid", "region", "tier"])
    holdout["sample"] = "holdout"
    wanted = pd.concat([focal, holdout], ignore_index=True)

    con = connect_scratch()
    con.execute("CREATE TABLE scratch.wanted (puuid TEXT PRIMARY KEY)")
    con.executemany("INSERT INTO scratch.wanted VALUES (?)", wanted[["puuid"]].itertuples(index=False, name=None))
    rows = pd.read_sql_query(
        "SELECT p.puuid, p.region, p.tier, p.division, p.league_points, p.wins, p.losses, p.created_at "
        "FROM scratch.wanted w CROSS JOIN players p ON p.puuid = w.puuid "
        f"WHERE p.region IN ({','.join('?' * len(REGIONS))})", con, params=REGIONS)
    con.close()
    rows["region"] = rows["region"].replace({"NA": "NA1", "EUW": "EUW1"})
    dup = int(rows.duplicated(["puuid", "region"]).sum())
    rows = rows.sort_values(["puuid", "region", "created_at"]).drop_duplicates(["puuid", "region"], keep="first")
    out = wanted.merge(rows.drop(columns=["tier"]), on=["puuid", "region"], how="left")
    out["division_num"] = out["division"].map(DIVISION_NUM)
    games = out["wins"] + out["losses"]
    out["ladder_games"] = games
    out["ladder_win_rate"] = out["wins"] / games.where(games > 0)
    out.to_parquet(DATA / "player_listing.parquet", index=False)
    qa = {
        "players": int(len(out)), "missing_listing": int(out["division"].isna().sum()),
        "duplicate_listing_rows_dropped": dup,
        "by_sample_region": out.groupby(["sample", "region"]).agg(
            n=("puuid", "size"), lp_mean=("league_points", "mean"), games_median=("ladder_games", "median"),
        ).reset_index().to_dict(orient="records"),
    }
    write_json(TABLES / "qa" / "player_listing.json", qa)
    log(qa)


if __name__ == "__main__":
    main()
