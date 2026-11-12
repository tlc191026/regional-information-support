"""Approximate within-server percentiles from archived public rank distributions.

The archived pages (2025 snapshots) describe the population tracked by a third-party site,
not Riot's full ladder at list collection; percentiles are used for sensitivity analyses only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config import CANONICAL_LINKS, DATA, TABLES, TIERS, ensure_dirs, log, write_json
from db_extract import connect_scratch

STATIC = DATA / "static"
ORDER = [(t, d) for t in ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
         for d in ["IV", "III", "II", "I"]] + [("MASTER", ""), ("GRANDMASTER", ""), ("CHALLENGER", "")]
ROW = re.compile(r'<td class="nowrap">.*?title="([A-Za-z]+)(?: ([IV]+))?".*?</td>\s*<td class="text-center">\s*'
                 r'<i>([\d.]+)%</i>', re.S)


def parse(region: str) -> pd.DataFrame:
    text = (STATIC / f"rank_distribution_{region}.html").read_text(encoding="utf-8", errors="replace")
    found = {}
    for tier, division, share in ROW.findall(text):
        key = (tier.upper(), division or "")
        found.setdefault(key, float(share))
    rows = [{"region": region, "tier": t, "division": d, "share_pct": found.get((t, d), np.nan)} for t, d in ORDER]
    frame = pd.DataFrame(rows)
    frame["share"] = frame["share_pct"] / frame["share_pct"].sum()
    frame["cum_before"] = frame["share"].cumsum() - frame["share"]
    return frame


def main() -> None:
    ensure_dirs()
    dist = pd.concat([parse(r) for r in ["NA1", "EUW1", "KR"]], ignore_index=True)
    dist.to_csv(TABLES / "ladder_distribution_snapshots.csv", index=False)
    missing = dist[dist["share_pct"].isna()]
    log(f"parsed {len(dist)} rows; missing {len(missing)}; totals {dist.groupby('region')['share_pct'].sum().round(2).to_dict()}")

    focal = pd.read_parquet(CANONICAL_LINKS, columns=["puuid"]).drop_duplicates()
    holdout = pd.read_parquet(DATA / "holdout" / "primary" / "holdout_players.parquet", columns=["puuid"])
    con = connect_scratch()
    players = pd.read_sql_query("SELECT puuid, region, tier, division, league_points FROM players", con)
    con.close()
    players = players[players["puuid"].isin(set(focal["puuid"]) | set(holdout["puuid"])) & players["tier"].isin(TIERS)]
    merged = players.merge(dist, on=["region", "tier", "division"], how="left", validate="many_to_one")
    lp = merged["league_points"].clip(0, 99).fillna(50) / 100
    merged["percentile"] = merged["cum_before"] + merged["share"] * lp
    merged["percentile_mid"] = merged["cum_before"] + merged["share"] / 2
    merged[["puuid", "region", "tier", "division", "league_points", "percentile", "percentile_mid"]].to_parquet(
        DATA / "player_percentiles.parquet", index=False)
    summary = merged.groupby(["region", "tier"])["percentile_mid"].agg(["min", "max"]).round(4)
    summary.to_csv(TABLES / "ladder_percentile_ranges.csv")
    write_json(TABLES / "qa" / "ladder_percentiles.json", {
        "players": int(len(merged)), "missing_percentile": int(merged["percentile"].isna().sum()),
        "snapshot_dates": {"KR": "2025-11-12", "EUW1": "2025-10-30", "NA1": "2025-04-23"},
        "caveat": "third-party tracked population; previous season; sensitivity use only",
    })
    log(summary.to_string())


if __name__ == "__main__":
    main()
