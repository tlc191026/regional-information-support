"""Extract all ten participants for the fixed, held-out and focal-extra match sets (read-only)."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time

import pandas as pd

from config import CANONICAL_LINKS, DATA, MATCH_METADATA, TABLES, ensure_dirs, log, write_json
from db_extract import (PARTICIPANT_COLUMNS, add_patch, connect_scratch, matches_for_ids, participants_for_matches,
                        valid_matches_for_players)

OUT = DATA / "participants"
CHUNK = 50000


def extract_chunk(job) -> dict:
    label, index, match_ids = job
    dest = OUT / label / f"part-{index:04d}.parquet"
    if dest.exists():
        return {"index": index, "rows": None, "cached": True}
    con = connect_scratch()
    try:
        parts = participants_for_matches(con, match_ids, PARTICIPANT_COLUMNS)
        meta = add_patch(matches_for_ids(con, match_ids))
    finally:
        con.close()
    frame = parts.merge(meta[["match_id", "region", "game_duration", "game_version", "patch", "queue_id"]],
                        on="match_id", how="left", validate="many_to_one")
    tmp = dest.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(dest)
    return {"index": index, "rows": len(frame), "matches": int(frame["match_id"].nunique()), "cached": False}


def focal_valid_matches() -> pd.DataFrame:
    path = DATA / "focal_valid_matches.parquet"
    if path.exists():
        return pd.read_parquet(path)
    links = pd.read_parquet(CANONICAL_LINKS, columns=["match_id", "puuid", "region", "tier"])
    players = links.drop_duplicates("puuid")[["puuid", "region", "tier"]]
    con = connect_scratch()
    rows = valid_matches_for_players(con, players)
    con.close()
    own = links[["puuid", "match_id"]].assign(in_own_fixed20=True)
    rows = rows.merge(own, on=["puuid", "match_id"], how="left")
    rows["in_own_fixed20"] = rows["in_own_fixed20"].fillna(False).astype(bool)
    missing = own.merge(rows[["puuid", "match_id"]], on=["puuid", "match_id"], how="left", indicator=True)
    write_json(TABLES / "qa" / "focal_valid_matches.json", {
        "rows": len(rows), "players": int(rows["puuid"].nunique()),
        "fixed_rows_found": int(rows["in_own_fixed20"].sum()),
        "fixed_rows_not_found_in_current_db": int(missing["_merge"].eq("left_only").sum()),
        "extra_rows_per_player": rows.loc[~rows["in_own_fixed20"]].groupby("puuid").size().describe().round(2).to_dict(),
    })
    rows.to_parquet(path, index=False)
    return rows


def match_set(label: str) -> list[str]:
    fixed = pd.read_parquet(MATCH_METADATA, columns=["match_id"])["match_id"]
    if label == "fixed":
        return sorted(fixed.astype(str))
    if label == "holdout":
        held = pd.read_parquet(DATA / "holdout" / "holdout_parse_matches.parquet")
        return sorted(held["match_id"].astype(str).unique())
    rows = focal_valid_matches()
    extra = set(rows.loc[~rows["in_own_fixed20"], "match_id"]) - set(fixed)
    return sorted(extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["fixed", "holdout", "focal_extra"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ensure_dirs()
    (OUT / args.set).mkdir(parents=True, exist_ok=True)
    ids = match_set(args.set)
    jobs = [(args.set, i // CHUNK, ids[i:i + CHUNK]) for i in range(0, len(ids), CHUNK)]
    log(f"{args.set}: {len(ids):,} matches in {len(jobs)} chunks")
    start = time.monotonic()
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(extract_chunk, jobs):
            results.append(result)
            if len(results) % 5 == 0 or len(results) == len(jobs):
                log(f"{args.set}: {len(results)}/{len(jobs)} chunks; {(time.monotonic() - start) / 60:.1f} min")
    frame = pd.read_parquet(OUT / args.set, columns=["match_id"])
    per_match = frame.groupby("match_id").size()
    write_json(TABLES / "qa" / f"participants_{args.set}.json", {
        "requested_matches": len(ids), "matches_found": int(per_match.size),
        "rows": int(len(frame)), "matches_without_ten_rows": int(per_match.ne(10).sum()),
        "seconds": time.monotonic() - start,
    })
    log(f"{args.set}: {per_match.size:,} matches, {len(frame):,} rows")


if __name__ == "__main__":
    main()
