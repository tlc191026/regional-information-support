"""Index timeline payloads for fixed-sample and held-out matches from the current database."""
from __future__ import annotations

import pandas as pd

from config import DATA, MATCH_METADATA, TABLES, ensure_dirs, log, write_json
from db_extract import connect_scratch, load_ids


def downloads_for(con, match_ids) -> pd.DataFrame:
    load_ids(con, "tids", match_ids, "match_id")
    return pd.read_sql_query(
        "SELECT d.match_id, d.status, d.shard_path, d.shard_line FROM scratch.tids AS t "
        "JOIN match_timeline_downloads AS d ON d.match_id = t.match_id",
        con,
    )


def main() -> None:
    ensure_dirs()
    fixed_ids = pd.read_parquet(MATCH_METADATA, columns=["match_id"])["match_id"]
    holdout_ids = pd.read_parquet(DATA / "holdout" / "holdout_parse_matches.parquet")["match_id"].unique()
    assert not set(fixed_ids) & set(holdout_ids)
    con = connect_scratch()
    frames = []
    for sample, ids in [("fixed", fixed_ids), ("holdout", holdout_ids)]:
        found = downloads_for(con, ids).assign(sample=sample)
        log(f"{sample}: {len(ids):,} matches; index status {found['status'].value_counts().to_dict()}")
        frames.append(found)
    con.close()
    targets = pd.concat(frames, ignore_index=True)
    summary = targets.groupby(["sample", "status"]).size().rename("matches").reset_index()
    summary.to_csv(TABLES / "qa" / "timeline_target_status.csv", index=False)
    targets = targets[targets["status"].eq("downloaded")].drop(columns="status")
    targets["shard_line"] = targets["shard_line"].astype(int)
    assert targets["match_id"].is_unique
    targets.to_parquet(DATA / "timeline_targets_ijim.parquet", index=False)
    write_json(TABLES / "qa" / "timeline_targets.json", {
        "downloaded_by_sample": targets.groupby("sample").size().to_dict(),
        "shards": int(targets["shard_path"].nunique()),
    })


if __name__ == "__main__":
    main()
