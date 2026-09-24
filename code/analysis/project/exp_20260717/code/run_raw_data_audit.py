"""Audit raw MatchDto dates/patches and sampled timeline event semantics."""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import DIRS, ROOT, child_seed, ensure_dirs, rng, write_json
from constructs import MASTER_SEED, REGION_ORDER

RAW = ROOT / "lol_data_snapshot_20260708_1404" / "raw"
DB = ROOT / "lol_data_snapshot_20260708_1404" / "database" / "lol_data.db"
META = DIRS["data"] / "canonical_unique_match_metadata.parquet"


DATE_SCHEMA = pa.schema(
    [
        ("match_id", pa.string()),
        ("region", pa.string()),
        ("game_creation_ms", pa.int64()),
        ("game_start_ms", pa.int64()),
        ("game_end_ms", pa.int64()),
        ("game_version_raw", pa.string()),
        ("patch", pa.string()),
        ("queue_id", pa.int32()),
        ("game_duration", pa.int32()),
    ]
)


def patch_from_version(version: str) -> str:
    fields = str(version or "").split(".")
    return ".".join(fields[:2]) if len(fields) >= 2 else str(version or "")


def flush_rows(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=DATE_SCHEMA))
        rows.clear()


def audit_matchdto_region(region: str, target_ids: set[str]) -> dict:
    out = DIRS["data_audit"] / f"canonical_raw_match_dates_{region}.parquet"
    if out.exists():
        cached = pd.read_parquet(out, columns=["match_id"])
        if len(cached) == len(target_ids) and cached.match_id.nunique() == len(
            target_ids
        ):
            return {
                "region": region,
                "target_matches": len(target_ids),
                "found_matches": len(cached),
                "cache_used": True,
            }
        out.unlink()

    files = sorted((RAW / "match_details" / region).glob("matchdto_*.jsonl.gz"))
    writer = pq.ParquetWriter(out, DATE_SCHEMA, compression="zstd")
    rows: list[dict] = []
    found = set()
    malformed = 0
    for file_no, path in enumerate(files, 1):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    wrapper = json.loads(line)
                    match_id = str(wrapper.get("match_id") or "")
                    if match_id not in target_ids:
                        continue
                    info = (wrapper.get("match") or {}).get("info") or {}
                    rows.append(
                        {
                            "match_id": match_id,
                            "region": region,
                            "game_creation_ms": int(info.get("gameCreation") or 0),
                            "game_start_ms": int(info.get("gameStartTimestamp") or 0),
                            "game_end_ms": int(info.get("gameEndTimestamp") or 0),
                            "game_version_raw": str(info.get("gameVersion") or ""),
                            "patch": patch_from_version(info.get("gameVersion")),
                            "queue_id": int(info.get("queueId") or 0),
                            "game_duration": int(info.get("gameDuration") or 0),
                        }
                    )
                    found.add(match_id)
                    if len(rows) >= 100_000:
                        flush_rows(writer, rows)
                except Exception:
                    malformed += 1
        if file_no % 20 == 0:
            print(
                json.dumps(
                    {
                        "audit": "matchdto",
                        "region": region,
                        "files": file_no,
                        "found": len(found),
                    }
                ),
                flush=True,
            )
    flush_rows(writer, rows)
    writer.close()
    missing = target_ids - found
    if missing:
        pd.Series(sorted(missing), name="match_id").to_csv(
            DIRS["data_audit"] / f"canonical_raw_match_dates_missing_{region}.csv",
            index=False,
        )
    return {
        "region": region,
        "target_matches": len(target_ids),
        "found_matches": len(found),
        "missing_matches": len(missing),
        "malformed_lines": malformed,
        "n_shards": len(files),
        "cache_used": False,
    }


def summarize_dates() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    meta = pd.read_parquet(META, columns=["match_id", "target_region"])
    audits = []
    region_frames = []
    for region in REGION_ORDER:
        targets = set(meta.loc[meta.target_region.eq(region), "match_id"].astype(str))
        audits.append(audit_matchdto_region(region, targets))
        region_frames.append(
            pd.read_parquet(
                DIRS["data_audit"] / f"canonical_raw_match_dates_{region}.parquet"
            )
        )
    dates = pd.concat(region_frames, ignore_index=True)
    for column in ["game_creation_ms", "game_start_ms", "game_end_ms"]:
        dates[column.replace("_ms", "_utc")] = pd.to_datetime(
            dates[column], unit="ms", utc=True, errors="coerce"
        )
    summary = (
        dates.groupby("region", observed=True)
        .agg(
            n_matches=("match_id", "nunique"),
            first_game_creation_utc=("game_creation_utc", "min"),
            last_game_creation_utc=("game_creation_utc", "max"),
            first_game_start_utc=("game_start_utc", "min"),
            last_game_start_utc=("game_start_utc", "max"),
            n_patches=("patch", "nunique"),
            min_duration_seconds=("game_duration", "min"),
            max_duration_seconds=("game_duration", "max"),
        )
        .reset_index()
    )
    summary.to_csv(
        DIRS["data_audit"] / "canonical_raw_match_date_window_by_region.csv",
        index=False,
    )
    patches = (
        dates.groupby(["region", "patch"], observed=True)
        .size()
        .rename("n_matches")
        .reset_index()
    )
    patches.to_csv(
        DIRS["data_audit"] / "canonical_raw_patch_distribution.csv", index=False
    )
    dates[
        [
            "match_id",
            "region",
            "game_creation_ms",
            "game_start_ms",
            "game_end_ms",
            "patch",
            "queue_id",
        ]
    ].to_parquet(
        DIRS["data_audit"] / "canonical_raw_match_dates_all_regions.parquet",
        index=False,
    )
    return summary, patches, audits


def sampled_timeline_semantic_audit() -> dict:
    audit_json = DIRS["data_audit"] / "sampled_raw_timeline_semantic_audit.json"
    if audit_json.exists():
        return json.loads(audit_json.read_text(encoding="utf-8"))
    canonical = set(pd.read_parquet(META, columns=["match_id"]).match_id.astype(str))
    event_counter = Counter()
    ward_counter = Counter()
    monster_counter = Counter()
    monster_sub_counter = Counter()
    killer_team_counter = Counter()
    participant_mapping_counter = Counter()
    sampled_files = []
    n_lines = n_canonical = n_events = 0
    generator = rng("raw_timeline_shard_audit")
    for region in REGION_ORDER:
        files = sorted((RAW / "timelines" / region).glob("timeline_*.jsonl.gz"))
        selected = files[int(generator.integers(0, len(files)))]
        sampled_files.append(str(selected.relative_to(ROOT)))
        with gzip.open(selected, "rt", encoding="utf-8") as handle:
            for line in handle:
                n_lines += 1
                try:
                    wrapper = json.loads(line)
                except Exception:
                    continue
                match_id = str(wrapper.get("match_id") or "")
                if match_id not in canonical:
                    continue
                n_canonical += 1
                frames = ((wrapper.get("timeline") or {}).get("info") or {}).get(
                    "frames"
                ) or []
                for frame in frames:
                    for event in frame.get("events") or []:
                        event_type = str(event.get("type") or "")
                        event_counter[event_type] += 1
                        n_events += 1
                        if event_type in {"WARD_PLACED", "WARD_KILL"}:
                            ward_counter[
                                (event_type, str(event.get("wardType") or "MISSING"))
                            ] += 1
                            pid = (
                                event.get("creatorId")
                                if event_type == "WARD_PLACED"
                                else event.get("killerId")
                            )
                            try:
                                pid = int(pid)
                                participant_mapping_counter[
                                    (
                                        "team_100"
                                        if 1 <= pid <= 5
                                        else (
                                            "team_200" if 6 <= pid <= 10 else "unmapped"
                                        )
                                    )
                                ] += 1
                            except Exception:
                                participant_mapping_counter["unmapped"] += 1
                        elif event_type == "ELITE_MONSTER_KILL":
                            monster_counter[
                                str(event.get("monsterType") or "MISSING")
                            ] += 1
                            monster_sub_counter[
                                str(event.get("monsterSubType") or "MISSING")
                            ] += 1
                            killer_team_counter[
                                str(event.get("killerTeamId") or "MISSING")
                            ] += 1
    pd.DataFrame(
        [
            {"event_type": a, "ward_type": b, "count": n}
            for (a, b), n in ward_counter.items()
        ]
    ).to_csv(DIRS["data_audit"] / "sampled_raw_timeline_ward_types.csv", index=False)
    pd.DataFrame(
        [{"monster_type": a, "count": n} for a, n in monster_counter.items()]
    ).to_csv(DIRS["data_audit"] / "sampled_raw_timeline_monster_types.csv", index=False)
    pd.DataFrame(
        [{"monster_sub_type": a, "count": n} for a, n in monster_sub_counter.items()]
    ).to_csv(
        DIRS["data_audit"] / "sampled_raw_timeline_monster_subtypes.csv", index=False
    )
    summary = {
        "master_seed": MASTER_SEED,
        "derived_seed": child_seed("raw_timeline_shard_audit"),
        "sampled_shards": sampled_files,
        "raw_lines_read": n_lines,
        "canonical_matches_in_sample": n_canonical,
        "events_audited": n_events,
        "elite_monster_events": int(sum(monster_counter.values())),
        "killer_team_ids": dict(killer_team_counter),
        "participant_to_team_mapping": dict(participant_mapping_counter),
        "interpretation": "Deterministic shard-level parser audit; not an inferential sample.",
    }
    write_json(audit_json, summary)
    return summary


def download_window() -> pd.DataFrame:
    con = sqlite3.connect(DB)
    result = pd.read_sql_query(
        "SELECT region, MIN(downloaded_at) AS first_downloaded_at, MAX(downloaded_at) AS last_downloaded_at, "
        "COUNT(DISTINCT match_id) AS downloaded_matches FROM match_detail_downloads "
        "WHERE status='downloaded' GROUP BY region",
        con,
    )
    con.close()
    result.to_csv(
        DIRS["data_audit"] / "snapshot_matchdto_download_window.csv", index=False
    )
    return result


def main() -> None:
    ensure_dirs()
    started = time.time()
    date_summary, patches, audits = summarize_dates()
    timeline = sampled_timeline_semantic_audit()
    downloads = download_window()
    summary = {
        "master_seed": MASTER_SEED,
        "matchdto_date_audit": audits,
        "date_window_by_region": date_summary.astype(str).to_dict("records"),
        "patch_rows": len(patches),
        "timeline_semantic_audit": timeline,
        "download_window": downloads.astype(str).to_dict("records"),
        "tier_measurement_note": "Tier is the player's sampling-time rank and is not a historical match-time rank.",
        "elapsed_minutes": (time.time() - started) / 60,
    }
    write_json(
        DIRS["reports"] / "raw_data_audit_machine_readable_summary.json", summary
    )
    print(
        json.dumps(
            {"status": "complete", "elapsed_minutes": summary["elapsed_minutes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
