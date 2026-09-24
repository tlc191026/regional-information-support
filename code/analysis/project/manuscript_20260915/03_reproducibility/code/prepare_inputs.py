from settings import *
import sqlite3, platform, shutil
import pandas as pd
import numpy as np


def main():
    before = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or NEW in p.parents:
            continue
        s = p.stat()
        before.append(
            {
                "path": str(p.relative_to(ROOT)),
                "bytes": s.st_size,
                "mtime_ns": s.st_mtime_ns,
            }
        )
    write_json(QA / "outside_workspace_before.json", before)
    meta = pd.read_parquet(
        ROOT / "exp_20260717/data/canonical_unique_match_metadata.parquet"
    )
    assert len(meta) == 2240354 and meta.match_id.is_unique
    meta = meta.rename(
        columns={
            "target_region": "region",
            "target_tier": "tier",
            "target_patch": "patch",
            "target_duration": "game_duration",
        }
    )
    c = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True, timeout=10)
    c.execute("PRAGMA query_only=ON")
    for kind, table in [
        ("timeline", "match_timeline_downloads"),
        ("matchdto", "match_detail_downloads"),
    ]:
        cols = "match_id,status,shard_path,shard_line,downloaded_at" + (
            ",frames_count,events_count" if kind == "timeline" else ""
        )
        idx = pd.read_sql_query(f"SELECT {cols} FROM {table}", c)
        merged = meta.merge(idx, on="match_id", how="left", validate="one_to_one")
        merged["status"] = merged.status.fillna("unindexed")
        merged.to_parquet(DATA / f"{kind}_targets_all.parquet", index=False)
        merged.groupby(["region", "tier", "status"], observed=True).size().rename(
            "n_matches"
        ).reset_index().to_csv(TABLES / f"{kind}_indexed_coverage.csv", index=False)
        paths = []
        for name, g in merged[merged.status.eq("downloaded")].groupby("shard_path"):
            p = Path(name)
            s = p.stat() if p.exists() else None
            paths.append(
                {
                    "path": name,
                    "exists": s is not None,
                    "bytes": s.st_size if s else 0,
                    "mtime_ns": s.st_mtime_ns if s else 0,
                    "targets": len(g),
                }
            )
        write_json(DATA / f"{kind}_source_file_inventory.json", paths)
        log(f"{kind}: {merged.status.value_counts().to_dict()}, {len(paths)} shards")
        del idx, merged
    c.close()
    # This compact parquet retains the original fixed sample's player linkage.
    focus = pd.read_parquet(
        ROOT / "exp_20260717/data/context_unresidualized_participant_scores.parquet",
        columns=["match_id", "puuid", "region", "tier"],
    )
    # Restore the complete linkage if the compact input has fewer records.
    if len(focus) != 2520000:
        full = pd.read_pickle(
            ROOT / "outputs/study0/data/study0_canonical_participant_match_sample.pkl"
        )
        focus = full[["match_id", "puuid", "region", "tier"]].copy()
        del full
    focus = focus.drop_duplicates(["match_id", "puuid"])
    assert (
        len(focus) == 2520000
        and focus.puuid.nunique() == 126000
        and focus.groupby("puuid").size().eq(20).all()
    )
    focus.to_parquet(DATA / "canonical_focal_links.parquet", index=False)
    players = (
        focus[["puuid", "region", "tier"]].drop_duplicates("puuid").sort_values("puuid")
    )
    players["split"] = "train"
    for (r, t), g in players.groupby(["region", "tier"], observed=True):
        ids = np.random.default_rng(child_seed(f"player_split_{r}_{t}")).permutation(
            g.index
        )
        players.loc[ids[: round(len(ids) * 0.30)], "split"] = "test"
    linked = focus[["match_id", "puuid"]].merge(
        players[["puuid", "split"]], on="puuid", validate="many_to_one"
    )
    grouped = linked.groupby("match_id").split.agg(["first", "nunique"])
    grouped["focal_split"] = np.where(
        grouped["nunique"].eq(1), grouped["first"], "mixed"
    )
    grouped[["focal_split"]].reset_index().to_parquet(
        DATA / "focal_player_match_split.parquet", index=False
    )
    # Never expose raw PUUIDs in public tables.
    write_json(
        QA / "focal_player_split.json",
        {
            "players": len(players),
            "focal_records": len(focus),
            "match_split_counts": grouped.focal_split.value_counts().to_dict(),
            "mixed_matches_excluded_from_prediction": True,
            "same_focal_player_across_retained_sets": False,
        },
    )
    old = pd.read_parquet(
        ROOT
        / "outputs/study1_information_support_validation/data/study1_match_difference_table_scored.parquet",
        columns=["match_id", "window", "tier"],
    )
    old[old.window.eq("0_10")][["match_id", "tier"]].drop_duplicates(
        "match_id"
    ).to_parquet(DATA / "old_timeline_membership.parquet", index=False)
    write_json(
        REPRO / "run_config.json",
        {
            "seed": SEED,
            "primary_window_ms": 600000,
            "sensitivity_window_ms": 900000,
            "canonical_players": 126000,
            "canonical_focal_records": 2520000,
            "canonical_unique_matches": 2240354,
            "regions": REGIONS,
            "tiers": TIERS,
            "valid_ward_types": sorted(VALID_WARDS),
            "components": [
                "valid_wards_placed",
                "valid_ward_kills",
                "control_ward_purchases",
            ],
            "model": "logistic blue-minus-red, match-level nonparametric score bootstrap",
            "bootstrap_replicates": 1000,
            "preregistered": False,
            "source_database_read_only": str(DB),
            "parser_version": PARSER_VERSION,
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    )
    log("Input preparation finished")


if __name__ == "__main__":
    main()
