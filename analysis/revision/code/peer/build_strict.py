"""Build a common strict holdout for cross-match tendency analyses."""
from __future__ import annotations

import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"

import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import build_c1_data as build
from config import DATA, REGIONS, POSITIONS, PING_FIELDS, child_seed, write_json, log

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "peer" / "strict"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    links = pd.read_parquet(DATA / "holdout" / "strict" / "holdout_player_matches.parquet")
    links["pid"] = build.pid_hash(links.puuid)
    links["mnum"] = build.match_number(links.match_id)
    assert not links.contains_focal.any()
    eligible = set(links.pid)
    player_tier = links.drop_duplicates("pid").set_index("pid").tier
    type_frames = []
    qa = {"eligible_players": len(eligible), "unique_matches": links.match_id.nunique(),
          "source": str(DATA / "holdout" / "strict" / "holdout_player_matches.parquet"),
          "duo_definition": "Repeated co-team counts from the original complete holdout history, including fixed matches; identical screen across variants."}
    dataset = ds.dataset(DATA / "participants" / "holdout", format="parquet")
    cols = ["match_id", "puuid", "team_id", "team_position", "champion_id", "game_duration", "patch", "region", *build.SKILL, *build.COUNT_FIELDS]
    for region in REGIONS:
        region_links = links[links.region.eq(region)]
        ids = region_links.match_id.drop_duplicates().tolist()
        log(f"Strict {region}: reading {len(ids):,} matches")
        table = dataset.to_table(columns=cols, filter=(ds.field("region") == region) & ds.field("match_id").isin(ids), use_threads=False)
        part = table.to_pandas().drop_duplicates(["match_id", "puuid"])
        del table
        part["pid"] = build.pid_hash(part.puuid)
        part["mnum"] = build.match_number(part.match_id)
        part["team_position"] = part.team_position.where(part.team_position.isin(POSITIONS), "NONE")
        for col in build.COUNT_FIELDS:
            part[col] = part[col].astype(np.int16)
        assert part.groupby("match_id").size().eq(10).all()
        pids = set(region_links.pid)
        part = part.drop(columns=["puuid", "region"])

        est = region_links.loc[region_links.s2_set.eq("type"), ["pid", "match_id"]].merge(part, on=["pid", "match_id"], how="inner", validate="one_to_one")
        est["region"], est["tier"] = region, est.pid.map(player_tier)
        type_frames.append(est)
        outcome = region_links.loc[region_links.s2_set.eq("outcome"), ["pid", "match_id", "tier"]].rename(columns={"pid": "focal_pid", "tier": "focal_tier"})
        focal = outcome.merge(part, left_on=["focal_pid", "match_id"], right_on=["pid", "match_id"], how="inner", validate="one_to_one")
        focal = focal[["focal_pid", "match_id", "focal_tier", "team_id", "team_position", "champion_id"]].rename(columns={"team_id": "focal_team", "team_position": "focal_position", "champion_id": "focal_champion"})
        rows = focal.merge(part, on="match_id", how="inner")
        rows["relation"] = np.where(rows.pid.eq(rows.focal_pid), "self", np.where(rows.team_id.eq(rows.focal_team), "teammate", "opponent"))
        duo = pd.read_parquet(DATA / "c1_holdout" / f"duo_counts_{region}.parquet", columns=["focal_pid", "pid", "n_same_team", "n_opponent"])
        duo = duo[duo.focal_pid.isin(pids)]
        rows = rows.merge(duo, on=["focal_pid", "pid"], how="left", validate="many_to_one")
        rows[["n_same_team", "n_opponent"]] = rows[["n_same_team", "n_opponent"]].fillna(0).astype(np.int32)
        rows["other_is_focal"] = rows.pid.isin(eligible) & rows.relation.ne("self")
        rows["region"] = region
        rows.to_parquet(OUT / f"outcome_rows_{region}.parquet", index=False)
        qa[region] = {"players": len(pids), "matches": len(ids), "outcome_focal_rows": len(focal), "outcome_matches": focal.match_id.nunique(), "habit_rows": len(est), "relation_rows": rows.relation.value_counts().to_dict()}
        log(f"Strict {region}: {qa[region]}")
        del rows, part, focal, outcome, duo
        gc.collect()

    est = pd.concat(type_frames, ignore_index=True)
    del type_frames
    est = pd.concat([est, build.rates(est)], axis=1)
    metrics = list(build.TYPE_RATES)
    bounds = {}
    for metric in metrics + build.SKILL_RATES:
        lo, hi = np.nanquantile(est[metric], [.01, .99])
        est[metric] = est[metric].clip(lo, hi)
        bounds[metric] = [float(lo), float(hi)]
    est["half"] = build.half_of(child_seed("c1_type_halves"), est.pid.to_numpy(), est.mnum.to_numpy())
    resid_cols = metrics + build.SKILL_RATES
    residuals = build.residualize(est[["region", "tier", "team_position", "patch", "game_duration", *resid_cols]], resid_cols)
    est[[f"{m}_resid" for m in resid_cols]] = residuals[resid_cols].to_numpy()
    keep = ["pid", "mnum", "match_id", "region", "tier", "team_position", "champion_id", "patch", "game_duration", "half", *metrics, *[f"{m}_resid" for m in metrics]]
    est[keep].to_parquet(OUT / "type_rows.parquet", index=False)
    agg = {f"{m}_resid": "mean" for m in resid_cols}
    agg.update(win="mean", kda="mean", gold_pm="mean", damage_pm="mean", mnum="size")
    types = est.groupby("pid").agg(agg).rename(columns={"mnum": "n_type_matches", "win": "win_rate"})
    halves = est.pivot_table(index="pid", columns="half", values=[f"{m}_resid" for m in metrics], aggfunc="mean")
    halves.columns = [f"{m}_{h}" for m,h in halves.columns]
    types = types.join(halves).reset_index().join(links.drop_duplicates("pid").set_index("pid")[["region", "tier"]], on="pid")
    assert len(types) == 23193 and types.n_type_matches.ge(10).all()
    types.to_parquet(OUT / "focal_types.parquet", index=False)
    qa["types"] = {"players": len(types), "minimum_habit_matches": int(types.n_type_matches.min()), "winsorization_bounds": bounds}
    write_json(ROOT / "qa" / "peer_strict_build.json", qa)
    log("Strict build complete")


if __name__ == "__main__":
    main()
