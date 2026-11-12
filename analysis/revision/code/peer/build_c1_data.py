"""Build Study 2 (C1) inputs: habit estimation rows, duo counts, outcome rows and focal pairs.

Only data engineering happens here. No association between a focal player's habits and teammates'
behaviour is computed. ``--sample discovery`` uses the fixed sample (outcome matches = each focal
player's 20 fixed matches; habits from that player's other valid matches). ``--sample holdout`` uses
the primary held-out players (outcome and habit matches as assigned in build_holdout_sample.py).
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from config import (CANONICAL_LINKS, DATA, PING_FIELDS, PING_GROUPS, POSITIONS, REGIONS, TABLES, child_seed,
                    ensure_dirs, log, write_json)

COUNT_FIELDS = ["control_wards_bought", "wards_killed", "wards_placed", *PING_FIELDS]
TYPE_RATES = {
    "control_wards_pm": ["control_wards_bought"],
    "wards_killed_pm": ["wards_killed"],
    "wards_placed_pm": ["wards_placed"],
    "initiation_p10": PING_GROUPS["initiation"],
    "caution_p10": PING_GROUPS["caution"],
    "total_pings_p10": PING_FIELDS,
}
SKILL = ["win", "kills", "deaths", "assists", "gold_earned", "total_damage_dealt"]
SKILL_RATES = ["kda", "gold_pm", "damage_pm"]


def out_dir(sample: str):
    return TABLES.parents[1] / "data" / "peer" / f"base_{sample}"


def pid_hash(values: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(np.uint64)


def match_number(values: pd.Series) -> np.ndarray:
    return values.str.split("_").str[-1].astype(np.int64).to_numpy()


def load_region(region: str, sources: list[str]) -> pd.DataFrame:
    cols = ["match_id", "puuid", "team_id", "team_position", "champion_id", "game_duration", "patch", "region",
            *SKILL, *COUNT_FIELDS]
    parts = [pd.read_parquet(DATA / "participants" / name, columns=cols, filters=[("region", "==", region)])
             for name in sources]
    frame = pd.concat(parts, ignore_index=True).drop_duplicates(["match_id", "puuid"])
    frame["pid"] = pid_hash(frame["puuid"])
    frame["mnum"] = match_number(frame["match_id"])
    frame["team_position"] = frame["team_position"].where(frame["team_position"].isin(POSITIONS), "NONE")
    for col in COUNT_FIELDS:
        frame[col] = frame[col].astype(np.int16)
    return frame.drop(columns=["match_id", "puuid", "region"])


def half_of(salt: int, pid: np.ndarray, mnum: np.ndarray) -> np.ndarray:
    keys = np.array([int.from_bytes(hashlib.blake2b(f"{salt}:{a}:{b}".encode(), digest_size=8).digest(), "little")
                     for a, b in zip(pid, mnum)], dtype=np.uint64)
    order = pd.DataFrame({"pid": pid, "key": keys}).sort_values(["pid", "key"])
    rank = order.groupby("pid").cumcount()
    half = pd.Series(np.where(rank % 2 == 0, "A", "B"), index=order.index)
    return half.sort_index().to_numpy()


def rates(frame: pd.DataFrame) -> pd.DataFrame:
    minutes = frame["game_duration"].to_numpy(float) / 60.0
    out = {}
    for name, fields in TYPE_RATES.items():
        scale = 10.0 if name.endswith("_p10") else 1.0
        out[name] = frame[fields].sum(axis=1).to_numpy(float) / minutes * scale
    out["kda"] = ((frame["kills"] + frame["assists"]) / frame["deaths"].clip(lower=1)).to_numpy(float)
    out["gold_pm"] = frame["gold_earned"].to_numpy(float) / minutes
    out["damage_pm"] = frame["total_damage_dealt"].to_numpy(float) / minutes
    return pd.DataFrame(out, index=frame.index)


def residualize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Within-region OLS residuals on position, listing tier, patch and duration."""
    out = frame.copy()
    for region, idx in frame.groupby("region").groups.items():
        sub = frame.loc[idx]
        blocks = []
        for col in ["team_position", "tier", "patch"]:
            codes, uniques = pd.factorize(sub[col].astype(str))
            blocks.append(sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(uniques))))
        duration = (sub["game_duration"].to_numpy(float) - sub["game_duration"].mean()) / sub["game_duration"].std()
        intercept = sp.csr_matrix(np.ones((len(sub), 1)))
        x = sp.hstack([intercept] + blocks + [sp.csr_matrix(duration.reshape(-1, 1))], format="csr")
        for metric in metrics:
            y = sub[metric].to_numpy(float)
            model = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False).fit(x, y)
            out.loc[idx, metric] = y - model.predict(x)
    return out


def sample_links(sample: str) -> dict[str, pd.DataFrame]:
    """Player-match links for outcome matches, habit matches and duo counting (pid, mnum, region, tier)."""
    if sample == "discovery":
        links = pd.read_parquet(CANONICAL_LINKS)
        links["pid"] = pid_hash(links["puuid"])
        links["mnum"] = match_number(links["match_id"])
        tier = links.drop_duplicates("pid").set_index("pid")["tier"]
        valid = pd.read_parquet(DATA / "focal_valid_matches.parquet")
        valid["pid"] = pid_hash(valid["puuid"])
        valid["mnum"] = match_number(valid["match_id"])
        valid["tier"] = valid["pid"].map(tier)
        return {
            "outcome": links[["pid", "mnum", "region", "tier"]],
            "habit": valid.loc[~valid["in_own_fixed20"], ["pid", "mnum", "region", "tier"]],
            "duo": valid[["pid", "mnum", "region"]],
            "sources": ["fixed", "focal_extra"],
        }
    rows = pd.read_parquet(DATA / "holdout" / "primary" / "holdout_player_matches.parquet")
    rows["pid"] = pid_hash(rows["puuid"])
    rows["mnum"] = match_number(rows["match_id"])
    fixed = pd.read_parquet(DATA / "participants" / "fixed", columns=["match_id", "puuid", "region"])
    fixed = fixed[fixed["puuid"].isin(set(rows["puuid"]))]
    fixed = fixed.assign(pid=pid_hash(fixed["puuid"]), mnum=match_number(fixed["match_id"]))
    duo = pd.concat([rows[["pid", "mnum", "region"]], fixed[["pid", "mnum", "region"]]]).drop_duplicates()
    return {
        "outcome": rows.loc[rows["s2_set"].eq("outcome"), ["pid", "mnum", "region", "tier"]],
        "habit": rows.loc[rows["s2_set"].eq("type"), ["pid", "mnum", "region", "tier"]],
        "duo": duo,
        "sources": ["holdout", "fixed"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["discovery", "holdout"], default="discovery")
    args = ap.parse_args()
    ensure_dirs()
    out = out_dir(args.sample)
    out.mkdir(parents=True, exist_ok=True)
    links = sample_links(args.sample)
    outcome_all, habit_all, duo_all = links["outcome"], links["habit"], links["duo"]
    sample_pids = set(outcome_all["pid"])
    player_tier = outcome_all.drop_duplicates("pid").set_index("pid")["tier"]

    type_frames, qa = [], {"sample": args.sample}
    for region in REGIONS:
        log(f"{args.sample} {region}: loading participants")
        part = load_region(region, links["sources"])
        est = habit_all.loc[habit_all["region"].eq(region), ["pid", "mnum"]].merge(part, on=["pid", "mnum"], how="inner")
        est["region"] = region
        est["tier"] = est["pid"].map(player_tier)
        type_frames.append(est)

        own = duo_all.loc[duo_all["region"].eq(region), ["pid", "mnum"]].merge(part[["pid", "mnum", "team_id"]],
                                                                              on=["pid", "mnum"], how="inner")
        focal_team = own.rename(columns={"pid": "focal_pid", "team_id": "focal_team"})
        everyone = focal_team.merge(part[["mnum", "pid", "team_id"]], on="mnum")
        everyone = everyone[everyone["pid"] != everyone["focal_pid"]]
        everyone["same_team"] = everyone["team_id"].eq(everyone["focal_team"])
        duo = everyone.groupby(["focal_pid", "pid"])["same_team"].agg(n_same_team="sum", n_together="size").reset_index()
        duo["n_opponent"] = duo["n_together"] - duo["n_same_team"]
        duo.to_parquet(out / f"duo_counts_{region}.parquet", index=False)

        outcome = outcome_all.loc[outcome_all["region"].eq(region), ["pid", "mnum", "tier"]].rename(
            columns={"pid": "focal_pid", "tier": "focal_tier"})
        focal_rows = outcome.merge(part, left_on=["focal_pid", "mnum"], right_on=["pid", "mnum"], how="inner")
        focal_rows = focal_rows[["focal_pid", "mnum", "focal_tier", "team_id", "team_position", "champion_id"]].rename(
            columns={"team_id": "focal_team", "team_position": "focal_position", "champion_id": "focal_champion"})
        rows = focal_rows.merge(part, on="mnum", how="inner")
        rows["relation"] = np.where(rows["pid"].eq(rows["focal_pid"]), "self",
                                    np.where(rows["team_id"].eq(rows["focal_team"]), "teammate", "opponent"))
        rows = rows.merge(duo[["focal_pid", "pid", "n_same_team", "n_opponent"]], on=["focal_pid", "pid"], how="left")
        rows[["n_same_team", "n_opponent"]] = rows[["n_same_team", "n_opponent"]].fillna(0).astype(np.int32)
        rows["other_is_focal"] = rows["pid"].isin(sample_pids) & rows["relation"].ne("self")
        rows["region"] = region
        rows.to_parquet(out / f"outcome_rows_{region}.parquet", index=False)

        pairs = outcome.rename(columns={"focal_pid": "pid"})[["pid", "mnum"]].merge(part[["pid", "mnum", "team_id"]],
                                                                                    on=["pid", "mnum"])
        multi = pairs[pairs.duplicated("mnum", keep=False)]
        pair = multi.merge(multi, on="mnum", suffixes=("_a", "_b"))
        pair = pair[pair["pid_a"] < pair["pid_b"]]
        pair["relation"] = np.where(pair["team_id_a"].eq(pair["team_id_b"]), "teammate", "opponent")
        pair[["mnum", "pid_a", "pid_b", "relation"]].assign(region=region).to_parquet(
            out / f"focal_pairs_{region}.parquet", index=False)

        qa[region] = {
            "participant_rows": int(len(part)), "focal_rows_found": int(len(focal_rows)),
            "outcome_rows": int(len(rows)), "relation_counts": rows["relation"].value_counts().to_dict(),
            "estimation_rows": int(len(est)), "focal_pairs": pair["relation"].value_counts().to_dict(),
            "outcome_players_missing": int(outcome["focal_pid"].nunique() - focal_rows["focal_pid"].nunique()),
        }
        log(f"{region}: {qa[region]}")
        del part, own, everyone, rows

    est = pd.concat(type_frames, ignore_index=True)
    est = pd.concat([est, rates(est)], axis=1)
    metrics = list(TYPE_RATES)
    for metric in metrics + SKILL_RATES:
        lo, hi = np.nanquantile(est[metric], [0.01, 0.99])
        est[metric] = est[metric].clip(lo, hi)
    est["half"] = half_of(child_seed("c1_type_halves"), est["pid"].to_numpy(), est["mnum"].to_numpy())
    resid_cols = metrics + SKILL_RATES
    resid = residualize(est[["region", "tier", "team_position", "patch", "game_duration", *resid_cols]], resid_cols)
    est[[f"{m}_resid" for m in resid_cols]] = resid[resid_cols].to_numpy()
    keep_rows = ["pid", "mnum", "region", "tier", "team_position", "champion_id", "patch", "game_duration", "half",
                 *metrics, *[f"{m}_resid" for m in metrics]]
    est[keep_rows].to_parquet(out / "type_rows.parquet", index=False)

    agg = {f"{m}_resid": "mean" for m in resid_cols}
    agg.update(win="mean", kda="mean", gold_pm="mean", damage_pm="mean", mnum="size")
    types = est.groupby("pid").agg(agg).rename(columns={"mnum": "n_type_matches", "win": "win_rate"})
    halves = est.pivot_table(index="pid", columns="half", values=[f"{m}_resid" for m in metrics], aggfunc="mean")
    halves.columns = [f"{m}_{h}" for m, h in halves.columns]
    types = types.join(halves).reset_index()
    regions = outcome_all.drop_duplicates("pid").set_index("pid")[["region", "tier"]]
    types = types.join(regions, on="pid", how="left")
    types.to_parquet(out / "focal_types.parquet", index=False)
    qa["types"] = {
        "players_with_types": int(len(types)),
        "players_ge5": int(types["n_type_matches"].ge(5).sum()),
        "players_ge10": int(types["n_type_matches"].ge(10).sum()),
        "winsorization": "pooled 1st/99th percentiles of per-match rates",
        "residualization": "within region on position, listing tier, patch and standardized duration",
    }
    name = "c1_data_build.json" if args.sample == "discovery" else "c1_holdout_data_build.json"
    write_json(TABLES / "qa" / name, qa)
    log(f"types: {qa['types']}")


if __name__ == "__main__":
    main()
