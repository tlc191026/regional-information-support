"""Study 2 Step 0 diagnostics (run before the analysis freeze).

Reads habit estimates, duo counts, focal-pair structure, listing covariates and the focal player's own behaviour in
outcome matches. Teammate and opponent behaviour in outcome matches is never loaded here.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import DATA, POSITIONS, REGIONS, TABLES, TYPE_MIN_MATCHES, TYPE_MIN_MATCHES_ROBUST, child_seed, ensure_dirs, log, write_json

C1 = DATA / "c1"
OUT = TABLES / "study2" / "step0"
TYPES = ["control_wards_pm", "wards_killed_pm", "wards_placed_pm", "initiation_p10", "caution_p10", "total_pings_p10"]
SKILL = ["win_rate", "kda", "gold_pm", "damage_pm"]
LISTING = ["league_points", "division_num", "ladder_win_rate", "ladder_games"]
STRUCT_COLS = ["focal_pid", "mnum", "relation", "n_same_team", "n_opponent", "other_is_focal", "focal_position", "team_position"]
SELF_RATE = {
    "control_wards_pm": (["control_wards_bought"], 1.0), "wards_killed_pm": (["wards_killed"], 1.0),
    "wards_placed_pm": (["wards_placed"], 1.0), "initiation_p10": (["push_pings", "all_in_pings"], 10.0),
    "caution_p10": (["get_back_pings", "danger_pings"], 10.0),
}


def load_types() -> pd.DataFrame:
    types = pd.read_parquet(C1 / "focal_types.parquet")
    listing = pd.read_parquet(DATA / "player_listing.parquet")
    listing = listing[listing["sample"].eq("discovery")].copy()
    listing["pid"] = pd.util.hash_pandas_object(listing["puuid"].astype(str), index=False).to_numpy(np.uint64)
    return types.merge(listing[["pid", *LISTING]], on="pid", how="left")


def reliability(types: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [TYPE_MIN_MATCHES, TYPE_MIN_MATCHES_ROBUST]:
        sub = types[types["n_type_matches"].ge(threshold)]
        for region in ["pooled", *REGIONS]:
            part = sub if region == "pooled" else sub[sub["region"].eq(region)]
            for metric in TYPES:
                a, b = part[f"{metric}_resid_A"], part[f"{metric}_resid_B"]
                ok = a.notna() & b.notna()
                r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                cov = float(np.cov(a[ok], b[ok])[0, 1])
                rows.append({
                    "min_matches": threshold, "region": region, "metric": metric, "n_players": int(ok.sum()),
                    "split_half_r": r, "spearman_brown": 2 * r / (1 + r),
                    "sd_observed_full": float(part[f"{metric}_resid"].std()),
                    "sd_reliable": float(np.sqrt(max(cov, 0.0))),
                    "median_type_matches": float(part["n_type_matches"].median()),
                })
    return pd.DataFrame(rows)


def type_skill_correlations(types: pd.DataFrame) -> pd.DataFrame:
    sub = types[types["n_type_matches"].ge(TYPE_MIN_MATCHES)]
    rows = []
    for region in ["pooled", *REGIONS]:
        part = sub if region == "pooled" else sub[sub["region"].eq(region)]
        corr = part[[f"{m}_resid" for m in TYPES] + SKILL + LISTING].corr()
        for metric in TYPES:
            for other in [f"{m}_resid" for m in TYPES] + SKILL + LISTING:
                if other != f"{metric}_resid":
                    rows.append({"region": region, "metric": metric, "other": other,
                                 "r": float(corr.loc[f"{metric}_resid", other])})
    return pd.DataFrame(rows)


def structure(region: str) -> tuple[dict, pd.DataFrame]:
    rows = pd.read_parquet(C1 / f"outcome_rows_{region}.parquet", columns=STRUCT_COLS)
    focal_per_match = rows.loc[rows["relation"].eq("self")].groupby("mnum").size()
    rows["multi_focal_match"] = rows["mnum"].map(focal_per_match).gt(1)
    team = rows[rows["relation"].eq("teammate")]
    opp = rows[rows["relation"].eq("opponent")]
    duo_team = team["n_same_team"].ge(2)
    repeat_opp = opp["n_opponent"].ge(2)
    any_duo = team.assign(duo=duo_team).groupby(["focal_pid", "mnum"])["duo"].any()
    info = {
        "region": region, "focal_players": int(rows["focal_pid"].nunique()), "outcome_matches": int(rows["mnum"].nunique()),
        "focal_match_units": int(len(any_duo)), "teammate_rows": int(len(team)), "opponent_rows": int(len(opp)),
        "teammate_rows_duo_ge2": int(duo_team.sum()), "teammate_duo_share": float(duo_team.mean()),
        "opponent_repeat_ge2_share": float(repeat_opp.mean()),
        "teammate_repeat_ge2_share_minus_opponent": float(duo_team.mean() - repeat_opp.mean()),
        "focal_match_units_with_any_duo_share": float(any_duo.mean()),
        "teammate_rows_after_duo_exclusion": int((~duo_team).sum()),
        "teammate_rows_in_multi_focal_matches_share": float(team["multi_focal_match"].mean()),
        "teammate_rows_other_is_focal_share": float(team["other_is_focal"].mean()),
        "matches_with_ge2_focal_share": float(focal_per_match.gt(1).mean()),
        "teammate_rows_nonstandard_position_share": float((~team["team_position"].isin(POSITIONS)).mean()),
    }
    dist = team["n_same_team"].clip(upper=6).value_counts().sort_index()
    odist = opp["n_opponent"].clip(upper=6).value_counts().sort_index()
    repeat = pd.DataFrame({"count": dist.index, "teammate_rows": dist.to_numpy()}).merge(
        pd.DataFrame({"count": odist.index, "opponent_rows": odist.to_numpy()}), on="count", how="outer").fillna(0)
    repeat["region"] = region
    return info, repeat


def double_entry_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.concatenate([a, b])
    y = np.concatenate([b, a])
    return float(np.corrcoef(x, y)[0, 1])


def balance(types: pd.DataFrame, reps: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(child_seed("c1_step0_balance"))
    variables = [f"{m}_resid" for m in TYPES] + SKILL + LISTING
    tvals = types[types["n_type_matches"].ge(TYPE_MIN_MATCHES)].set_index("pid")[variables + ["tier"]]
    out = []
    for region in REGIONS:
        pairs = pd.read_parquet(C1 / f"focal_pairs_{region}.parquet")
        duo = pd.read_parquet(C1 / f"duo_counts_{region}.parquet", columns=["focal_pid", "pid", "n_same_team"])
        pairs = pairs.merge(duo.rename(columns={"focal_pid": "pid_a", "pid": "pid_b"}), on=["pid_a", "pid_b"], how="left")
        pairs["n_same_team"] = pairs["n_same_team"].fillna(0)
        pairs = pairs[pairs["n_same_team"].lt(2)]
        pairs = pairs.join(tvals.add_suffix("_a"), on="pid_a", how="inner").join(tvals.add_suffix("_b"), on="pid_b", how="inner")
        pairs["same_tier"] = pairs["tier_a"].eq(pairs["tier_b"])
        for scope, sub in [("all", pairs), ("same_listing_tier", pairs[pairs["same_tier"]])]:
            groups = {rel: sub[sub["relation"].eq(rel)] for rel in ["teammate", "opponent"]}
            for var in variables:
                vals = {}
                for rel, g in groups.items():
                    g = g[g[f"{var}_a"].notna() & g[f"{var}_b"].notna()]
                    vals[rel] = (g[f"{var}_a"].to_numpy(float), g[f"{var}_b"].to_numpy(float))
                est = {rel: double_entry_corr(*vals[rel]) for rel in vals}
                diffs = np.empty(reps)
                for k in range(reps):
                    boot = {}
                    for rel, (a, b) in vals.items():
                        idx = rng.integers(0, len(a), len(a))
                        boot[rel] = double_entry_corr(a[idx], b[idx])
                    diffs[k] = boot["teammate"] - boot["opponent"]
                out.append({
                    "region": region, "scope": scope, "variable": var,
                    "n_teammate_pairs": len(vals["teammate"][0]), "n_opponent_pairs": len(vals["opponent"][0]),
                    "r_teammate": est["teammate"], "r_opponent": est["opponent"],
                    "difference": est["teammate"] - est["opponent"],
                    "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
                })
        log(f"{region}: balance pairs {len(pairs):,}")
    return pd.DataFrame(out)


def balance_by_position(types: pd.DataFrame, reps: int = 1000) -> pd.DataFrame:
    """Teammate versus opponent similarity within ordered match-position pairs.

    Teammates never share a match position, whereas one opponent in five does. Comparing pairs with the same ordered
    position combination (lane opponents excluded) mirrors the position-pair fixed effects of the outcome model.
    """
    rng = np.random.default_rng(child_seed("c1_step0_balance_position"))
    variables = [f"{m}_resid" for m in TYPES] + SKILL + LISTING
    tvals = types[types["n_type_matches"].ge(TYPE_MIN_MATCHES)].set_index("pid")[variables + ["tier"]]
    out = []
    for region in REGIONS:
        pos = pd.read_parquet(C1 / f"outcome_rows_{region}.parquet", columns=["focal_pid", "mnum", "team_position", "relation"],
                              filters=[("relation", "==", "self")]).drop(columns="relation")
        pos = pos.set_index(["focal_pid", "mnum"])["team_position"]
        pairs = pd.read_parquet(C1 / f"focal_pairs_{region}.parquet")
        duo = pd.read_parquet(C1 / f"duo_counts_{region}.parquet", columns=["focal_pid", "pid", "n_same_team"])
        pairs = pairs.merge(duo.rename(columns={"focal_pid": "pid_a", "pid": "pid_b"}), on=["pid_a", "pid_b"], how="left")
        pairs = pairs[pairs["n_same_team"].fillna(0).lt(2)]
        pairs["pos_a"] = pos.reindex(pd.MultiIndex.from_arrays([pairs["pid_a"], pairs["mnum"]])).to_numpy()
        pairs["pos_b"] = pos.reindex(pd.MultiIndex.from_arrays([pairs["pid_b"], pairs["mnum"]])).to_numpy()
        pairs = pairs[pairs["pos_a"].isin(POSITIONS) & pairs["pos_b"].isin(POSITIONS) & pairs["pos_a"].ne(pairs["pos_b"])]
        pairs = pairs.join(tvals.add_suffix("_a"), on="pid_a", how="inner").join(tvals.add_suffix("_b"), on="pid_b", how="inner")
        pairs = pairs.reset_index(drop=True)
        for scope, sub in [("position_matched", pairs), ("position_matched_same_tier", pairs[pairs["tier_a"].eq(pairs["tier_b"])])]:
            for var in variables:
                ok = sub[f"{var}_a"].notna() & sub[f"{var}_b"].notna()
                s = sub[ok]
                stacked = pd.DataFrame({
                    "pair": np.concatenate([s.index.to_numpy(), s.index.to_numpy()]),
                    "relation": np.concatenate([s["relation"], s["relation"]]),
                    "px": np.concatenate([s["pos_a"], s["pos_b"]]), "py": np.concatenate([s["pos_b"], s["pos_a"]]),
                    "x": np.concatenate([s[f"{var}_a"], s[f"{var}_b"]]), "y": np.concatenate([s[f"{var}_b"], s[f"{var}_a"]]),
                })
                cell = stacked.groupby(["relation", "px", "py"])
                stacked["x"] = stacked["x"] - cell["x"].transform("mean")
                stacked["y"] = stacked["y"] - cell["y"].transform("mean")
                est, boot = {}, {}
                for rel in ["teammate", "opponent"]:
                    g = stacked[stacked["relation"].eq(rel)]
                    est[rel] = float(np.corrcoef(g["x"], g["y"])[0, 1])
                    pair_ids = g["pair"].unique()
                    codes = pd.factorize(g["pair"])[0]
                    starts = np.r_[0, np.cumsum(np.bincount(codes))[:-1]]
                    order = np.argsort(codes, kind="stable")
                    xs, ys = g["x"].to_numpy()[order], g["y"].to_numpy()[order]
                    vals = np.empty(reps)
                    for k in range(reps):
                        pick = rng.integers(0, len(pair_ids), len(pair_ids))
                        idx = np.concatenate([starts[pick], starts[pick] + 1])
                        vals[k] = np.corrcoef(xs[idx], ys[idx])[0, 1]
                    boot[rel] = vals
                diff = boot["teammate"] - boot["opponent"]
                out.append({
                    "region": region, "scope": scope, "variable": var,
                    "n_teammate_pairs": int(s["relation"].eq("teammate").sum()),
                    "n_opponent_pairs": int(s["relation"].eq("opponent").sum()),
                    "r_teammate": est["teammate"], "r_opponent": est["opponent"],
                    "difference": est["teammate"] - est["opponent"],
                    "ci_low": float(np.quantile(diff, 0.025)), "ci_high": float(np.quantile(diff, 0.975)),
                })
        log(f"{region}: position-matched balance pairs {len(pairs):,}")
    return pd.DataFrame(out)


def own_behaviour(types: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample validity: habits predict the focal player's own behaviour in the outcome matches."""
    cols = ["focal_pid", "mnum", "relation", "focal_tier", "team_position", "game_duration", "region",
            "control_wards_bought", "wards_killed", "wards_placed", "push_pings", "all_in_pings", "get_back_pings",
            "danger_pings"]
    rows = []
    sub_types = types[types["n_type_matches"].ge(TYPE_MIN_MATCHES)].set_index("pid")
    for region in REGIONS:
        own = pd.read_parquet(C1 / f"outcome_rows_{region}.parquet", columns=cols, filters=[("relation", "==", "self")])
        minutes = own["game_duration"] / 60.0
        for metric, (fields, scale) in SELF_RATE.items():
            own[metric] = own[fields].sum(axis=1) / minutes * scale
            cell = own.groupby(["focal_tier", "team_position"])[metric].transform("mean")
            own[f"{metric}_dev"] = own[metric] - cell
        means = own.groupby("focal_pid")[[f"{m}_dev" for m in SELF_RATE]].mean()
        joined = means.join(sub_types[[f"{m}_resid" for m in SELF_RATE] + [f"{m}_resid_A" for m in SELF_RATE]], how="inner")
        for metric in SELF_RATE:
            x = joined[f"{metric}_resid"]
            y = joined[f"{metric}_dev"]
            slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
            rows.append({"region": region, "metric": metric, "n_players": int(len(joined)),
                         "r_type_vs_own_outcome_mean": float(np.corrcoef(x, y)[0, 1]),
                         "slope_own_outcome_on_type": slope,
                         "r_halfA_vs_own_outcome_mean": float(np.corrcoef(joined[f"{metric}_resid_A"], y)[0, 1])})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    types = load_types()
    if "--only-position" in sys.argv:
        balance_by_position(types).to_csv(OUT / "balance_focal_pairs_position_matched.csv", index=False)
        return
    rel = reliability(types)
    rel.to_csv(OUT / "type_reliability.csv", index=False)
    type_skill_correlations(types).to_csv(OUT / "type_correlations.csv", index=False)
    log("reliability done")

    infos, repeats = [], []
    for region in REGIONS:
        info, repeat = structure(region)
        infos.append(info)
        repeats.append(repeat)
        log(f"{region}: structure {info}")
    pd.DataFrame(infos).to_csv(OUT / "sample_structure.csv", index=False)
    pd.concat(repeats).to_csv(OUT / "repeat_encounters.csv", index=False)

    counts = types.groupby("region").agg(
        players_with_types=("pid", "size"),
        players_ge10=("n_type_matches", lambda s: int(s.ge(TYPE_MIN_MATCHES).sum())),
        players_ge5=("n_type_matches", lambda s: int(s.ge(TYPE_MIN_MATCHES_ROBUST).sum())),
        median_type_matches=("n_type_matches", "median"),
    ).reset_index()
    by_tier = types.assign(ge10=types["n_type_matches"].ge(TYPE_MIN_MATCHES)).groupby(["region", "tier"])["ge10"].agg(
        ["sum", "mean"]).reset_index().rename(columns={"sum": "players_ge10", "mean": "share_ge10"})
    counts.to_csv(OUT / "type_counts.csv", index=False)
    by_tier.to_csv(OUT / "type_counts_by_tier.csv", index=False)

    bal = balance(types)
    bal.to_csv(OUT / "balance_focal_pairs.csv", index=False)
    balance_by_position(types).to_csv(OUT / "balance_focal_pairs_position_matched.csv", index=False)
    log("balance done")
    own = own_behaviour(types)
    own.to_csv(OUT / "own_behaviour_validity.csv", index=False)
    log("own-behaviour validity done")

    write_json(TABLES / "qa" / "study2_step0.json", {
        "note": "No teammate or opponent behaviour in outcome matches was read.",
        "type_counts": counts.to_dict(orient="records"),
        "structure": infos,
        "max_abs_balance_difference_all_scope": float(bal.loc[bal["scope"].eq("all"), "difference"].abs().max()),
    })


if __name__ == "__main__":
    main()
