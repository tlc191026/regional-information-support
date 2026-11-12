"""Study 2: responses to the habitual information work of randomly assigned teammates.

Peer-type design. A focal player's habit is estimated from matches outside the outcome matches; the
outcome is the same behaviour of the other players in the focal player's outcome matches. Teammates
are compared with opponents of the same match, because symmetric matchmaking sorting affects both
teams whereas peer responses can only operate within a team.

Usage:
  python study2_analysis.py --sample discovery [--permute]
  python study2_analysis.py --sample holdout [--variant strict]

``--permute`` shuffles the outcome variables within region x focal-tier strata (dry run). All
outputs are written to tables/study2/<label>/.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

from config import (DATA, PING_FIELDS, PING_GROUPS, POSITIONS, REGIONS, SESOI, TABLES, TYPE_MIN_MATCHES,
                    TYPE_MIN_MATCHES_ROBUST, child_seed, ensure_dirs, log, write_json)
from twoway import ClusterCodes, residualize

EXPOSURES = {
    "control_wards": ("control_wards_pm", 10.0),
    "wards_killed": ("wards_killed_pm", 10.0),
    "wards_placed": ("wards_placed_pm", 10.0),
    "initiation": ("initiation_p10", 1.0),
    "caution": ("caution_p10", 1.0),
    "total_pings": ("total_pings_p10", 1.0),
}
OUTCOME_FIELDS = {
    "control_wards": ["control_wards_bought"],
    "wards_killed": ["wards_killed"],
    "wards_placed": ["wards_placed"],
    "initiation": PING_GROUPS["initiation"],
    "caution": PING_GROUPS["caution"],
    "total_pings": PING_FIELDS,
}
NAMES = list(EXPOSURES)
PRIMARY = ["control_wards", "wards_killed"]
TIMELINE_METRICS = {"control_wards": "control_ward_undo_net", "wards_killed": "valid_ward_kills",
                    "wards_placed": "valid_wards_placed"}
SKILL_CONTROLS = ["win_rate", "kda_resid", "gold_pm_resid", "damage_pm_resid"]
LISTING_CONTROLS = ["league_points", "division_num", "ladder_win_rate", "ladder_games"]
TIER_GROUPS = {"IRON": "low", "BRONZE": "low", "SILVER": "low", "GOLD": "mid", "PLATINUM": "mid",
               "EMERALD": "high", "DIAMOND": "high"}
ROW_COLS = ["focal_pid", "mnum", "focal_tier", "focal_team", "focal_position", "focal_champion", "team_id",
            "team_position", "pid", "relation", "n_same_team", "other_is_focal", "game_duration", "patch",
            "control_wards_bought", "wards_killed", "wards_placed", *PING_FIELDS]


def pid_hash(values: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(np.uint64)


def c1_folder(sample: str):
    return DATA / ("c1" if sample == "discovery" else "c1_holdout")


# ----------------------------------------------------------------------------- data

def load_types(sample: str, threshold: int) -> pd.DataFrame:
    types = pd.read_parquet(c1_folder(sample) / "focal_types.parquet")
    types = types[types["n_type_matches"].ge(threshold)].copy()
    listing = pd.read_parquet(DATA / "player_listing.parquet")
    listing = listing[listing["sample"].eq(sample)].copy()
    listing["pid"] = pid_hash(listing["puuid"])
    types = types.merge(listing[["pid", *LISTING_CONTROLS]], on="pid", how="left")
    for name, (col, scale) in EXPOSURES.items():
        types[f"T_{name}"] = types[f"{col}_resid"] * scale
        types[f"TA_{name}"] = types[f"{col}_resid_A"] * scale
        types[f"TB_{name}"] = types[f"{col}_resid_B"] * scale
    return types.set_index("pid")


def strict_excluded_matches() -> set:
    rows = pd.read_parquet(DATA / "holdout" / "primary" / "holdout_player_matches.parquet",
                           columns=["match_id", "contains_focal"])
    ids = rows.loc[rows["contains_focal"], "match_id"].drop_duplicates()
    return set(ids.str.split("_").str[-1].astype(np.int64))


def load_timelines(sample: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Participant timelines (window-valid matches only) split by region, with team gold at the cutoff."""
    folder = DATA / "timeline_participants" / ("fixed" if sample == "discovery" else "holdout")
    metric_cols = [f"{m}_m{k:02d}" for m in TIMELINE_METRICS.values() for k in range(1, 11)]
    parts = []
    for path in sorted(folder.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["match_id", "puuid", "team_id", "window_valid", "total_gold", *metric_cols])
        frame = frame[frame["window_valid"]]
        region = frame["match_id"].str.split("_").str[0]
        out = pd.DataFrame({
            "region": region.to_numpy(), "mnum": frame["match_id"].str.split("_").str[-1].astype(np.int64).to_numpy(),
            "pid": pid_hash(frame["puuid"]), "team_id": frame["team_id"].to_numpy(np.int16),
            "total_gold": frame["total_gold"].to_numpy(np.float32),
        })
        for col in metric_cols:
            out[col] = frame[col].to_numpy(np.int16)
        parts.append(out[out["region"].isin(REGIONS)])
    allframe = pd.concat(parts, ignore_index=True)
    result = {}
    for region, sub in allframe.groupby("region"):
        gold = sub.groupby(["mnum", "team_id"])["total_gold"].sum().rename("team_gold").reset_index()
        result[region] = (sub.drop(columns=["region", "total_gold"]).reset_index(drop=True), gold)
    return result


def load_rows(sample: str, region: str, types: pd.DataFrame, spec: dict, rng: np.random.Generator | None,
              timelines: tuple[pd.DataFrame, pd.DataFrame] | None) -> pd.DataFrame:
    rows = pd.read_parquet(c1_folder(sample) / f"outcome_rows_{region}.parquet", columns=ROW_COLS)
    rows = rows[rows["relation"].isin(["teammate", "opponent", "self"])]
    rows = rows[rows["focal_position"].isin(POSITIONS) & rows["team_position"].isin(POSITIONS)]
    rows = rows[rows["focal_pid"].isin(types.index)]
    if spec.get("strict"):
        rows = rows[~rows["mnum"].isin(spec["strict"])]
    duo = rows["relation"].eq("teammate") & rows["n_same_team"].ge(2)
    if spec.get("no_duo_units"):
        bad = rows.loc[duo, ["focal_pid", "mnum"]].drop_duplicates()
        key = pd.MultiIndex.from_frame(rows[["focal_pid", "mnum"]])
        rows = rows[~key.isin(pd.MultiIndex.from_frame(bad))]
    rows = rows[~(rows["relation"].ne("self") & rows["n_same_team"].ge(2))]
    if spec.get("exclude_focal_others"):
        rows = rows[~rows["other_is_focal"]]
    minutes = rows["game_duration"].to_numpy(float) / 60.0
    for name, fields in OUTCOME_FIELDS.items():
        rows[f"Y_{name}"] = rows[fields].sum(axis=1).to_numpy(float) / minutes * 10.0
    rows = rows.drop(columns=[c for c in ["control_wards_bought", "wards_killed", "wards_placed", *PING_FIELDS]])
    if timelines is not None:
        tl, gold = timelines
        rows = rows.merge(tl, on=["mnum", "pid"], how="left", suffixes=("", "_tl"))
        rows["window_valid"] = rows[f"{TIMELINE_METRICS['control_wards']}_m01"].notna()
        own = gold.rename(columns={"team_id": "focal_team", "team_gold": "own_gold"})
        enemy = gold.rename(columns={"team_gold": "enemy_gold"})
        enemy["focal_team"] = np.where(enemy["team_id"].eq(100), 200, 100)
        rows = rows.merge(own, on=["mnum", "focal_team"], how="left").merge(
            enemy[["mnum", "focal_team", "enemy_gold"]], on=["mnum", "focal_team"], how="left")
        rows["gold_diff_10"] = (rows["own_gold"] - rows["enemy_gold"]) / 1000.0
        for name, metric in TIMELINE_METRICS.items():
            cols = [f"{metric}_m{k:02d}" for k in range(1, 11)]
            rows[f"E_{name}"] = rows[cols].sum(axis=1, min_count=10)
    if rng is not None:
        rows = permute_outcomes(rows, rng)
    return rows.reset_index(drop=True)


def permute_outcomes(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Dry run: shuffle every outcome variable jointly within relation x focal-tier strata."""
    ycols = [c for c in rows.columns if c.startswith(("Y_", "E_")) or c[-4:-2] == "_m" or c.endswith("_m10")]
    ycols += ["window_valid"] if "window_valid" in rows.columns else []
    order = np.arange(len(rows))
    for _, idx in rows.groupby(["relation", "focal_tier"]).indices.items():
        order[idx] = idx[rng.permutation(len(idx))]
    out = rows.copy()
    for col in ycols:
        out[col] = rows[col].to_numpy()[order]
    return out


# ----------------------------------------------------------------------------- estimation

def controls(rows: pd.DataFrame, types: pd.DataFrame, extra_cats: tuple = (), extra_conts: tuple = (),
             self_rows: bool = False, exposure_cols: tuple = ()):
    tv = types.loc[rows["focal_pid"].to_numpy(), SKILL_CONTROLS + LISTING_CONTROLS + list(exposure_cols)]
    pos = rows["focal_position"].astype(str) if self_rows else rows["focal_position"].astype(str) + "|" + rows["team_position"].astype(str)
    cats = [rows["focal_tier"].to_numpy(), rows["patch"].astype(str).to_numpy(), pos.to_numpy()]
    cats += [rows[c].astype(str).to_numpy() for c in extra_cats]
    cont = tv[SKILL_CONTROLS + LISTING_CONTROLS].copy()
    for col in list(cont.columns):
        miss = cont[col].isna()
        if miss.any():
            cont[col + "_missing"] = miss.astype(float)
            cont[col] = cont[col].fillna(cont[col].median())
    for col in extra_conts:
        cont[col] = rows[col].to_numpy(float)
    arr = cont.to_numpy(float)
    arr = (arr - arr.mean(axis=0)) / np.where(arr.std(axis=0) > 0, arr.std(axis=0), 1.0)
    return cats, arr, tv


def slope(y, t):
    s = t @ t
    b = (t @ y) / s
    return b, t * (y - b * t) / s


def oriv_slope(y, ta, tb):
    c, sa, sb = ta @ tb, ta @ ta, tb @ tb
    p1, p2 = c / sb, c / sa
    d = p1 * c + p2 * c
    b = (p1 * (tb @ y) + p2 * (ta @ y)) / d
    return b, (p1 * tb * (y - b * ta) + p2 * ta * (y - b * tb)) / d


def fit_relation_pair(rows: pd.DataFrame, types: pd.DataFrame, pairs: list[tuple[str, str]],
                      extra_cats=(), extra_conts=(), relations=("teammate", "opponent")) -> list[dict]:
    """Teammate and opponent slopes (OLS and ORIV) with their joint two-way covariance."""
    ycols = sorted({o for _, o in pairs})
    exps = sorted({e for e, _ in pairs})
    blocks = {}
    for rel in relations:
        r = rows[rows["relation"].eq(rel)]
        if len(r) == 0:
            return []
        tcols = [f"{p}_{e}" for e in exps for p in ["T", "TA", "TB"]]
        cats, conts, tv = controls(r, types, extra_cats, extra_conts, self_rows=(rel == "self"), exposure_cols=tcols)
        values = np.column_stack([r[ycols].to_numpy(float), tv[tcols].to_numpy(float)])
        res = residualize(values, cats, conts)
        blocks[rel] = {"res": dict(zip(ycols + tcols, res.T)), "focal": r["focal_pid"].to_numpy(),
                       "match": r["mnum"].to_numpy(), "n": len(r)}
    codes = ClusterCodes(np.concatenate([blocks[r]["focal"] for r in relations]),
                         np.concatenate([blocks[r]["match"] for r in relations]))
    sizes = [blocks[r]["n"] for r in relations]
    out = []
    for e, o in pairs:
        for method in ["ols", "oriv"]:
            betas, psi = [], np.zeros((sum(sizes), len(relations)))
            start = 0
            for k, rel in enumerate(relations):
                res = blocks[rel]["res"]
                if method == "ols":
                    b, p = slope(res[o], res[f"T_{e}"])
                else:
                    b, p = oriv_slope(res[o], res[f"TA_{e}"], res[f"TB_{e}"])
                betas.append(b)
                psi[start:start + sizes[k], k] = p
                start += sizes[k]
            cov = codes.covariance(psi)
            rec = {"exposure": e, "outcome": o, "method": method}
            for k, rel in enumerate(relations):
                rec[f"b_{rel}"] = float(betas[k])
                rec[f"v_{rel}"] = float(cov[k, k])
                rec[f"n_{rel}"] = int(sizes[k])
            if len(relations) == 2:
                rec["c_teammate_opponent"] = float(cov[0, 1])
            rec["n_focal"] = codes.nf
            rec["n_match"] = codes.nm
            out.append(rec)
    return out


def fit_twoway_design(rows: pd.DataFrame, types: pd.DataFrame, outcomes: list[str]) -> list[dict]:
    """Rows whose other player is also a focal player observed in this outcome match (habit known, outcome
    match excluded from that player's habit estimate). OLS with the other player's own habit and the
    interaction of both habits."""
    self_keys = pd.MultiIndex.from_frame(rows.loc[rows["relation"].eq("self"), ["focal_pid", "mnum"]])
    out = []
    for rel in ["teammate", "opponent"]:
        r = rows[rows["relation"].eq(rel) & rows["pid"].isin(types.index)]
        keys = pd.MultiIndex.from_arrays([r["pid"].to_numpy(), r["mnum"].to_numpy()])
        r = r[keys.isin(self_keys)]
        if len(r) < 1000:
            continue
        cats, conts, tv = controls(r, types, exposure_cols=tuple(f"T_{o}" for o in outcomes))
        tj = types.loc[r["pid"].to_numpy(), [f"T_{o}" for o in outcomes]]
        for o in outcomes:
            ti_raw = tv[f"T_{o}"].to_numpy(float)
            tj_raw = tj[f"T_{o}"].to_numpy(float)
            inter = (ti_raw - ti_raw.mean()) * (tj_raw - tj_raw.mean())
            values = np.column_stack([r[f"Y_{o}"].to_numpy(float), ti_raw, tj_raw, inter])
            res = residualize(values, cats, conts)
            y, x = res[:, 0], res[:, 1:]
            codes = ClusterCodes(r["focal_pid"].to_numpy(), r["mnum"].to_numpy())
            for label, cols in [("own_only", [0]), ("with_other_habit", [0, 1]), ("with_interaction", [0, 1, 2])]:
                xx = x[:, cols]
                inv = np.linalg.inv(xx.T @ xx)
                b = inv @ (xx.T @ y)
                psi = (xx * (y - xx @ b)[:, None]) @ inv
                cov = codes.covariance(psi)
                for k, term in enumerate(["focal_habit", "other_habit", "interaction"][:len(cols)]):
                    out.append({"relation": rel, "outcome": o, "model": label, "term": term, "b": float(b[k]),
                                "v": float(cov[k, k]), "n_rows": int(len(r)), "n_focal": codes.nf})
    return out


# ----------------------------------------------------------------------------- summaries

def sd_table(all_rows: dict, types: pd.DataFrame) -> dict:
    """Pooled and within-region SDs used for standardization (teammate outcomes; player-level habits)."""
    out = {"pooled": {}, "region": {}}
    team = pd.concat([r.loc[r["relation"].eq("teammate"), ["focal_pid", *[f"Y_{n}" for n in NAMES]]].assign(region=g)
                      for g, r in all_rows.items()])
    used = types.loc[types.index.isin(team["focal_pid"].unique())]
    for scope, groups in [("pooled", [("pooled", team, used)]),
                          ("region", [(g, team[team["region"].eq(g)], used[used["region"].eq(g)]) for g in all_rows])]:
        for g, t, u in groups:
            entry = {}
            for n in NAMES:
                cov = np.cov(u[f"TA_{n}"], u[f"TB_{n}"])[0, 1]
                entry[n] = {"sd_y": float(t[f"Y_{n}"].std()), "sd_t": float(u[f"T_{n}"].std()),
                            "sd_t_reliable": float(np.sqrt(max(cov, 0.0))), "mean_y": float(t[f"Y_{n}"].mean())}
            out[scope][g] = entry
    return out


def accumulate(acc: dict, key: str, values: np.ndarray) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    s = acc.setdefault(key, np.zeros(3))
    s += [len(values), values.sum(), (values ** 2).sum()]


def acc_sd(acc: dict) -> dict:
    return {k: float(np.sqrt((v[2] - v[1] ** 2 / v[0]) / (v[0] - 1))) for k, v in acc.items()}


def combine(records: list[dict], group_cols: list[str]) -> pd.DataFrame:
    """Region estimates -> per-region rows for teammate, opponent and difference, plus equal-weight pooled."""
    frame = pd.DataFrame(records)
    out = []
    for key, g in frame.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key))
        per = {}
        for _, r in g.iterrows():
            est = {"teammate": (r["b_teammate"], r["v_teammate"]), "opponent": (r["b_opponent"], r["v_opponent"]),
                   "difference": (r["b_teammate"] - r["b_opponent"],
                                  r["v_teammate"] + r["v_opponent"] - 2 * r["c_teammate_opponent"])}
            per[r["region"]] = est
            for rel, (b, v) in est.items():
                out.append({**base, "region": r["region"], "relation": rel, "b": b, "se": np.sqrt(v),
                            "n_teammate": r["n_teammate"], "n_opponent": r["n_opponent"], "n_focal": r["n_focal"],
                            "n_match": r["n_match"]})
        regions = [r for r in REGIONS if r in per]
        if len(regions) == len(REGIONS):
            for rel in ["teammate", "opponent", "difference"]:
                b = np.mean([per[r][rel][0] for r in regions])
                v = np.sum([per[r][rel][1] for r in regions]) / len(regions) ** 2
                bs = np.array([per[r][rel][0] for r in regions])
                vs = np.diag([per[r][rel][1] for r in regions])
                contrast = np.array([[1, -1, 0], [0, 1, -1]])
                diff = contrast @ bs
                wald = float(diff @ np.linalg.pinv(contrast @ vs @ contrast.T) @ diff)
                out.append({**base, "region": "pooled", "relation": rel, "b": b, "se": np.sqrt(v),
                            "heterogeneity_wald": wald, "heterogeneity_p": float(stats.chi2.sf(wald, 2))})
                for a, c in [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]:
                    out.append({**base, "region": f"{a}-{c}", "relation": rel, "b": per[a][rel][0] - per[c][rel][0],
                                "se": np.sqrt(per[a][rel][1] + per[c][rel][1])})
    res = pd.DataFrame(out)
    z = res["b"] / res["se"]
    res["p"] = 2 * stats.norm.sf(np.abs(z))
    res["ci_low"], res["ci_high"] = res["b"] - 1.96 * res["se"], res["b"] + 1.96 * res["se"]
    return res


def add_standardized(res: pd.DataFrame, sds: dict, outcome_sd: dict | None = None) -> pd.DataFrame:
    """outcome_sd optionally overrides the outcome SD, keyed by '<spec>|<outcome>' or '<outcome>'."""
    res = res.copy()
    factor = []
    for _, r in res.iterrows():
        e = sds["pooled"]["pooled"][r["exposure"]]
        key = f"{r.get('spec', '')}|{r['outcome']}"
        if outcome_sd and key in outcome_sd:
            sd_y = outcome_sd[key]
        elif outcome_sd and r["outcome"] in outcome_sd:
            sd_y = outcome_sd[r["outcome"]]
        else:
            sd_y = sds["pooled"]["pooled"][r["outcome"]]["sd_y"]
        sd_t = e["sd_t"] if r["method"] == "ols" else e["sd_t_reliable"]
        factor.append(sd_t / sd_y)
    res["std_factor"] = factor
    res["b_std"] = res["b"] * res["std_factor"]
    res["se_std"] = res["se"] * res["std_factor"]
    res["ci_low_std"] = res["b_std"] - 1.96 * res["se_std"]
    res["ci_high_std"] = res["b_std"] + 1.96 * res["se_std"]
    res["ci90_low_std"] = res["b_std"] - 1.645 * res["se_std"]
    res["ci90_high_std"] = res["b_std"] + 1.645 * res["se_std"]
    res["tost_equivalent"] = (res["ci90_low_std"] > -SESOI["beta_sd"]) & (res["ci90_high_std"] < SESOI["beta_sd"])
    res["beyond_sesoi"] = (res["ci_low_std"] > SESOI["beta_sd"]) | (res["ci_high_std"] < -SESOI["beta_sd"])
    return res


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))).clip(0, 1)
    return out


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    out[order] = np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1].clip(0, 1)
    return out


def primary_family(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for o in PRIMARY:
        sub = main[(main["exposure"].eq(o)) & (main["outcome"].eq(o)) & main["method"].eq("oriv") & main["region"].eq("pooled")]
        d = sub[sub["relation"].eq("difference")].iloc[0]
        t = sub[sub["relation"].eq("teammate")].iloc[0]
        rows.append({"outcome": o, "test": "pooled teammate-minus-opponent (ORIV)", "b_std": d["b_std"],
                     "ci_low_std": d["ci_low_std"], "ci_high_std": d["ci_high_std"], "p": d["p"],
                     "tost_equivalent": d["tost_equivalent"], "beyond_sesoi": d["beyond_sesoi"]})
        rows.append({"outcome": o, "test": "regional heterogeneity of teammate-minus-opponent (Wald, 2 df)",
                     "b_std": np.nan, "ci_low_std": np.nan, "ci_high_std": np.nan, "p": d["heterogeneity_p"],
                     "wald": d["heterogeneity_wald"]})
        rows.append({"outcome": o, "test": "pooled teammate slope (ORIV)", "b_std": t["b_std"],
                     "ci_low_std": t["ci_low_std"], "ci_high_std": t["ci_high_std"], "p": t["p"],
                     "tost_equivalent": t["tost_equivalent"], "beyond_sesoi": t["beyond_sesoi"]})
    fam = pd.DataFrame(rows)
    fam["p_holm"] = holm(fam["p"].to_numpy())
    return fam


# ----------------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["discovery", "holdout"], default="discovery")
    ap.add_argument("--variant", choices=["primary", "strict"], default="primary")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    label = args.sample + ("" if args.variant == "primary" else "_strict") + ("_dryrun_permuted" if args.permute else "")
    out = TABLES / "study2" / label
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(child_seed(f"study2_permutation_{label}")) if args.permute else None
    start = time.monotonic()
    base_spec = {"strict": strict_excluded_matches()} if args.variant == "strict" else {}
    types = load_types(args.sample, TYPE_MIN_MATCHES)
    pairs_main = [(e, o) for e in NAMES for o in NAMES]
    pairs_primary = [(o, o) for o in PRIMARY]

    records, dyn, dyn_self, mods, robust, twoway_rows, all_rows = [], [], [], [], [], [], {}
    extra_acc = {}
    timelines = load_timelines(args.sample)
    log(f"{label}: timelines loaded ({time.monotonic() - start:.0f}s)")
    for region in REGIONS:
        rows = load_rows(args.sample, region, types, base_spec, rng, timelines[region])
        all_rows[region] = rows[rows["relation"].ne("self")][["relation", "focal_pid", *[f"Y_{n}" for n in NAMES]]]
        main_rows = rows[rows["relation"].ne("self")]
        team_mask = main_rows["relation"].eq("teammate")
        records += [{**r, "region": region, "spec": "main"} for r in fit_relation_pair(
            main_rows.rename(columns={f"Y_{n}": n for n in NAMES}), types, pairs_main)]
        log(f"{label} {region}: main done ({time.monotonic() - start:.0f}s)")

        valid = rows[rows["window_valid"].fillna(False).astype(bool)]
        for name, metric in TIMELINE_METRICS.items():
            cols = {f"{metric}_m{k:02d}": f"{name}_m{k:02d}" for k in range(1, 11)}
            sub = valid.rename(columns=cols)
            pairs = [(name, f"{name}_m{k:02d}") for k in range(1, 11)]
            for _, o in pairs:
                accumulate(extra_acc, o, sub.loc[sub["relation"].eq("teammate"), o])
            dyn += [{**r, "region": region} for r in fit_relation_pair(sub, types, pairs)]
            dyn_self += [{**r, "region": region} for r in fit_relation_pair(sub, types, pairs, relations=("self",))]
        log(f"{label} {region}: dynamics done ({time.monotonic() - start:.0f}s)")

        prim = main_rows.rename(columns={f"Y_{n}": n for n in NAMES})
        utility_jungle = prim["focal_position"].eq("UTILITY") & prim["team_position"].eq("JUNGLE")
        for mod, mask in [("support_to_jungle", utility_jungle), ("other_position_pairs", ~utility_jungle)]:
            mods += [{**r, "region": region, "moderator": "position", "level": mod}
                     for r in fit_relation_pair(prim[mask], types, pairs_primary)]
        groups = prim["focal_tier"].map(TIER_GROUPS)
        for level in ["low", "mid", "high"]:
            mods += [{**r, "region": region, "moderator": "tier_group", "level": level}
                     for r in fit_relation_pair(prim[groups.eq(level)], types, pairs_primary)]
        twoway_rows += [{**r, "region": region} for r in fit_twoway_design(rows, types, PRIMARY + ["wards_placed"])]
        log(f"{label} {region}: moderators done ({time.monotonic() - start:.0f}s)")

        if not args.skip_robustness:
            prim_valid = prim[prim.index.isin(valid.index)]
            for name in ["champion_fe", "log1p_outcome", "early_0_10", "state_gold_10", "exclude_lane_opponent"]:
                if name == "champion_fe":
                    recs = fit_relation_pair(prim, types, pairs_primary, extra_cats=("focal_champion",))
                elif name == "exclude_lane_opponent":
                    lane = prim["relation"].eq("opponent") & prim["team_position"].eq(prim["focal_position"])
                    recs = fit_relation_pair(prim[~lane], types, pairs_primary)
                elif name == "log1p_outcome":
                    tmp = prim.copy()
                    for o in PRIMARY:
                        tmp[o] = np.log1p(tmp[o])
                        accumulate(extra_acc, f"{name}|{o}", tmp.loc[team_mask, o])
                    recs = fit_relation_pair(tmp, types, pairs_primary)
                elif name == "early_0_10":
                    tmp = valid.copy()
                    for o in PRIMARY:
                        tmp[o] = tmp[f"E_{o}"]
                        accumulate(extra_acc, f"{name}|{o}", tmp.loc[tmp["relation"].eq("teammate"), o])
                    recs = fit_relation_pair(tmp, types, pairs_primary)
                else:
                    tmp = prim_valid[prim_valid["gold_diff_10"].notna()]
                    recs = fit_relation_pair(tmp, types, pairs_primary, extra_conts=("gold_diff_10",))
                robust += [{**r, "region": region, "spec": name} for r in recs]
            for name, spec in [("no_duo_units", {"no_duo_units": True}), ("exclude_focal_others", {"exclude_focal_others": True})]:
                tmp = load_rows(args.sample, region, types, {**base_spec, **spec}, rng, None)
                tmp = tmp[tmp["relation"].ne("self")].rename(columns={f"Y_{n}": n for n in NAMES})
                robust += [{**r, "region": region, "spec": name} for r in fit_relation_pair(tmp, types, pairs_primary)]
            types5 = load_types(args.sample, TYPE_MIN_MATCHES_ROBUST)
            tmp = load_rows(args.sample, region, types5, base_spec, rng, None)
            tmp = tmp[tmp["relation"].ne("self")].rename(columns={f"Y_{n}": n for n in NAMES})
            robust += [{**r, "region": region, "spec": "type_ge5"} for r in fit_relation_pair(tmp, types5, pairs_primary)]
            log(f"{label} {region}: robustness done ({time.monotonic() - start:.0f}s)")
        del rows, main_rows, valid, prim
        del timelines[region]

    sds = sd_table(all_rows, types)
    extra_sd = acc_sd(extra_acc)
    write_json(out / "standardization.json", {**sds, "extra_outcome_sd": extra_sd})
    main_res = add_standardized(combine(records, ["exposure", "outcome", "method", "spec"]), sds)
    main_res.to_csv(out / "main_effects.csv", index=False)
    # Within-region standardization (robustness r3).
    within = main_res[main_res["region"].isin(REGIONS)].copy()
    within["std_factor_within"] = [
        (sds["region"][r.region][r.exposure]["sd_t" if r.method == "ols" else "sd_t_reliable"]
         / sds["region"][r.region][r.outcome]["sd_y"]) for r in within.itertuples()]
    within["b_std_within"] = within["b"] * within["std_factor_within"]
    within["se_std_within"] = within["se"] * within["std_factor_within"]
    within.to_csv(out / "main_effects_within_region_sd.csv", index=False)
    fam = primary_family(main_res)
    fam.to_csv(out / "primary_family.csv", index=False)
    secondary = main_res[main_res["region"].eq("pooled") & main_res["relation"].isin(["difference", "teammate"])
                         & main_res["method"].eq("oriv")].copy()
    secondary = secondary[~(secondary["exposure"].eq(secondary["outcome"]) & secondary["exposure"].isin(PRIMARY))]
    secondary["q_bh"] = bh(secondary["p"].to_numpy())
    secondary.to_csv(out / "secondary_family.csv", index=False)

    if dyn:
        combine(dyn, ["exposure", "outcome", "method"]).pipe(add_standardized, sds, extra_sd).to_csv(
            out / "dynamics.csv", index=False)
    if dyn_self:
        s = pd.DataFrame(dyn_self)
        s["se"] = np.sqrt(s["v_self"])
        s = s.rename(columns={"b_self": "b"})
        s["spec"] = ""
        s = add_standardized(s, sds, extra_sd)
        s.to_csv(out / "dynamics_self.csv", index=False)
    if mods:
        combine(mods, ["moderator", "level", "exposure", "outcome", "method"]).pipe(add_standardized, sds).to_csv(
            out / "moderators.csv", index=False)
    if robust:
        combine(robust, ["spec", "exposure", "outcome", "method"]).pipe(add_standardized, sds, extra_sd).to_csv(
            out / "robustness.csv", index=False)
    if twoway_rows:
        tw = pd.DataFrame(twoway_rows)
        tw["se"] = np.sqrt(tw["v"])
        tw.to_csv(out / "twoway_design.csv", index=False)
    write_json(out / "run_info.json", {
        "label": label, "sample": args.sample, "variant": args.variant, "permuted": args.permute,
        "type_min_matches": TYPE_MIN_MATCHES, "seconds": time.monotonic() - start,
        "focal_players_with_types": int(len(types)),
    })
    log(f"{label}: finished in {time.monotonic() - start:.0f}s")


if __name__ == "__main__":
    main()
