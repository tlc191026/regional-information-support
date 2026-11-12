"""Performance link: early information maintenance and the next neutral objective (match level).

Blue-minus-red differences in the first 10 minutes predict which team secures the first neutral
objective after minute 10. Models extend the original specification (game-state controls; region,
listing tier and patch fixed effects) with pre-specified map-control proxies from the 10-minute
participant frames, objective-type splits, and natural-unit components.

  python performance_link.py --sample discovery            # exploratory (old RQ3 data)
  python performance_link.py --sample holdout [--variant strict]   # Study 3, R5
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.linear_model import LogisticRegression

from config import DATA, REGIONS, REPRO, TABLES, TIMELINE_FEATURES_10, ensure_dirs, log, write_json

STATE = ["gold_at_window_diff", "xp_at_window_diff", "level_at_window_diff", "kills_pre_diff",
         "neutral_objectives_pre_diff", "turret_plates_pre_diff"]
MAP = ["forward_diff", "enemy_half_diff", "jungle_cs_diff"]
COMPONENTS = ["valid_wards_placed", "valid_ward_kills", "control_ward_purchases"]
OUTCOME = "first_post_neutral_blue"
MAP_DIAGONAL = 14870.0 + 14980.0
SCALE = json.loads((REPRO / "parameters" / "maintenance_scale_primary.json").read_text(encoding="utf-8"))
SESOI_PP = 0.01


def maintenance(df: pd.DataFrame) -> np.ndarray:
    """Frozen discovery scale (log1p, winsorized, z-scored components; blue minus red; standardized)."""
    sides = []
    for side in ["blue", "red"]:
        parts = []
        for col in SCALE["components"]:
            low, high, mean, sd = SCALE["log1p_winsor_mean_sd"][col]
            z = (np.clip(np.log1p(np.maximum(df[f"{side}_{col}"].to_numpy(float), 0)), low, high) - mean) / (sd or 1.0)
            parts.append(z)
        sides.append((np.mean(parts, axis=0) - SCALE["score_mean"]) / SCALE["score_sd"])
    return (sides[0] - sides[1] - SCALE["diff_mean"]) / SCALE["diff_sd"]


def add_differences(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["gold_at_window", "xp_at_window", "level_at_window", "kills_pre", "neutral_objectives_pre",
                "turret_plates_pre"]:
        df[f"{col}_diff"] = df[f"blue_{col}"].to_numpy(float) - df[f"red_{col}"].to_numpy(float)
    for col in COMPONENTS:
        df[f"{col}_diff"] = df[f"blue_{col}"].to_numpy(float) - df[f"red_{col}"].to_numpy(float)
    df["maintenance_diff_z"] = maintenance(df)
    return df


def map_control(sample: str) -> pd.DataFrame:
    folder = DATA / "timeline_participants" / ("fixed" if sample == "discovery" else "holdout")
    out = []
    for path in sorted(folder.glob("*.parquet")):
        f = pd.read_parquet(path, columns=["match_id", "team_id", "window_valid", "x", "y", "jungle_minions_killed"])
        f = f[f["window_valid"]]
        progress = (f["x"] + f["y"]) / MAP_DIAGONAL
        blue = f["team_id"].eq(100)
        f = f.assign(forward=np.where(blue, progress, 1 - progress), enemy_half=(np.where(blue, progress, 1 - progress) > 0.5))
        g = f.groupby(["match_id", "team_id"]).agg(forward=("forward", "mean"), enemy_half=("enemy_half", "sum"),
                                                     jungle=("jungle_minions_killed", "sum")).unstack("team_id")
        out.append(pd.DataFrame({
            "match_id": g.index, "forward_diff": (g[("forward", 100)] - g[("forward", 200)]).to_numpy(),
            "enemy_half_diff": (g[("enemy_half", 100)] - g[("enemy_half", 200)]).to_numpy(),
            "jungle_cs_diff": (g[("jungle", 100)] - g[("jungle", 200)]).to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def holdout_features(variant: str) -> pd.DataFrame:
    team = pd.concat([pd.read_parquet(p, filters=[("window", "==", "0_10")])
                      for p in sorted((DATA / "timeline_team_holdout").glob("*.parquet"))], ignore_index=True)
    parts = pd.read_parquet(DATA / "participants" / "holdout", columns=["match_id", "team_id", "win", "patch", "game_duration"])
    meta = parts.groupby("match_id").agg(patch=("patch", "first"), game_duration=("game_duration", "first"))
    meta["blue_win_raw"] = parts[parts["team_id"].eq(100)].groupby("match_id")["win"].first().astype(float)
    hp = pd.read_parquet(DATA / "holdout" / "primary" / "holdout_player_matches.parquet",
                         columns=["puuid", "match_id", "tier", "contains_focal"])
    hp = hp.sort_values(["match_id", "puuid"]).drop_duplicates("match_id")
    df = team.merge(meta.reset_index(), on="match_id", how="inner").merge(hp[["match_id", "tier", "contains_focal"]],
                                                                         on="match_id", how="inner")
    if variant == "strict":
        df = df[~df["contains_focal"]]
    df["region"] = df["match_id"].str.split("_").str[0]
    df["winner_agreement"] = df["blue_win"].astype(float).eq(df["blue_win_raw"])
    df["analysis_eligible"] = df["window_valid"].astype(bool) & df["winner_agreement"]
    return df


def load(sample: str, variant: str) -> pd.DataFrame:
    if sample == "discovery":
        df = pd.read_parquet(TIMELINE_FEATURES_10)
    else:
        df = holdout_features(variant)
    df = df[df["analysis_eligible"].astype(bool) & df["region"].isin(REGIONS)].copy()
    df = add_differences(df)
    df = df.merge(map_control(sample), on="match_id", how="left")
    return df.dropna(subset=[OUTCOME, "maintenance_diff_z", *STATE]).reset_index(drop=True)


def design(df: pd.DataFrame, predictors: list[str], cats: list[str], conts: list[str]):
    blocks, names = [np.ones((len(df), 1))], ["intercept"]
    for c in cats:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
        blocks.append(d.to_numpy())
        names += list(d.columns)
    if conts:
        x = df[conts].to_numpy(float)
        blocks.append((x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0))
        names += conts
    blocks.append(df[predictors].to_numpy(float))
    names += predictors
    return np.hstack(blocks), names


def logit(df: pd.DataFrame, predictors: list[str], cats: list[str], conts: list[str], ame: dict | None = None) -> list[dict]:
    """Logistic regression with HC0 match-level inference. ame maps predictor -> step for discrete changes."""
    x, names = design(df, predictors, cats, conts)
    y = df[OUTCOME].to_numpy(float)
    model = LogisticRegression(C=1e8, solver="newton-cholesky", fit_intercept=False, max_iter=200, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    coef = model.coef_[0]
    eta = x @ coef
    p = special.expit(eta)
    resid = y - p
    hessian = (x * (p * (1 - p))[:, None]).T @ x
    bread = np.linalg.pinv(hessian)
    out = []
    for name in predictors:
        j = names.index(name)
        psi = (x @ bread[:, j]) * resid
        se = float(np.sqrt(psi @ psi))
        b = float(coef[j])
        rec = {"predictor": name, "beta": b, "se": se, "odds_ratio": float(np.exp(b)),
               "or_low": float(np.exp(b - 1.96 * se)), "or_high": float(np.exp(b + 1.96 * se)),
               "p": float(2 * stats.norm.sf(abs(b / se))), "n_matches": int(len(df)),
               "outcome_mean": float(y.mean())}
        if ame and name in ame:
            step = ame[name]
            p1 = special.expit(eta + coef[j] * step)
            w0, w1 = p * (1 - p), p1 * (1 - p1)
            grad = (x * (w1 - w0)[:, None]).mean(axis=0)
            grad[j] += step * w1.mean()
            psi_a = (x @ (bread @ grad)) * resid
            change = float((p1 - p).mean())
            se_a = float(np.sqrt(psi_a @ psi_a))
            rec.update(ame_step=step, prob_change=change, prob_change_se=se_a,
                       prob_change_low=change - 1.96 * se_a, prob_change_high=change + 1.96 * se_a)
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["discovery", "holdout"], default="discovery")
    ap.add_argument("--variant", choices=["primary", "strict"], default="primary")
    args = ap.parse_args()
    ensure_dirs()
    label = args.sample + ("" if args.variant == "primary" else "_strict")
    out = TABLES / "performance" / label
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.sample, args.variant)
    log(f"{label}: {len(df):,} matches; map proxies missing {int(df[MAP].isna().any(axis=1).sum()):,}")
    df = df.dropna(subset=MAP).reset_index(drop=True)
    cats = ["region", "tier", "patch"]
    rows = []
    specs = [
        ("M0_original", ["maintenance_diff_z"], STATE),
        ("M1_map_control", ["maintenance_diff_z"], STATE + MAP),
    ]
    for name, preds, conts in specs:
        for rec in logit(df, preds, cats, conts, ame={"maintenance_diff_z": 1.0}):
            rows.append({"model": name, "region": "pooled", "objective": "any", **rec})
        log(f"{label}: {name} pooled done")
    for region in REGIONS:
        sub = df[df["region"].eq(region)].reset_index(drop=True)
        for rec in logit(sub, ["maintenance_diff_z"], ["tier", "patch"], STATE + MAP, ame={"maintenance_diff_z": 1.0}):
            rows.append({"model": "M1_map_control", "region": region, "objective": "any", **rec})
    counts = df["first_monster_type"].value_counts()
    for objective in counts.index[counts.ge(10000)]:
        sub = df[df["first_monster_type"].eq(objective)].reset_index(drop=True)
        for rec in logit(sub, ["maintenance_diff_z"], cats, STATE + MAP, ame={"maintenance_diff_z": 1.0}):
            rows.append({"model": "M2_objective_type", "region": "pooled", "objective": objective, **rec})
    natural = [f"{c}_diff" for c in COMPONENTS]
    for rec in logit(df, natural, cats, STATE + MAP, ame={c: 1.0 for c in natural}):
        rows.append({"model": "M3_natural_units", "region": "pooled", "objective": "any", **rec})
    for region in REGIONS:
        sub = df[df["region"].eq(region)].reset_index(drop=True)
        for rec in logit(sub, natural, ["tier", "patch"], STATE + MAP, ame={c: 1.0 for c in natural}):
            rows.append({"model": "M3_natural_units", "region": region, "objective": "any", **rec})
    for rec in logit(df, MAP, cats, STATE + ["maintenance_diff_z"], ame={m: float(df[m].std()) for m in MAP}):
        rows.append({"model": "M1_map_proxies_themselves", "region": "pooled", "objective": "any", **rec})
    res = pd.DataFrame(rows)
    res["beyond_sesoi_pp"] = res.get("prob_change_low", pd.Series(np.nan, index=res.index)) > SESOI_PP
    res.to_csv(out / "performance_models.csv", index=False)
    write_json(out / "run_info.json", {
        "label": label, "matches": int(len(df)), "by_region": df["region"].value_counts().to_dict(),
        "objective_types": counts.to_dict(), "outcome_mean": float(df[OUTCOME].mean()),
        "map_proxies": {"forward_diff": "mean forward progress along the base-to-base diagonal at the 10-minute frame, blue minus red",
                        "enemy_half_diff": "players beyond the river diagonal, blue minus red",
                        "jungle_cs_diff": "jungle monsters killed by the 10-minute frame, blue minus red"},
        "maintenance_scale": "frozen discovery scale (maintenance_scale_primary.json)",
        "sesoi_probability_change": SESOI_PP,
    })
    log(f"{label}: performance models written")


if __name__ == "__main__":
    main()
