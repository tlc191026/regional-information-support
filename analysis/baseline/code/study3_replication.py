"""Study 3: apply the frozen replication criteria R1-R5 to the held-out sample.

Reads the discovery (Bronze-Diamond) and held-out outputs written by the frozen analysis scripts and
classifies each pattern as replicated, partially replicated, or not replicated. Criteria were fixed in
plan/analysis_plan_study2_study3.md before the held-out sample was analysed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import SESOI, TABLES, ensure_dirs, log, write_json

REL = SESOI["relative_difference"]
OCC = SESOI["occurrence_pp"]
BETA = SESOI["beta_sd"]
SESOI_PP = 0.01


def verdict_ratio(row, higher: bool) -> str:
    """Ratios: replicated if the CI lies beyond 1 +/- 10%; partial if only the direction holds."""
    if higher:
        if row["ratio_ci_low"] > 1 + REL:
            return "replicated"
        return "partially replicated" if row["ratio"] > 1 else "not replicated"
    if row["ratio_ci_high"] < 1 - REL:
        return "replicated"
    return "partially replicated" if row["ratio"] < 1 else "not replicated"


def verdict_interval(est, low, high, threshold, higher: bool) -> str:
    if higher:
        if low > threshold:
            return "replicated"
        return "partially replicated" if est > 0 else "not replicated"
    if high < -threshold:
        return "replicated"
    return "partially replicated" if est < 0 else "not replicated"


def study1_patterns(folder) -> list[dict]:
    out = []
    contrasts = pd.read_csv(folder / "regional_contrasts_natural.csv")
    means = pd.read_csv(folder / "regional_means_natural.csv")
    comp = pd.read_csv(folder / "ping_composition.csv")
    decomp = pd.read_csv(folder / "champion_decomposition.csv")
    grads = pd.read_csv(folder / "percentile_gradients.csv")
    for metric in ["control_wards_bought", "wards_killed"]:
        for contrast in ["KR-NA1", "KR-EUW1"]:
            row = contrasts[(contrasts["scale"].eq("per_10_min")) & contrasts["metric"].eq(metric)
                            & contrasts["contrast"].eq(contrast)].iloc[0]
            out.append({"pattern": "R1", "item": f"{metric} ratio {contrast}", "estimate": row["ratio"],
                        "ci_low": row["ratio_ci_low"], "ci_high": row["ratio_ci_high"],
                        "verdict": verdict_ratio(row, higher=True)})
            d = decomp[decomp["metric"].eq(metric) & decomp["contrast"].eq(contrast)].iloc[0]
            base = means[means["scale"].eq("per_10_min") & means["metric"].eq(metric)
                         & means["region"].eq(contrast.split("-")[1])]["estimate"].iloc[0]
            rel, low, high = d["within_champion"] / base, d["within_ci_low"] / base, d["within_ci_high"] / base
            out.append({"pattern": "R1", "item": f"{metric} within-champion gap {contrast} (relative)", "estimate": rel,
                        "ci_low": low, "ci_high": high, "verdict": verdict_interval(rel, low, high, REL, True)})
    for family, higher in [("caution", True), ("initiation", False)]:
        for contrast in ["KR-NA1", "KR-EUW1"]:
            row = contrasts[(contrasts["scale"].eq("occurrence")) & contrasts["metric"].eq(family)
                            & contrasts["contrast"].eq(contrast)].iloc[0]
            est, low, high = row["difference"], row["difference_ci_low"], row["difference_ci_high"]
            out.append({"pattern": "R2", "item": f"{family} occurrence difference {contrast}", "estimate": est,
                        "ci_low": low, "ci_high": high, "verdict": verdict_interval(est, low, high, OCC, higher)})
            share = comp[comp["kind"].eq("share_ratio") & comp["family"].eq(family) & comp["region"].eq(contrast)].iloc[0]
            srow = {"ratio": share["estimate"], "ratio_ci_low": share["ci_low"], "ratio_ci_high": share["ci_high"]}
            out.append({"pattern": "R2", "item": f"{family} share of all pings, ratio {contrast}", "estimate": share["estimate"],
                        "ci_low": share["ci_low"], "ci_high": share["ci_high"], "verdict": verdict_ratio(srow, higher)})
    for contrast, west in [("KR-NA1", "NA1"), ("KR-EUW1", "EUW1")]:
        row = grads[grads["percentile"].eq("gradient_difference") & grads["metric"].eq("initiation")
                    & grads["region"].eq(contrast)].iloc[0]
        west_grad = grads[grads["percentile"].eq("gradient_0.97_minus_0.10") & grads["metric"].eq("initiation")
                          & grads["region"].eq(west)]["estimate"].iloc[0]
        threshold = REL * abs(west_grad)
        out.append({"pattern": "R3", "item": f"initiation gradient difference {contrast}", "estimate": row["estimate"],
                    "ci_low": row["ci_low"], "ci_high": row["ci_high"], "threshold": threshold,
                    "verdict": verdict_interval(row["estimate"], row["ci_low"], row["ci_high"], threshold, False)})
    return out


def study2_patterns(disc: pd.DataFrame, hold: pd.DataFrame) -> list[dict]:
    out = []
    for outcome in ["control_wards", "wards_killed"]:
        for relation in ["difference", "teammate"]:
            def pick(frame, region="pooled"):
                return frame[(frame["exposure"].eq(outcome)) & frame["outcome"].eq(outcome) & frame["method"].eq("oriv")
                             & frame["relation"].eq(relation) & frame["region"].eq(region) & frame["spec"].eq("main")].iloc[0]
            d, h = pick(disc), pick(hold)
            if d["ci_low_std"] > BETA or d["ci_high_std"] < -BETA or d["p"] < 0.05:
                sign = np.sign(d["b_std"])
                if np.sign(h["b_std"]) != sign:
                    verdict = "not replicated"
                elif (h["ci_low_std"] > 0) if sign > 0 else (h["ci_high_std"] < 0):
                    verdict = "replicated"
                else:
                    verdict = "partially replicated"
                rule = "discovery non-zero: same sign and holdout 95% CI excluding zero"
            elif d["tost_equivalent"]:
                verdict = "replicated" if h["tost_equivalent"] else (
                    "not replicated" if (h["ci_low_std"] > 0 or h["ci_high_std"] < 0) else "partially replicated")
                rule = "discovery equivalent to zero: holdout 90% CI within +/-SESOI"
            else:
                verdict = "no replication test (discovery inconclusive)"
                rule = "discovery inconclusive"
            out.append({"pattern": "R4", "item": f"{outcome} {relation} pooled (ORIV, SD units)",
                        "discovery": d["b_std"], "estimate": h["b_std"], "ci_low": h["ci_low_std"],
                        "ci_high": h["ci_high_std"], "rule": rule, "verdict": verdict})
        d = disc[(disc["exposure"].eq(outcome)) & disc["outcome"].eq(outcome) & disc["method"].eq("oriv")
                 & disc["relation"].eq("difference") & disc["region"].eq("pooled") & disc["spec"].eq("main")].iloc[0]
        if d["heterogeneity_p"] < 0.05:
            def region_values(frame):
                sub = frame[(frame["exposure"].eq(outcome)) & frame["outcome"].eq(outcome) & frame["method"].eq("oriv")
                            & frame["relation"].eq("difference") & frame["region"].isin(["KR-NA1", "KR-EUW1"])
                            & frame["spec"].eq("main")]
                return sub.set_index("region")
            dv, hv = region_values(disc), region_values(hold)
            same = all(np.sign(dv.loc[c, "b"]) == np.sign(hv.loc[c, "b"]) for c in ["KR-NA1", "KR-EUW1"])
            sig = all((hv.loc[c, "ci_low"] > 0) or (hv.loc[c, "ci_high"] < 0) for c in ["KR-NA1", "KR-EUW1"])
            out.append({"pattern": "R4", "item": f"{outcome} regional contrasts of teammate-minus-opponent (KR vs NA, EUW)",
                        "verdict": "replicated" if same and sig else ("partially replicated" if same else "not replicated")})
    return out


def performance_patterns(disc: pd.DataFrame, hold: pd.DataFrame) -> list[dict]:
    out = []
    pick = lambda f, model, pred: f[f["model"].eq(model) & f["region"].eq("pooled") & f["objective"].eq("any")
                                    & f["predictor"].eq(pred)].iloc[0]
    h = pick(hold, "M1_map_control", "maintenance_diff_z")
    out.append({"pattern": "R5", "item": "maintenance difference, +1 SD probability change (M1)",
                "discovery": pick(disc, "M1_map_control", "maintenance_diff_z")["prob_change"],
                "estimate": h["prob_change"], "ci_low": h["prob_change_low"], "ci_high": h["prob_change_high"],
                "verdict": verdict_interval(h["prob_change"], h["prob_change_low"], h["prob_change_high"], SESOI_PP, True)})
    for pred in ["control_ward_purchases_diff", "valid_ward_kills_diff"]:
        d, h = pick(disc, "M3_natural_units", pred), pick(hold, "M3_natural_units", pred)
        sign = np.sign(d["prob_change"])
        excl = (h["prob_change_low"] > 0) if sign > 0 else (h["prob_change_high"] < 0)
        verdict = "replicated" if np.sign(h["prob_change"]) == sign and excl else (
            "partially replicated" if np.sign(h["prob_change"]) == sign else "not replicated")
        out.append({"pattern": "R5", "item": f"{pred}: +1 unit probability change (M3)", "discovery": d["prob_change"],
                    "estimate": h["prob_change"], "ci_low": h["prob_change_low"], "ci_high": h["prob_change_high"],
                    "verdict": verdict})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["primary", "strict"], default="primary")
    args = ap.parse_args()
    ensure_dirs()
    suffix = "" if args.variant == "primary" else "_strict"
    rows = study1_patterns(TABLES / "study3" / f"holdout_{args.variant}_six")
    disc2 = pd.read_csv(TABLES / "study2" / "discovery" / "main_effects.csv")
    hold2 = pd.read_csv(TABLES / "study2" / f"holdout{suffix}" / "main_effects.csv")
    rows += study2_patterns(disc2, hold2)
    perf_d = pd.read_csv(TABLES / "performance" / "discovery" / "performance_models.csv")
    perf_h = pd.read_csv(TABLES / "performance" / f"holdout{suffix}" / "performance_models.csv")
    rows += performance_patterns(perf_d, perf_h)
    res = pd.DataFrame(rows)
    out = TABLES / "study3"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"replication_verdicts_{args.variant}.csv", index=False)
    write_json(out / f"replication_summary_{args.variant}.json", res["verdict"].value_counts().to_dict())
    log(res[["pattern", "item", "verdict"]].to_string())


if __name__ == "__main__":
    main()
