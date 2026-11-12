"""Study 1 re-analysis in natural units.

Estimands use equal weights across listing tiers and match positions (direct standardization);
intervals use stratified player-cluster bootstrap weights with exact recomputation.

  python study1_analysis.py                                   # discovery, all seven tiers (exploratory)
  python study1_analysis.py --sample discovery --tiers six    # discovery, Bronze-Diamond (replication target)
  python study1_analysis.py --sample holdout --tiers six      # Study 3 replication (R1-R3)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA, PING_FIELDS, PING_GROUPS, POSITIONS, REGIONS, TABLES, TIERS, child_seed, ensure_dirs, log, write_json
from estimation import (CONTRASTS, cell_sums, contrast_table, load_weights, mean_table, player_order,
                        standardized_means, stratified_weights)

OUT = TABLES / "study1"
TIER_SET = list(TIERS)
VISION = ["wards_placed", "wards_killed", "control_wards_bought"]
PINGS = list(PING_FIELDS)
FAMILIES = list(PING_GROUPS)
KEY = ["control_wards_bought", "wards_killed", "wards_placed", "get_back_pings", "push_pings", "all_in_pings",
       "caution", "initiation"]
COVERAGE = TABLES / "qa" / "ping_field_coverage.json"


def load_rows(path, freeze_fields: bool) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    rows = rows[rows["team_position"].isin(POSITIONS) & rows["tier"].isin(TIER_SET)].reset_index(drop=True)
    if freeze_fields:
        nonzero = {c: int((rows[c] > 0).sum()) for c in PING_FIELDS}
        # Fields that are never recorded carry no information and break ratios and log-ratios.
        PINGS[:] = [c for c in PING_FIELDS if nonzero[c] > 0]
        FAMILIES[:] = [f for f, members in PING_GROUPS.items() if any(nonzero[m] > 0 for m in members)]
        write_json(COVERAGE, {
            "nonzero_player_match_records": nonzero, "records": int(len(rows)),
            "analysed_fields": PINGS, "analysed_families": FAMILIES,
        })
    else:
        frozen = json.loads(COVERAGE.read_text(encoding="utf-8"))
        PINGS[:] = frozen["analysed_fields"]
        FAMILIES[:] = frozen["analysed_families"]
    for name, fields in PING_GROUPS.items():
        rows[name] = rows[fields].sum(axis=1)
    rows["total_pings"] = rows[PING_FIELDS].sum(axis=1)
    return rows


def rate_columns(rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return rows[columns].to_numpy(float) / rows["duration_min"].to_numpy(float)[:, None] * 10.0


def regional_rates(rows, codes, players, weights) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = VISION + PINGS + FAMILIES + ["total_pings"]
    pos = pd.Categorical(rows["team_position"], categories=POSITIONS).codes
    sums = cell_sums(codes, len(players), pos, len(POSITIONS), rate_columns(rows, metrics))
    counts = cell_sums(codes, len(players), pos, len(POSITIONS), np.ones((len(rows), 1)))[:, :, 0]
    point, boot = standardized_means(sums, counts, players, weights, TIER_SET)
    occ_cols = VISION + PINGS + FAMILIES
    occ = cell_sums(codes, len(players), pos, len(POSITIONS), (rows[occ_cols].to_numpy() > 0).astype(float))
    opoint, oboot = standardized_means(occ, counts, players, weights, TIER_SET)
    means = pd.concat([mean_table(point, boot, metrics).assign(scale="per_10_min"),
                       mean_table(opoint, oboot, occ_cols).assign(scale="occurrence")], ignore_index=True)
    contrasts = pd.concat([contrast_table(point, boot, metrics).assign(scale="per_10_min"),
                           contrast_table(opoint, oboot, occ_cols).assign(scale="occurrence")], ignore_index=True)
    ranks = []
    for k, metric in enumerate(metrics):
        order = [REGIONS[i] for i in np.argsort(-point[:, k])]
        ranks.append({"metric": metric, "order_high_to_low": ">".join(order)})
    pd.DataFrame(ranks).to_csv(OUT / "regional_rank_orders.csv", index=False)
    return means, contrasts


def composition(rows, codes, players, weights) -> pd.DataFrame:
    """Family shares of all pings: pooled ratio within region x tier, equal tier weights; clr and Aitchison."""
    values = rows[FAMILIES].to_numpy(float)
    sums = cell_sums(codes, len(players), np.zeros(len(rows), dtype=np.int64), 1, values)[:, 0, :]
    reps = weights.shape[0]
    share = np.zeros((len(REGIONS), len(FAMILIES)))
    share_b = np.zeros((reps, len(REGIONS), len(FAMILIES)))
    for r, region in enumerate(REGIONS):
        for tier in TIER_SET:
            members = np.flatnonzero((players["region"] == region).to_numpy() & (players["tier"] == tier).to_numpy())
            s = sums[members].sum(axis=0)
            share[r] += s / s.sum() / len(TIER_SET)
            sb = weights[:, members] @ sums[members]
            share_b[:, r] += sb / sb.sum(axis=1, keepdims=True) / len(TIER_SET)

    def clr(x):
        logx = np.log(x)
        return logx - logx.mean(axis=-1, keepdims=True)

    rows_out = []
    for r, region in enumerate(REGIONS):
        for k, fam in enumerate(FAMILIES):
            rows_out.append({"kind": "share", "region": region, "family": fam, "estimate": share[r, k],
                             "ci_low": np.percentile(share_b[:, r, k], 2.5),
                             "ci_high": np.percentile(share_b[:, r, k], 97.5)})
    c, cb = clr(share), clr(share_b)
    for a, b in CONTRASTS:
        i, j = REGIONS.index(a), REGIONS.index(b)
        dist = float(np.linalg.norm(c[i] - c[j]))
        db = np.linalg.norm(cb[:, i] - cb[:, j], axis=1)
        rows_out.append({"kind": "aitchison_distance", "region": f"{a}-{b}", "family": "all", "estimate": dist,
                         "ci_low": np.percentile(db, 2.5), "ci_high": np.percentile(db, 97.5)})
        for k, fam in enumerate(FAMILIES):
            ratio_b = share_b[:, i, k] / share_b[:, j, k]
            rows_out.append({"kind": "share_ratio", "region": f"{a}-{b}", "family": fam,
                             "estimate": share[i, k] / share[j, k],
                             "ci_low": np.percentile(ratio_b, 2.5), "ci_high": np.percentile(ratio_b, 97.5)})
    return pd.DataFrame(rows_out)


def cliffs_delta(rows, codes, players) -> pd.DataFrame:
    metrics = ["control_wards_bought", "wards_killed", "wards_placed", "get_back_pings", "push_pings",
               "all_in_pings", "caution", "initiation", "total_pings"]
    rates = pd.DataFrame(rate_columns(rows, metrics), columns=metrics)
    player_means = rates.groupby(codes).mean()
    player_means["region"] = players["region"].to_numpy()[player_means.index]
    out = []
    for a, b in CONTRASTS:
        x, y = player_means[player_means["region"].eq(a)], player_means[player_means["region"].eq(b)]
        for metric in metrics:
            u = stats.mannwhitneyu(x[metric], y[metric], alternative="two-sided").statistic
            out.append({"contrast": f"{a}-{b}", "metric": metric,
                        "cliffs_delta": 2 * u / (len(x) * len(y)) - 1, "n_a": len(x), "n_b": len(y)})
    return pd.DataFrame(out)


def champion_decomposition(rows, codes, players, weights) -> pd.DataFrame:
    """Split regional gaps into within-champion and composition parts (direct standardization).

    For contrast a-b: within = mean_a - sum_c share_a(c) * mean_b(c); composition = remainder.
    Restricted to champion x position cells with at least 30 player-matches in every region.
    """
    import scipy.sparse as sp

    names = ["control_wards_bought", "wards_killed", "wards_placed", "get_back_pings", "push_pings", "all_in_pings"]
    rates = rate_columns(rows, names)
    cell, labels = pd.factorize(rows["champion_id"].astype(str) + "|" + rows["team_position"])
    region_code = pd.Categorical(rows["region"], categories=REGIONS).codes
    counts = np.zeros((len(REGIONS), len(labels)))
    np.add.at(counts, (region_code, cell), 1)
    support = np.flatnonzero((counts >= 30).all(axis=0))
    remap = np.full(len(labels), -1)
    remap[support] = np.arange(len(support))
    keep = remap[cell] >= 0
    ck, pk = remap[cell[keep]], codes[keep]
    n, m = len(players), len(support)
    reg = players["region"].to_numpy()
    c_mat = sp.csr_matrix((np.ones(keep.sum()), (pk, ck)), shape=(n, m))
    count_pt = np.vstack([np.asarray(c_mat[reg == r].sum(axis=0)).ravel() for r in REGIONS])
    count_bt = np.stack([np.asarray(weights[:, reg == r] @ c_mat[reg == r].toarray()) for r in REGIONS], axis=1)

    def parts(s, c, i, j):
        mean_i = s[..., i, :].sum(-1) / c[..., i, :].sum(-1)
        mean_j = s[..., j, :].sum(-1) / c[..., j, :].sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cell_j = np.where(c[..., j, :] > 0, s[..., j, :] / c[..., j, :], 0.0)
        share_i = c[..., i, :] / c[..., i, :].sum(-1, keepdims=True)
        reweighted = (share_i * cell_j).sum(-1)
        return mean_i - mean_j, mean_i - reweighted, reweighted - mean_j

    out = []
    for k, metric in enumerate(names):
        s_mat = sp.csr_matrix((rates[keep, k], (pk, ck)), shape=(n, m))
        sum_pt = np.vstack([np.asarray(s_mat[reg == r].sum(axis=0)).ravel() for r in REGIONS])
        sum_bt = np.stack([np.asarray(weights[:, reg == r] @ s_mat[reg == r].toarray()) for r in REGIONS], axis=1)
        for a, b in CONTRASTS:
            i, j = REGIONS.index(a), REGIONS.index(b)
            total, within, comp = parts(sum_pt, count_pt, i, j)
            tb, wb, cb = parts(sum_bt, count_bt, i, j)
            out.append({"metric": metric, "contrast": f"{a}-{b}", "total_gap": total, "within_champion": within,
                        "composition": comp, "within_share": within / total if total else np.nan,
                        "total_ci_low": np.percentile(tb, 2.5), "total_ci_high": np.percentile(tb, 97.5),
                        "within_ci_low": np.percentile(wb, 2.5), "within_ci_high": np.percentile(wb, 97.5),
                        "composition_ci_low": np.percentile(cb, 2.5), "composition_ci_high": np.percentile(cb, 97.5),
                        "support_cells": int(m), "support_share_of_rows": float(keep.mean())})
    return pd.DataFrame(out)


def same_champion_examples(rows) -> pd.DataFrame:
    metrics = ["control_wards_bought", "wards_killed", "wards_placed", "get_back_pings", "push_pings", "all_in_pings"]
    rates = pd.DataFrame(rate_columns(rows, metrics), columns=metrics)
    frame = pd.concat([rows[["region", "team_position", "champion_name"]].reset_index(drop=True), rates], axis=1)
    counts = frame.groupby(["team_position", "champion_name", "region"]).size().unstack("region").fillna(0)
    eligible = counts[(counts >= 5000).all(axis=1)].sum(axis=1).sort_values(ascending=False)
    top = eligible.groupby(level="team_position").head(3).index
    sub = frame.set_index(["team_position", "champion_name"]).loc[top].reset_index()
    out = sub.groupby(["team_position", "champion_name", "region"])[metrics].mean().reset_index()
    out["n_player_matches"] = sub.groupby(["team_position", "champion_name", "region"]).size().to_numpy()
    return out


def percentile_gradients(rows, codes, players, weights) -> pd.DataFrame:
    perc = pd.read_parquet(DATA / "player_percentiles.parquet", columns=["puuid", "percentile"])
    p = players.merge(perc, on="puuid", how="left")["percentile"].to_numpy()
    metrics = ["control_wards_bought", "wards_killed", "wards_placed", "get_back_pings", "push_pings", "all_in_pings",
               "caution", "initiation", "total_pings"]
    rates = pd.DataFrame(rate_columns(rows, metrics), columns=metrics).groupby(codes).mean().to_numpy()
    grid = np.array([0.10, 0.25, 0.50, 0.75, 0.90, 0.97])
    knots = np.quantile(p[~np.isnan(p)], [0.25, 0.5, 0.75])

    def basis(x):
        cols = [np.ones_like(x), x, x ** 2, x ** 3] + [np.clip(x - k, 0, None) ** 3 for k in knots]
        return np.column_stack(cols)

    out = []
    grads = {}
    for region in REGIONS:
        use = (players["region"] == region).to_numpy() & ~np.isnan(p)
        b = basis(p[use])
        bg = basis(grid)
        outer = np.einsum("ni,nj->nij", b, b).reshape(len(b), -1)
        w = weights[:, use]
        xtx = (w @ outer).reshape(-1, b.shape[1], b.shape[1])
        point_xtx = b.T @ b
        for k, metric in enumerate(metrics):
            y = rates[use, k]
            beta = np.linalg.solve(point_xtx, b.T @ y)
            xty = w @ (b * y[:, None])
            boot = np.linalg.solve(xtx, xty[..., None])[..., 0]
            fit, fit_b = bg @ beta, boot @ bg.T
            grad, grad_b = fit[-1] - fit[0], fit_b[:, -1] - fit_b[:, 0]
            grads[(region, metric)] = (grad, grad_b)
            for g, value, col in zip(grid, fit, fit_b.T):
                out.append({"region": region, "metric": metric, "percentile": g, "estimate": value,
                            "ci_low": np.percentile(col, 2.5), "ci_high": np.percentile(col, 97.5)})
            out.append({"region": region, "metric": metric, "percentile": "gradient_0.97_minus_0.10", "estimate": grad,
                        "ci_low": np.percentile(grad_b, 2.5), "ci_high": np.percentile(grad_b, 97.5)})
    for a, c in CONTRASTS:
        for metric in metrics:
            ga, gab = grads[(a, metric)]
            gc, gcb = grads[(c, metric)]
            diff = gab - gcb
            out.append({"region": f"{a}-{c}", "metric": metric, "percentile": "gradient_difference",
                        "estimate": ga - gc, "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5)})
    return pd.DataFrame(out)


def bootstrap_weights(rows: pd.DataFrame, sample: str, label: str):
    """Discovery: frozen weights, subset to the retained tiers. Holdout: fresh stratified weights."""
    if sample == "discovery":
        full_rows = pd.read_parquet(DATA / "study1_focal_rows.parquet", columns=["puuid", "region", "tier"])
        _, all_players = player_order(full_rows)
        weights = load_weights(all_players)
        codes, players = player_order(rows)
        position = pd.Index(all_players["puuid"]).get_indexer(players["puuid"])
        assert (position >= 0).all()
        return codes, players, weights[:, position]
    codes, players = player_order(rows)
    return codes, players, stratified_weights(players, child_seed(f"study1_bootstrap_{label}"))


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["discovery", "holdout"], default="discovery")
    ap.add_argument("--variant", choices=["primary", "strict"], default="primary")
    ap.add_argument("--tiers", choices=["all", "six"], default="all")
    args = ap.parse_args()
    ensure_dirs()
    TIER_SET[:] = TIERS if args.tiers == "all" else [t for t in TIERS if t != "IRON"]
    exploratory = args.sample == "discovery" and args.tiers == "all"
    label = f"{args.sample}_{args.variant}_{args.tiers}"
    OUT = TABLES / "study1" if exploratory else TABLES / "study3" / label
    OUT.mkdir(parents=True, exist_ok=True)
    path = DATA / ("study1_focal_rows.parquet" if args.sample == "discovery" else f"study1_holdout_rows_{args.variant}.parquet")
    rows = load_rows(path, freeze_fields=exploratory)
    codes, players, weights = bootstrap_weights(rows, args.sample, label)
    log(f"{label}: rows {len(rows):,}; players {len(players):,}")
    means, contrasts = regional_rates(rows, codes, players, weights)
    means.to_csv(OUT / "regional_means_natural.csv", index=False)
    contrasts.to_csv(OUT / "regional_contrasts_natural.csv", index=False)
    log("regional natural-unit estimates done")
    composition(rows, codes, players, weights).to_csv(OUT / "ping_composition.csv", index=False)
    cliffs_delta(rows, codes, players).to_csv(OUT / "cliffs_delta.csv", index=False)
    log("composition and Cliff's delta done")
    champion_decomposition(rows, codes, players, weights).to_csv(OUT / "champion_decomposition.csv", index=False)
    same_champion_examples(rows).to_csv(OUT / "same_champion_examples.csv", index=False)
    log("champion analyses done")
    percentile_gradients(rows, codes, players, weights).to_csv(OUT / "percentile_gradients.csv", index=False)
    write_json(OUT / "run_info.json", {
        "label": label, "rows": int(len(rows)), "players": int(len(players)),
        "bootstrap_replicates": int(weights.shape[0]), "tiers": TIER_SET,
        "estimand": "equal weights across listing tiers and match positions; per 10 minutes",
        "ping_groups": PING_GROUPS, "analysed_fields": PINGS, "analysed_families": FAMILIES,
    })
    log("study 1 analysis complete")


if __name__ == "__main__":
    main()
