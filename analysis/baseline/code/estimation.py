"""Shared helpers for player-cluster bootstrap estimates in natural units."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BOOTSTRAP_WEIGHTS, POSITIONS, REGIONS, TIERS

CONTRASTS = [("KR", "NA1"), ("KR", "EUW1"), ("EUW1", "NA1")]


def player_order(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Players in the order used by the frozen bootstrap weights (sorted PUUID strings)."""
    labels, codes = np.unique(frame["puuid"].astype(str).to_numpy(), return_inverse=True)
    players = (frame[["puuid", "region", "tier"]].drop_duplicates("puuid").set_index("puuid").loc[labels]
               .reset_index())
    return codes.astype(np.int64), players


def load_weights(players: pd.DataFrame) -> np.ndarray:
    weights = np.load(BOOTSTRAP_WEIGHTS, mmap_mode="r")
    assert weights.shape == (1000, len(players))
    for _, members in players.groupby(["region", "tier"]).indices.items():
        assert np.all(np.asarray(weights[:, members]).sum(axis=1) == len(members))
    return np.asarray(weights, dtype=np.float32)


def cell_sums(codes: np.ndarray, n_players: int, cell: np.ndarray, n_cells: int, values: np.ndarray) -> np.ndarray:
    """Per-player sums of each value column within each cell: shape (players, cells, columns)."""
    key = codes * n_cells + cell
    summed = pd.DataFrame(values).groupby(key).sum()
    out = np.zeros((n_players * n_cells, values.shape[1]))
    out[summed.index.to_numpy()] = summed.to_numpy()
    return out.reshape(n_players, n_cells, values.shape[1])


def standardized_means(sums: np.ndarray, counts: np.ndarray, players: pd.DataFrame, weights: np.ndarray | None,
                       tiers: list[str] = TIERS):
    """Region means with equal weights across listing tiers and match positions.

    sums: (players, positions, metrics); counts: (players, positions).
    Returns point (regions, metrics) and replicates (B, regions, metrics) if weights are given.
    """
    n_metrics = sums.shape[2]
    point = np.zeros((len(REGIONS), n_metrics))
    boot = None if weights is None else np.zeros((weights.shape[0], len(REGIONS), n_metrics))
    flat = sums.reshape(sums.shape[0], -1)
    for r, region in enumerate(REGIONS):
        for tier in tiers:
            members = np.flatnonzero((players["region"] == region).to_numpy() & (players["tier"] == tier).to_numpy())
            s = flat[members].sum(axis=0).reshape(len(POSITIONS), n_metrics)
            c = counts[members].sum(axis=0)
            point[r] += (s / c[:, None]).mean(axis=0) / len(tiers)
            if weights is not None:
                w = weights[:, members]
                sb = (w @ flat[members]).reshape(-1, len(POSITIONS), n_metrics)
                cb = w @ counts[members]
                boot[:, r] += (sb / cb[:, :, None]).mean(axis=1) / len(tiers)
    return point, boot


def stratified_weights(players: pd.DataFrame, seed: int, reps: int = 1000) -> np.ndarray:
    """Player-cluster bootstrap counts resampled within region x tier strata."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((reps, len(players)), dtype=np.float32)
    for _, members in players.groupby(["region", "tier"]).indices.items():
        n = len(members)
        weights[:, members] = rng.multinomial(n, np.full(n, 1.0 / n), size=reps)
    return weights


def contrast_table(point: np.ndarray, boot: np.ndarray, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in CONTRASTS:
        i, j = REGIONS.index(a), REGIONS.index(b)
        for k, metric in enumerate(metrics):
            diff = point[i, k] - point[j, k]
            ratio = point[i, k] / point[j, k] if point[j, k] else np.nan
            bd = boot[:, i, k] - boot[:, j, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                br = boot[:, i, k] / boot[:, j, k]
            rows.append({
                "contrast": f"{a}-{b}", "metric": metric, "difference": diff,
                "difference_ci_low": np.percentile(bd, 2.5), "difference_ci_high": np.percentile(bd, 97.5),
                "ratio": ratio, "ratio_ci_low": np.nanpercentile(br, 2.5), "ratio_ci_high": np.nanpercentile(br, 97.5),
            })
    return pd.DataFrame(rows)


def mean_table(point: np.ndarray, boot: np.ndarray, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for r, region in enumerate(REGIONS):
        for k, metric in enumerate(metrics):
            rows.append({"region": region, "metric": metric, "estimate": point[r, k],
                         "ci_low": np.percentile(boot[:, r, k], 2.5), "ci_high": np.percentile(boot[:, r, k], 97.5)})
    return pd.DataFrame(rows)
