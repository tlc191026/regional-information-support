"""Linear estimators with two-way cluster-robust inference (focal player and match).

All estimators work on variables from which the controls have already been partialled out
(Frisch-Waugh-Lovell). Each estimator returns point estimates and per-row influence values whose
cluster sums give the sandwich variance; ``twoway_covariance`` combines them across estimators that
share clusters (Cameron, Gelbach and Miller, 2011).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def one_hot_design(cats: list[np.ndarray], conts: np.ndarray | None, rows: slice | np.ndarray,
                   levels: list[int]) -> np.ndarray:
    """Dense design for a block of rows: all levels of the first factor, drop-first for the others."""
    first = cats[0][rows]
    n = len(first)
    blocks = [np.eye(levels[0])[first]]
    for codes, k in zip(cats[1:], levels[1:]):
        if k > 1:
            blocks.append(np.eye(k)[codes[rows]][:, 1:])
    if conts is not None and conts.shape[1]:
        blocks.append(conts[rows])
    return np.hstack(blocks) if blocks else np.ones((n, 1))


def residualize(values: np.ndarray, cats: list[np.ndarray], conts: np.ndarray | None = None,
                chunk: int = 400_000) -> np.ndarray:
    """Residuals of each column of ``values`` on categorical fixed effects and continuous controls."""
    n = values.shape[0]
    cats = [np.asarray(pd.factorize(c)[0], dtype=np.int64) for c in cats]
    levels = [int(c.max()) + 1 for c in cats]
    if conts is not None:
        conts = np.asarray(conts, dtype=float)
    width = levels[0] + sum(max(k - 1, 0) for k in levels[1:]) + (0 if conts is None else conts.shape[1])
    xtx = np.zeros((width, width))
    xty = np.zeros((width, values.shape[1]))
    for start in range(0, n, chunk):
        block = slice(start, min(n, start + chunk))
        x = one_hot_design(cats, conts, block, levels)
        xtx += x.T @ x
        xty += x.T @ values[block]
    coef = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    out = np.empty_like(values, dtype=float)
    for start in range(0, n, chunk):
        block = slice(start, min(n, start + chunk))
        out[block] = values[block] - one_hot_design(cats, conts, block, levels) @ coef
    return out


@dataclass
class Estimate:
    names: list[str]
    beta: np.ndarray
    psi: np.ndarray          # rows x parameters; sum over a cluster gives that cluster's score
    focal: np.ndarray        # cluster codes for each row
    match: np.ndarray
    extra: dict


def ols(y: np.ndarray, x: np.ndarray, names: list[str], focal: np.ndarray, match: np.ndarray) -> Estimate:
    x = x.reshape(len(y), -1)
    xtx = x.T @ x
    inv = np.linalg.inv(xtx)
    beta = inv @ (x.T @ y)
    resid = y - x @ beta
    psi = (x * resid[:, None]) @ inv
    return Estimate(names, beta, psi, focal, match, {"n": int(len(y))})


def oriv(y: np.ndarray, ta: np.ndarray, tb: np.ndarray, name: str, focal: np.ndarray, match: np.ndarray) -> Estimate:
    """Obviously related instrumental variables with two split-half measures (Gillen et al., 2019)."""
    c = float(ta @ tb)
    sa, sb = float(ta @ ta), float(tb @ tb)
    pi1, pi2 = c / sb, c / sa
    denom = pi1 * c + pi2 * c
    beta = (pi1 * float(tb @ y) + pi2 * float(ta @ y)) / denom
    psi = (pi1 * tb * (y - beta * ta) + pi2 * ta * (y - beta * tb)) / denom
    first_stage_r = c / np.sqrt(sa * sb)
    return Estimate([name], np.array([beta]), psi[:, None], focal, match,
                    {"n": int(len(y)), "split_half_r_rows": float(first_stage_r)})


def _cluster_sums(codes: np.ndarray, psi: np.ndarray, n_groups: int) -> np.ndarray:
    return np.column_stack([np.bincount(codes, weights=psi[:, j], minlength=n_groups) for j in range(psi.shape[1])])


def twoway_covariance(estimates: list[Estimate]) -> np.ndarray:
    """Joint two-way cluster-robust covariance for all parameters of the given estimates.

    Clusters: focal player, match, and their intersection; small-sample factor G/(G-1) per dimension.
    Negative eigenvalues of the combined matrix are set to zero.
    """
    focal = np.concatenate([e.focal for e in estimates])
    match = np.concatenate([e.match for e in estimates])
    f_codes, f_uni = pd.factorize(focal)
    m_codes, m_uni = pd.factorize(match)
    i_codes, i_uni = pd.factorize(f_codes.astype(np.int64) * (len(m_uni) + 1) + m_codes)
    width = sum(e.psi.shape[1] for e in estimates)
    total = 0.0
    for codes, groups, sign in [(f_codes, len(f_uni), 1.0), (m_codes, len(m_uni), 1.0), (i_codes, len(i_uni), -1.0)]:
        sums = np.zeros((groups, width))
        start_row, col = 0, 0
        for e in estimates:
            rows = slice(start_row, start_row + len(e.focal))
            sums[:, col:col + e.psi.shape[1]] = _cluster_sums(codes[rows], e.psi, groups)
            start_row += len(e.focal)
            col += e.psi.shape[1]
        total = total + sign * (groups / (groups - 1)) * (sums.T @ sums)
    vals, vecs = np.linalg.eigh((total + np.transpose(total)) / 2)
    return (vecs * np.clip(vals, 0, None)) @ vecs.T


def oneway_covariance(estimates: list[Estimate], dimension: str) -> np.ndarray:
    codes_all = np.concatenate([getattr(e, dimension) for e in estimates])
    codes, uni = pd.factorize(codes_all)
    width = sum(e.psi.shape[1] for e in estimates)
    sums = np.zeros((len(uni), width))
    start_row, col = 0, 0
    for e in estimates:
        rows = slice(start_row, start_row + len(e.focal))
        sums[:, col:col + e.psi.shape[1]] = _cluster_sums(codes[rows], e.psi, len(uni))
        start_row += len(e.focal)
        col += e.psi.shape[1]
    return (len(uni) / (len(uni) - 1)) * (sums.T @ sums)


class ClusterCodes:
    """Precomputed focal, match and intersection codes for a fixed stack of rows."""

    def __init__(self, focal: np.ndarray, match: np.ndarray) -> None:
        self.f, uf = pd.factorize(focal)
        self.m, um = pd.factorize(match)
        self.i, ui = pd.factorize(self.f.astype(np.int64) * (len(um) + 1) + self.m)
        self.nf, self.nm, self.ni = len(uf), len(um), len(ui)

    def covariance(self, psi: np.ndarray, twoway: bool = True) -> np.ndarray:
        psi = psi.reshape(len(self.f), -1)
        dims = [(self.f, self.nf, 1.0), (self.m, self.nm, 1.0), (self.i, self.ni, -1.0)] if twoway else [(self.f, self.nf, 1.0)]
        total = np.zeros((psi.shape[1], psi.shape[1]))
        for codes, groups, sign in dims:
            sums = _cluster_sums(codes, psi, groups)
            total += sign * (groups / (groups - 1)) * (sums.T @ sums)
        vals, vecs = np.linalg.eigh((total + total.T) / 2)
        return (vecs * np.clip(vals, 0, None)) @ vecs.T
