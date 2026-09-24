"""Shared numerical, export and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from constructs import MASTER_SEED

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "exp_20260717"
DIRS = {
    name: EXP / name
    for name in [
        "code",
        "data",
        "data_audit",
        "tables",
        "figures",
        "source_data",
        "extended_data",
        "reports",
        "manifest",
        "logs",
    ]
}


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def child_seed(namespace: str) -> int:
    digest = hashlib.blake2b(
        f"{MASTER_SEED}:{namespace}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def rng(namespace: str) -> np.random.Generator:
    return np.random.default_rng(child_seed(namespace))


def zscore(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - mean) / sd).astype(np.float32)


def winsorize(
    values, lo: float = 0.01, hi: float = 0.99
) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    qlo, qhi = np.nanquantile(arr, [lo, hi])
    return np.clip(arr, qlo, qhi), float(qlo), float(qhi)


def transform_metric(values) -> tuple[np.ndarray, float, float]:
    raw = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    logged = np.log1p(np.clip(raw, 0, None))
    clipped, qlo, qhi = winsorize(logged)
    return clipped, qlo, qhi


def one_hot(drop: str | None = "first") -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop=drop, sparse=True)


def control_design(
    df: pd.DataFrame, categorical: list[str], numeric: list[str], fit=None
):
    work = df[categorical + numeric].copy()
    for col in categorical:
        work[col] = work[col].astype(str)
    if fit is None:
        transformers = []
        if categorical:
            transformers.append(("cat", one_hot(), categorical))
        if numeric:
            transformers.append(("num", StandardScaler(), numeric))
        fit = ColumnTransformer(transformers, sparse_threshold=1.0)
        x = fit.fit_transform(work)
    else:
        x = fit.transform(work)
    return (
        sparse.hstack([sparse.csr_matrix(np.ones((len(df), 1))), x], format="csr"),
        fit,
    )


def residualize_metrics(
    df: pd.DataFrame,
    specs: list[dict],
    categorical: list[str],
    numeric: list[str],
    cache: Path,
    audit_path: Path,
) -> pd.DataFrame:
    if cache.exists():
        cached = pd.read_parquet(cache)
        expected = {item["metric"] for item in specs}
        if len(cached) == len(df) and expected.issubset(cached.columns):
            return cached[list(item["metric"] for item in specs)]
    x, _ = control_design(df, categorical, numeric)
    out = pd.DataFrame(index=df.index)
    audit = []
    for item in specs:
        metric = item["metric"]
        transformed, qlo, qhi = transform_metric(df[metric])
        model = Ridge(alpha=1e-6, solver="lsqr", fit_intercept=False).fit(
            x, transformed
        )
        residual = transformed - model.predict(x)
        if item.get("reverse", False):
            residual = -residual
        out[metric] = zscore(residual)
        audit.append(
            {
                "metric": metric,
                "reverse_coded": bool(item.get("reverse", False)),
                "winsor_log1p_p01": qlo,
                "winsor_log1p_p99": qhi,
                "residual_mean": float(out[metric].mean()),
                "residual_sd": float(out[metric].std(ddof=0)),
            }
        )
    out.to_parquet(cache, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    return out


def cohen_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(
        ((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def bootstrap_d(a, b, namespace: str, n_boot: int = 1000) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    generator = rng(namespace)
    observed = cohen_d(a, b)
    boots = np.empty(n_boot, dtype=np.float32)
    for idx in range(n_boot):
        boots[idx] = cohen_d(
            a[generator.integers(0, len(a), len(a))],
            b[generator.integers(0, len(b), len(b))],
        )
    return observed, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def fit_sparse_linear(
    x: sparse.csr_matrix, y: np.ndarray, interest_indices: list[int]
) -> dict:
    y = np.asarray(y, dtype=float)
    model = LinearRegression(fit_intercept=False).fit(x, y)
    pred = model.predict(x)
    residual = y - pred
    sse = float(residual @ residual)
    sst = float(np.sum((y - y.mean()) ** 2))
    xtx = (x.T @ x).toarray()
    inv = np.linalg.pinv(xtx)
    sigma2 = sse / max(len(y) - x.shape[1], 1)
    coef = np.asarray(model.coef_, dtype=float)
    rows = []
    for idx in interest_indices:
        se = math.sqrt(max(sigma2 * inv[idx, idx], 0.0))
        rows.append(
            (
                float(coef[idx]),
                float(se),
                float(coef[idx] - 1.96 * se),
                float(coef[idx] + 1.96 * se),
            )
        )
    return {"model": model, "pred": pred, "r2": 1 - sse / sst, "effects": rows}


def fit_sparse_logistic(
    x: sparse.csr_matrix, y: np.ndarray, interest_indices: list[int], sample_weight=None
) -> dict:
    y = np.asarray(y, dtype=int)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=500,
        fit_intercept=False,
        random_state=MASTER_SEED,
    )
    model.fit(x, y, sample_weight=sample_weight)
    beta = np.asarray(model.coef_[0], dtype=float)
    prob = model.predict_proba(x)[:, 1]
    weights = prob * (1 - prob)
    if sample_weight is not None:
        weights = weights * np.asarray(sample_weight, dtype=float)
    hessian = (x.T @ x.multiply(weights[:, None])).toarray()
    inv = np.linalg.pinv(hessian)
    rows = []
    for idx in interest_indices:
        se = math.sqrt(max(inv[idx, idx], 0.0))
        b = float(beta[idx])
        ame = float(np.mean(prob * (1 - prob)) * b)
        rows.append((b, se, b - 1.96 * se, b + 1.96 * se, float(np.exp(b)), ame))
    return {"model": model, "prob": prob, "effects": rows}


def set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def runtime_metadata() -> dict:
    import matplotlib
    import scipy
    import sklearn

    return {
        "master_seed": MASTER_SEED,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
