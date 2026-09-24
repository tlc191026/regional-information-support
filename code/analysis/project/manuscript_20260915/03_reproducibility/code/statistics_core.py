from settings import *
import warnings
import numpy as np
import pandas as pd
from scipy import sparse, special, stats
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

STATE = [
    "gold_at_window_diff",
    "xp_at_window_diff",
    "level_at_window_diff",
    "kills_pre_diff",
    "neutral_objectives_pre_diff",
    "turret_plates_pre_diff",
]
COMPONENTS = ["valid_wards_placed", "valid_ward_kills", "control_ward_purchases"]


class MaintenanceScale:
    def fit(self, df, components=COMPONENTS):
        self.components = list(components)
        self.params = {}
        values = []
        for col in components:
            a = np.log1p(
                np.maximum(
                    np.r_[
                        df["blue_" + col].to_numpy(float),
                        df["red_" + col].to_numpy(float),
                    ],
                    0,
                )
            )
            low, high = np.quantile(a, [0.01, 0.99])
            a = np.clip(a, low, high)
            mean = a.mean()
            sd = a.std()
            self.params[col] = [float(low), float(high), float(mean), float(sd)]
            values.append((a - mean) / (sd or 1.0))
        score = np.mean(values, axis=0)
        self.score_mean = float(score.mean())
        self.score_sd = float(score.std())
        n = len(df)
        diff = (score[:n] - score[n:]) / self.score_sd
        self.diff_mean = float(diff.mean())
        self.diff_sd = float(diff.std())
        return self

    def apply(self, df):
        result = {}
        sides = []
        for side in ["blue", "red"]:
            arr = []
            for col in self.components:
                low, high, mean, sd = self.params[col]
                z = (
                    np.clip(
                        np.log1p(np.maximum(df[side + "_" + col].to_numpy(float), 0)),
                        low,
                        high,
                    )
                    - mean
                ) / (sd or 1.0)
                arr.append(z)
                result[side + "_" + col + "_z"] = z
            result[side + "_maintenance"] = (np.mean(arr, axis=0) - self.score_mean) / (
                self.score_sd or 1.0
            )
            sides.append(result[side + "_maintenance"])
        result["maintenance_diff_raw"] = sides[0] - sides[1]
        result["maintenance_diff_z"] = (sides[0] - sides[1] - self.diff_mean) / (
            self.diff_sd or 1.0
        )
        return pd.DataFrame(result, index=df.index)

    def metadata(self):
        return {
            "components": self.components,
            "log1p_winsor_mean_sd": self.params,
            "score_mean": self.score_mean,
            "score_sd": self.score_sd,
            "diff_mean": self.diff_mean,
            "diff_sd": self.diff_sd,
            "both_sides_pooled": True,
        }


def add_differences(df):
    for col in [
        "gold_at_window",
        "xp_at_window",
        "level_at_window",
        "kills_pre",
        "neutral_objectives_pre",
        "turret_plates_pre",
        "early_assists_pre",
        "neutral_objectives_post",
        "building_objectives_post",
        "kills_post",
    ]:
        df[col + "_diff"] = df["blue_" + col].to_numpy(float) - df[
            "red_" + col
        ].to_numpy(float)
    for col, stem in [("gold", "gold"), ("damage", "damage")]:
        df[col + "_growth_post_diff"] = (
            df["blue_" + stem + "_final"] - df["blue_" + stem + "_at_window"]
        ) - (df["red_" + stem + "_final"] - df["red_" + stem + "_at_window"])
    return df


class Design:
    def __init__(self, cats, nums):
        self.cats = list(cats)
        self.nums = list(nums)

    def fit(self, df):
        self.encoder = OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=True, dtype=np.float64
        )
        if self.cats:
            self.encoder.fit(df[self.cats].astype(str))
        self.mean = df[self.nums].mean().to_numpy(float) if self.nums else np.zeros(0)
        self.sd = (
            df[self.nums].std(ddof=0).replace(0, 1).to_numpy(float)
            if self.nums
            else np.zeros(0)
        )
        self.names = (
            ["intercept"]
            + (
                self.encoder.get_feature_names_out(self.cats).tolist()
                if self.cats
                else []
            )
            + self.nums
        )
        return self

    def transform(self, df):
        arr = [sparse.csr_matrix(np.ones((len(df), 1)))]
        if self.cats:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                arr.append(self.encoder.transform(df[self.cats].astype(str)))
        if self.nums:
            arr.append(
                sparse.csr_matrix((df[self.nums].to_numpy(float) - self.mean) / self.sd)
            )
        return sparse.hstack(arr, format="csr")


def design_fit(df, predictors, cats=["region", "tier", "patch"], controls=STATE):
    d = Design(cats, controls).fit(df)
    x = d.transform(df)
    x = sparse.hstack(
        [x, sparse.csr_matrix(df[predictors].to_numpy(float))], format="csr"
    )
    names = d.names + list(predictors)
    return x, names, d


def fit_logistic(x, y, weights=None, label="logistic"):
    # A Newton solve avoids the slow L-BFGS progress caused by highly correlated
    # gold, experience and level controls. The very weak ridge only stabilizes
    # redundant nuisance columns; all inferential uncertainty uses match scores.
    model = LogisticRegression(
        C=1e6,
        solver="newton-cholesky",
        fit_intercept=False,
        max_iter=100,
        tol=1e-8,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y, sample_weight=weights)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Logistic model failed to converge: " + label)
    prob = model.predict_proba(x)[:, 1]
    residual = np.asarray(y) - prob
    if weights is not None:
        residual = residual * np.asarray(weights)
    gradient = np.asarray(x.T @ residual).ravel() - model.coef_[0] / 1e6
    denom = len(y) if weights is None else np.sum(weights)
    score_max = float(np.max(np.abs(gradient)) / denom)
    if not np.isfinite(prob).all() or score_max > 2e-7:
        raise RuntimeError(f"Unacceptable logistic score residual {score_max}: {label}")
    model.fit_diagnostics_ = {
        "solver": "newton-cholesky",
        "max_normalized_score": score_max,
        "iterations": int(model.n_iter_[0]),
        "C": 1e6,
        "n": len(y),
        "p": x.shape[1],
        "warnings": [str(w.message) for w in caught],
    }
    with (LOGS / "logistic_convergence.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(label=label, **model.fit_diagnostics_)) + "\n")
    return model


def binary_fit(
    df,
    outcome="first_post_neutral_blue",
    predictors=["maintenance_diff_z"],
    cats=["region", "tier", "patch"],
    controls=STATE,
    weights=None,
):
    x, names, design = design_fit(df, predictors, cats, controls)
    y = df[outcome].to_numpy(float)
    model = fit_logistic(x, y, weights, label=outcome + " | " + ",".join(predictors))
    prob = model.predict_proba(x)[:, 1]
    res = y - prob
    w = prob * (1 - prob)
    if weights is not None:
        w = w * weights
        res = res * weights
    h = (x.T @ x.multiply(w[:, None])).toarray()
    bread = np.linalg.pinv(h, rcond=1e-12)
    idx = np.arange(len(names) - len(predictors), len(names))
    psi = np.asarray(x @ bread[:, idx]) * res[:, None]
    covariance = psi.T @ psi
    effects = []
    for j, name in enumerate(predictors):
        b = float(model.coef_[0, idx[j]])
        se = float(np.sqrt(covariance[j, j]))
        p = float(2 * stats.norm.sf(abs(b / se)))
        effects.append(
            {
                "predictor": name,
                "beta": b,
                "se": se,
                "ci_low": b - 1.96 * se,
                "ci_high": b + 1.96 * se,
                "odds_ratio": float(np.exp(b)),
                "or_low": float(np.exp(b - 1.96 * se)),
                "or_high": float(np.exp(b + 1.96 * se)),
                "p_value": p,
                "n_matches": len(df),
                "outcome": outcome,
                "model": "logistic",
                "interval_method": "match-score HC0 Wald",
            }
        )
    return {
        "x": x,
        "model": model,
        "prob": prob,
        "residual": res,
        "bread": bread,
        "psi": psi,
        "effects": effects,
        "interest": idx,
        "design": design,
        "names": names,
    }


def probability_effects(fit, df, score="maintenance_diff_z"):
    x = fit["x"]
    idx = fit["names"].index(score)
    coef = fit["model"].coef_[0]
    eta = x @ coef
    rows = []
    influences = []
    for value in [-1.0, 0.0, 1.0]:
        p = special.expit(eta + coef[idx] * (value - df[score].to_numpy(float)))
        w = p * (1 - p)
        grad = np.asarray(x.T @ w).ravel() / len(p)
        grad[idx] = w.mean() * value
        psi = np.asarray(x @ (fit["bread"] @ grad)).ravel() * fit["residual"]
        rows.append(
            {
                "maintenance_sd": value,
                "probability": float(p.mean()),
                "n_matches": len(p),
            }
        )
        influences.append(psi)
    return rows, np.column_stack(influences)


def linear_fit(
    df,
    outcome,
    predictors=["maintenance_diff_z"],
    cats=["region", "tier", "patch"],
    controls=STATE,
):
    x, names, _ = design_fit(df, predictors, cats, controls)
    y = df[outcome].to_numpy(float)
    y = (y - y.mean()) / (y.std() or 1.0)
    coef = lsqr(x, y, atol=1e-9, btol=1e-9, iter_lim=500)[0]
    res = y - x @ coef
    bread = np.linalg.pinv((x.T @ x).toarray(), rcond=1e-12)
    idx = np.arange(len(names) - len(predictors), len(names))
    psi = np.asarray(x @ bread[:, idx]) * res[:, None]
    effects = []
    for j, name in enumerate(predictors):
        b = float(coef[idx[j]])
        se = float(np.sqrt(psi[:, j] @ psi[:, j]))
        effects.append(
            {
                "predictor": name,
                "beta": b,
                "se": se,
                "ci_low": b - 1.96 * se,
                "ci_high": b + 1.96 * se,
                "p_value": float(2 * stats.norm.sf(abs(b / se))),
                "n_matches": len(df),
                "outcome": outcome,
                "model": "standardized linear",
                "interval_method": "match-score HC0 Wald",
            }
        )
    return effects


def match_bootstrap(influence, strata, name, B=1000):
    """Stratified pairs resampling, evaluated by a one-step influence update.

    Each row is the SUM of score contributions for one unique match.
    The same resampled match indices are used for all reported primary estimands.
    """
    groups = [np.flatnonzero(np.asarray(strata) == v) for v in pd.unique(strata)]
    output = np.zeros((B, influence.shape[1]))
    generator = np.random.default_rng(child_seed(name))
    baseline = influence.sum(axis=0)
    for b in range(B):
        s = np.zeros(influence.shape[1])
        for g in groups:
            idx = g[generator.integers(0, len(g), size=len(g))]
            s += influence[idx].sum(axis=0)
        output[b] = s - baseline
        if (b + 1) % 100 == 0:
            log(f"Bootstrap {name}: {b+1}/{B}")
    return output


def bh_q(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    out[order] = np.minimum.accumulate(
        (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
    )[::-1].clip(0, 1)
    return out


def holm_p(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))).clip(
        0, 1
    )
    return out
