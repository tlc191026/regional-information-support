"""Check regional contrasts, alternative signal weights and bootstrap approximation."""

from common import *
import numpy as np
import pandas as pd
from scipy import sparse, stats
import gc

log("Read fixed participant scores")
src = ROOT / "exp_20260717/data"
f = pd.read_parquet(
    src / "context_unresidualized_participant_scores.parquet",
    columns=[
        "match_id",
        "puuid",
        "region",
        "tier",
        "team_position",
        "patch",
        "game_duration",
        M,
        S,
    ],
)
z = pd.read_parquet(src / "participant_metrics_transformed_z.parquet")
z = z.astype(float)


def standard(a):
    return (a - a.mean()) / a.std(ddof=0)


m = standard(
    z[["wards_placed_pm", "wards_killed_pm", "control_wards_bought_pm"]]
    .mean(axis=1)
    .to_numpy()
)
s = standard(
    np.column_stack(
        [
            standard(z[PINGS[a:b]].mean(axis=1).to_numpy())
            for a, b in [(0, 3), (3, 5), (5, 9)]
        ]
    ).mean(axis=1)
)
assert np.max(abs(m - f[M])) < 1e-6
assert np.max(abs(s - f[S])) < 2e-6
f["equal_ping_signalling"] = standard(z[PINGS].mean(axis=1).to_numpy())
del z, m, s
gc.collect()
f.region = f.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
valid = f.team_position.astype(str).isin(POS)
assert (~valid).sum() == 6
f = f.loc[valid].reset_index(drop=True)
n = len(f)
ids, pc = np.unique(f.puuid.astype(str), return_inverse=True)
ng = len(ids)
players = (
    f[["puuid", "region", "tier"]]
    .drop_duplicates("puuid")
    .set_index("puuid")
    .loc[ids]
    .reset_index()
)
strata = players.groupby(["region", "tier"], observed=True).indices
assert len(strata) == 21 and all(len(i) == 6000 for i in strata.values())
G = sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ng, n))
weights = np.load(src / "bootstrap_weights_primary_region_tier_1000.npy", mmap_mode="r")
patches = sorted(
    f.patch.astype(str).unique(), key=lambda x: tuple(map(int, x.split(".")))
)
levels = {"region": REG, "tier": TIERS, "team_position": POS, "patch": patches}
codes = {
    k: pd.Categorical(f[k].astype(str), categories=v).codes for k, v in levels.items()
}
duration = standard(f.game_duration.to_numpy(float))
Y = f[[M, S, "equal_ping_signalling"]].to_numpy(float)


def design(interactions):
    row = [np.arange(n)]
    col = [np.zeros(n, dtype=int)]
    data = [np.ones(n)]
    names = ["Intercept"]
    where = {}
    iw = {}
    for key, categories in levels.items():
        c = codes[key]
        use = c > 0
        start = len(names)
        where[key] = (start, categories)
        row.append(np.flatnonzero(use))
        col.append(start + c[use] - 1)
        data.append(np.ones(use.sum()))
        names += [key + "_" + v for v in categories[1:]]
    row.append(np.arange(n))
    col.append(np.full(n, len(names)))
    data.append(duration)
    names.append("duration_z")
    for key in interactions:
        for r in [1, 2]:
            for t in range(1, len(levels[key])):
                ix = np.flatnonzero((codes["region"] == r) & (codes[key] == t))
                iw[(key, r, t)] = len(names)
                row.append(ix)
                col.append(np.full(len(ix), len(names)))
                data.append(np.ones(len(ix)))
                names.append(f"region_{REG[r]}:{key}_{levels[key][t]}")
    X = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
        shape=(n, len(names)),
    )

    def ref(r, tier=None):
        v = np.zeros(len(names))
        v[0] = 1
        for key, (start, categories) in where.items():
            if key == "region":
                if r:
                    v[start + r - 1] = 1
            elif key == "tier" and tier is not None:
                if tier:
                    v[start + tier - 1] = 1
            elif key == "patch":
                v[start : start + len(categories) - 1] = (
                    np.bincount(codes[key], minlength=len(categories))[1:] / n
                )
            else:
                v[start : start + len(categories) - 1] = 1 / len(categories)
        for (key, rr, t), ix in iw.items():
            if r == rr:
                v[ix] = (
                    float(t == tier)
                    if key == "tier" and tier is not None
                    else 1 / len(levels[key])
                )
        return v

    return (
        X,
        np.array([ref(r) for r in range(3)]),
        np.array([ref(r, t) for r in range(3) for t in [0, 6]]),
        names,
    )


USE_PSEUDOINVERSE = False


def inference(X, L):
    gram = (X.T @ X).toarray()
    if USE_PSEUDOINVERSE:
        bread = np.linalg.pinv(gram, rcond=1e-12)
        assert np.allclose(L, L @ bread @ gram, atol=1e-8)
    else:
        assert np.linalg.matrix_rank(gram) == len(gram)
        bread = np.linalg.inv(gram)
    beta = bread @ (X.T @ Y)
    E = Y - X @ beta
    J = np.stack(
        [(G @ X.multiply(E[:, j, None])).toarray() @ (bread @ L.T) for j in range(3)],
        axis=2,
    )
    point = L @ beta
    J = J.reshape(ng, -1)
    for ix in strata.values():
        J[ix] -= J[ix].mean(axis=0)
    draws = np.empty((1000, J.shape[1]))
    for b in range(0, 1000, 25):
        draws[b : b + 25] = (
            point.ravel() + (np.asarray(weights[b : b + 25], float) - 1) @ J
        )
    return point, draws.reshape(1000, len(L), 3), J, bread, beta


def main():
    rows = []
    eqrows = []
    for label, terms in [
        ("additive", []),
        ("region_by_tier", ["tier"]),
        ("region_by_position", ["team_position"]),
        ("both_interactions", ["tier", "team_position"]),
    ]:
        log("Fit " + label)
        X, L, LE, names = design(terms)
        point, draw, J, bread, beta = inference(X, np.vstack([L, LE]))
        for a, b in [(2, 0), (2, 1), (1, 0)]:
            for signal, j in [("functional_groups", 1), ("equal_nine_pings", 2)]:
                for measure, v in [
                    ("maintenance", np.array([1.0, 0, 0])),
                    ("signalling", np.eye(3)[j]),
                    (
                        "maintenance_minus_signalling",
                        np.array([1.0, 0, 0]) - np.eye(3)[j],
                    ),
                ]:
                    p = (point[a] - point[b]) @ v
                    d = (draw[:, a] - draw[:, b]) @ v
                    lo, hi = np.quantile(d, [0.025, 0.975])
                    rows.append(
                        dict(
                            specification=label,
                            signal_score=signal,
                            contrast=REG[a] + "-" + REG[b],
                            measure=measure,
                            estimate=p,
                            ci_low=lo,
                            ci_high=hi,
                            bootstrap_sd=d.std(ddof=1),
                            n_records=n,
                            n_players=ng,
                        )
                    )
        if label == "region_by_tier":
            for j, signal in [(1, "functional_groups"), (2, "equal_nine_pings")]:
                endpoints = point[4::2, j] - point[3::2, j]
                dd = draw[:, 4::2, j] - draw[:, 3::2, j]
                for r in range(3):
                    lo, hi = np.quantile(dd[:, r], [0.025, 0.975])
                    eqrows.append(
                        dict(
                            score=signal,
                            contrast=REG[r] + " Diamond-Iron",
                            estimate=endpoints[r],
                            ci_low=lo,
                            ci_high=hi,
                        )
                    )
                for a, b in [(2, 0), (2, 1), (1, 0)]:
                    lo, hi = np.quantile(dd[:, a] - dd[:, b], [0.025, 0.975])
                    eqrows.append(
                        dict(
                            score=signal,
                            contrast=REG[a] + "-" + REG[b] + " Diamond-Iron",
                            estimate=endpoints[a] - endpoints[b],
                            ci_low=lo,
                            ci_high=hi,
                        )
                    )
        if label == "additive":
            log("Paired full-refit bootstrap in 2,100-player subsample")
            rng = np.random.default_rng(SEED)
            chosen = np.sort(
                np.concatenate(
                    [rng.choice(ix, 100, replace=False) for ix in strata.values()]
                )
            )
            sub = np.flatnonzero(np.isin(pc, chosen))
            cs = np.searchsorted(chosen, pc[sub])
            xs = X[sub].toarray()
            ys = Y[sub, :2]
            # Sufficient statistics retain every selected player's observed records.
            A = np.zeros((len(chosen), xs.shape[1], xs.shape[1]))
            B = np.zeros((len(chosen), xs.shape[1], 2))
            for i in range(len(chosen)):
                ix = cs == i
                A[i] = xs[ix].T @ xs[ix]
                B[i] = xs[ix].T @ ys[ix]
            gram = A.sum(0)
            br = np.linalg.pinv(gram, rcond=1e-12)
            bt = br @ B.sum(0)
            C = np.array([L[a] - L[b] for a, b in [(2, 0), (2, 1), (1, 0)]])

            def estim(b):
                q = C @ b
                return np.column_stack([q[:, 0], q[:, 1], q[:, 0] - q[:, 1]]).ravel()

            base = estim(bt)
            cluster = B - np.einsum("ijk,kl->ijl", A, bt)
            strsub = [np.flatnonzero(np.isin(chosen, ix)) for ix in strata.values()]
            one = []
            full = []
            for b in range(300):
                counts = np.zeros(len(chosen))
                for ix in strsub:
                    counts[ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)))
                one.append(estim(bt + br @ np.einsum("i,ijk->jk", counts - 1, cluster)))
                gf = np.einsum("i,ijk->jk", counts, A)
                bf = np.einsum("i,ijk->jk", counts, B)
                full.append(estim(np.linalg.pinv(gf, rcond=1e-12) @ bf))
            one = np.array(one)
            full = np.array(full)
            qa = []
            for k in range(9):
                l1, h1 = np.quantile(one[:, k], [0.025, 0.975])
                l2, h2 = np.quantile(full[:, k], [0.025, 0.975])
                qa.append(
                    dict(
                        contrast=["KR-NA", "KR-EUW", "EUW-NA"][k // 3],
                        measure=[
                            "maintenance",
                            "signalling",
                            "maintenance_minus_signalling",
                        ][k % 3],
                        estimate=base[k],
                        one_step_sd=one[:, k].std(ddof=1),
                        full_refit_sd=full[:, k].std(ddof=1),
                        sd_ratio=one[:, k].std(ddof=1) / full[:, k].std(ddof=1),
                        one_step_low=l1,
                        one_step_high=h1,
                        full_refit_low=l2,
                        full_refit_high=h2,
                        paired_rmse=float(
                            np.sqrt(np.mean((one[:, k] - full[:, k]) ** 2))
                        ),
                        paired_correlation=float(
                            np.corrcoef(one[:, k], full[:, k])[0, 1]
                        ),
                        n_records=len(sub),
                        n_players=len(chosen),
                        replicates=300,
                    )
                )
            save("bootstrap_ols_comparison.csv", pd.DataFrame(qa))
            np.savez_compressed(
                W / "data/bootstrap_ols_draws.npz", one_step=one, full_refit=full
            )
        del X, J
        gc.collect()
    save("regional_interaction_standardization.csv", pd.DataFrame(rows))
    save("equal_ping_tier_endpoints.csv", pd.DataFrame(eqrows))
    dump(
        "regional_checks.json",
        dict(
            n_records=n,
            n_players=ng,
            n_unique_matches=int(f.match_id.nunique()),
            score_reconstruction_verified=True,
            seed=SEED,
            main_bootstrap_replicates=1000,
            validation_bootstrap_replicates=300,
        ),
    )
    log("Regional checks complete")


if __name__ == "__main__":
    main()
