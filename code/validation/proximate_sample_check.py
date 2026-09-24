"""Compare core regional patterns in matches close to player-list collection."""

from common import *
import numpy as np
import pandas as pd
from scipy import sparse
import sqlite3

# Share the design and inference functions with the regional validation.
import regional_checks as regional

regional.USE_PSEUDOINVERSE = True
ns = vars(regional)
f = ns["f"]
dates = pd.read_parquet(
    ROOT
    / "manuscript_20260915/03_reproducibility/data/match_dates_and_metadata.parquet",
    columns=["match_id", "game_start_ms"],
)
db = sqlite3.connect(
    Path(os.environ["RESEARCH_DATABASE"]).resolve().as_uri() + "?mode=ro", uri=True
)
db.execute("PRAGMA query_only=ON")
players = pd.read_sql_query("SELECT puuid,created_at FROM players", db)
g = (
    f[["match_id", "puuid", "region"]]
    .merge(dates, on="match_id", how="left", validate="many_to_one")
    .merge(players, on="puuid", how="left", validate="many_to_one")
)
day = (
    pd.to_datetime(g.game_start_ms, unit="ms", utc=True)
    .dt.tz_localize(None)
    .dt.normalize()
    - pd.to_datetime(g.created_at).dt.normalize()
).dt.days
expected = g.region.replace({"NA": "NA1", "EUW": "EUW1"})
keep = (day.abs() <= 30) & g.match_id.str.split("_").str[0].eq(expected)
base_n = len(f)
f = f.loc[keep.to_numpy()].reset_index(drop=True)
pc = ns["pc"][keep.to_numpy()]
n = len(f)
ns.update(
    f=f,
    n=n,
    pc=pc,
    G=sparse.csr_matrix((np.ones(n), (pc, np.arange(n))), shape=(ns["ng"], n)),
    Y=ns["Y"][keep.to_numpy()],
    duration=ns["duration"][keep.to_numpy()],
    codes={k: v[keep.to_numpy()] for k, v in ns["codes"].items()},
)
rows = []
for spec, terms in [("additive", []), ("region_by_tier", ["tier"])]:
    X, L, LE, names = ns["design"](terms)
    goodcol = np.asarray(X.getnnz(axis=0) > 0).ravel()
    point, draw, *_ = ns["inference"](X[:, goodcol], np.vstack([L, LE])[:, goodcol])
    if spec == "additive":
        for a, b in [(2, 0), (2, 1), (1, 0)]:
            for measure, v in [
                ("maintenance", np.array([1, 0, 0])),
                ("signalling", np.array([0, 1, 0])),
                ("maintenance_minus_signalling", np.array([1, -1, 0])),
            ]:
                lo, hi = np.quantile((draw[:, a] - draw[:, b]) @ v, [0.025, 0.975])
                rows.append(
                    dict(
                        type=spec,
                        contrast=REG[a] + "-" + REG[b],
                        measure=measure,
                        estimate=(point[a] - point[b]) @ v,
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
    else:
        for j, measure in [(0, "maintenance"), (1, "signalling")]:
            ep = point[4::2, j] - point[3::2, j]
            dd = draw[:, 4::2, j] - draw[:, 3::2, j]
            for r in range(3):
                lo, hi = np.quantile(dd[:, r], [0.025, 0.975])
                rows.append(
                    dict(
                        type="tier_endpoint",
                        contrast=REG[r] + " Diamond-Iron",
                        measure=measure,
                        estimate=ep[r],
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
            for a, b in [(2, 0), (2, 1), (1, 0)]:
                lo, hi = np.quantile(dd[:, a] - dd[:, b], [0.025, 0.975])
                rows.append(
                    dict(
                        type="regional_tier_endpoint",
                        contrast=REG[a] + "-" + REG[b],
                        measure=measure,
                        estimate=ep[a] - ep[b],
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
save("proximate_sample_contrasts.csv", pd.DataFrame(rows))
save(
    "proximate_sample_counts.csv",
    f.groupby(["region", "tier"], observed=True)
    .agg(
        n_records=("match_id", "size"),
        n_players=("puuid", "nunique"),
        n_matches=("match_id", "nunique"),
    )
    .reset_index(),
)
dump(
    "proximate_sample_check.json",
    dict(
        n_records=n,
        n_players=f.puuid.nunique(),
        n_unique_matches=f.match_id.nunique(),
        retained_record_percent=100 * n / base_n,
        day_window=30,
        matching_platform_prefix=True,
        excluded_cross_platform_records=int(
            (~g.match_id.str.split("_").str[0].eq(expected)).sum()
        ),
        resampling="Original 126,000-player strata retained; players without eligible records contribute zero estimating scores.",
    ),
)
log("Proximate sample check complete")
