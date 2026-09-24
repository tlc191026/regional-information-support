from settings import *
from statistics_core import *
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import gc


def auc_influence(y, p):
    positive = p[y == 1]
    negative = p[y == 0]
    pos = np.sort(positive)
    neg = np.sort(negative)
    u = (
        np.searchsorted(neg, positive, "left") + np.searchsorted(neg, positive, "right")
    ) / (2 * len(neg))
    v = 1 - (
        np.searchsorted(pos, negative, "left") + np.searchsorted(pos, negative, "right")
    ) / (2 * len(pos))
    auc = float(u.mean())
    psi = np.empty(len(y))
    psi[y == 1] = (u - auc) / len(pos)
    psi[y == 0] = (v - auc) / len(neg)
    return auc, psi


def evaluate_scope(df, train_index, test_index, name, cats):
    train = df.loc[train_index].copy()
    test = df.loc[test_index].copy()
    assert not set(train.match_id) & set(test.match_id)
    scale = MaintenanceScale().fit(train)
    train["maintenance_diff_z"] = scale.apply(train).maintenance_diff_z
    test["maintenance_diff_z"] = scale.apply(test).maintenance_diff_z
    blocks = [
        ("M0 task context", []),
        ("M1 early economy", STATE[:3]),
        ("M2 early combat and objectives", STATE),
        ("M3 joint-action proxy", STATE + ["early_assists_pre_diff"]),
        (
            "M4 shared-information maintenance",
            STATE + ["early_assists_pre_diff", "maintenance_diff_z"],
        ),
    ]
    yt = test.first_post_neutral_blue.to_numpy(int)
    rows = []
    predictions = []
    prev = None
    unseen = {}
    for label, nums in blocks:
        design = Design(cats, nums).fit(train)
        xtr = design.transform(train)
        xte = design.transform(test)
        for i, col in enumerate(cats):
            unseen[col] = int(
                (~test[col].astype(str).isin(design.encoder.categories_[i])).sum()
            )
        model = fit_logistic(
            xtr, train.first_post_neutral_blue.to_numpy(int), label=name + " " + label
        )
        p = model.predict_proba(xte)[:, 1]
        predictions.append(p)
        row = {
            "validation": name,
            "model": label,
            "n_train": len(train),
            "n_test": len(test),
            "auc": roc_auc_score(yt, p),
            "log_loss": log_loss(yt, p),
            "brier": brier_score_loss(yt, p),
        }
        if prev:
            row.update(
                delta_auc=row["auc"] - prev["auc"],
                delta_log_loss=prev["log_loss"] - row["log_loss"],
                delta_brier=prev["brier"] - row["brier"],
            )
        rows.append(row)
        prev = row
        log(f'{name} {label}: AUC {row["auc"]:.6f}')
        del xtr, xte, model
        gc.collect()
    p0, p1 = predictions[-2:]
    a0, psi0 = auc_influence(yt, p0)
    a1, psi1 = auc_influence(yt, p1)
    ll0 = -(
        yt * np.log(np.clip(p0, 1e-15, 1))
        + (1 - yt) * np.log(np.clip(1 - p0, 1e-15, 1))
    )
    ll1 = -(
        yt * np.log(np.clip(p1, 1e-15, 1))
        + (1 - yt) * np.log(np.clip(1 - p1, 1e-15, 1))
    )
    ldiff = ll0 - ll1
    bdiff = (yt - p0) ** 2 - (yt - p1) ** 2
    influence = np.column_stack(
        [
            psi1 - psi0,
            (ldiff - ldiff.mean()) / len(yt),
            (bdiff - bdiff.mean()) / len(yt),
        ]
    )
    point = np.array([a1 - a0, ldiff.mean(), bdiff.mean()])
    strata = test.region.astype(str) + "|" + test.tier.astype(str)
    boot = (
        match_bootstrap(influence, strata.to_numpy(), "predict_" + name, B=1000) + point
    )
    delta = []
    for j, metric in enumerate(["auc", "log_loss", "brier"]):
        low, high = np.quantile(boot[:, j], [0.025, 0.975])
        delta.append(
            {
                "validation": name,
                "metric": metric,
                "improvement": point[j],
                "ci_low": low,
                "ci_high": high,
                "n_test": len(test),
                "interval_method": "paired stratified match bootstrap; AUC uses one-step placement influence",
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / f"prediction_{name}.csv", index=False)
    pd.DataFrame(delta).to_csv(TABLES / f"prediction_gain_{name}.csv", index=False)
    # Decile calibration for M4 is descriptive, separate from discrimination.
    cal = pd.DataFrame({"p": p1, "y": yt})
    cal["bin"] = pd.qcut(cal.p, 10, duplicates="drop")
    cal.groupby("bin", observed=True).agg(
        mean_prediction=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    ).reset_index().to_csv(TABLES / f"calibration_{name}.csv", index=False)
    metadata = {
        "validation": name,
        "n_train": len(train),
        "n_test": len(test),
        "same_match_across_sets": False,
        "preprocessing_fit_on": "training only",
        "maintenance_scale": scale.metadata(),
        "unseen_test_category_counts": unseen,
        "unseen_category_policy": "all-zero coding relative to training reference; reported, no test fitting",
        "train_first_date": str(train.game_date.min()),
        "train_last_date": str(train.game_date.max()),
        "test_first_date": str(test.game_date.min()),
        "test_last_date": str(test.game_date.max()),
    }
    write_json(QA / f"prediction_{name}.json", metadata)
    return rows, delta


def main():
    df = pd.read_parquet(DATA / "analysis_scored_0_10.parquet")
    df = df.dropna(subset=["first_post_neutral_blue", *STATE, "early_assists_pre_diff"])
    splits = pd.read_parquet(DATA / "focal_player_match_split.parquet")
    df = df.merge(splits, on="match_id", how="left", validate="one_to_one").reset_index(
        drop=True
    )
    allrows = []
    alldeltas = []
    # Verify actual retained focal identities rather than just asserting match disjointness.
    focus = pd.read_parquet(
        DATA / "canonical_focal_links.parquet", columns=["match_id", "puuid"]
    )
    links = focus.merge(df[["match_id", "focal_split"]], on="match_id", how="inner")
    ptrain = set(links.loc[links.focal_split.eq("train"), "puuid"])
    ptest = set(links.loc[links.focal_split.eq("test"), "puuid"])
    assert not ptrain & ptest
    write_json(
        QA / "retained_prediction_focal_identity_check.json",
        {
            "train_focal_players": len(ptrain),
            "test_focal_players": len(ptest),
            "overlap": 0,
            "non_focal_player_overlap_not_prohibited": True,
        },
    )
    del focus, links, ptrain, ptest
    gc.collect()
    r, d = evaluate_scope(
        df,
        df.index[df.focal_split.eq("train")],
        df.index[df.focal_split.eq("test")],
        "focal_player_holdout",
        ["region", "tier", "patch"],
    )
    allrows += r
    alldeltas += d
    r, d = evaluate_scope(
        df,
        df.index[df.focal_split.eq("train")],
        df.index[df.focal_split.eq("test")],
        "no_sampling_tier_holdout",
        ["region", "patch"],
    )
    allrows += r
    alldeltas += d
    train = []
    test = []
    cutoffs = []
    for region, g in df.dropna(subset=["game_date"]).groupby("region", observed=True):
        ordered = g.sort_values(["game_date", "match_id"])
        cut = ordered.game_date.iloc[int(len(ordered) * 0.70)]
        train.extend(ordered.index[ordered.game_date < cut])
        test.extend(ordered.index[ordered.game_date >= cut])
        cutoffs.append({"region": region, "cutoff_utc": str(cut)})
    write_json(QA / "temporal_holdout_cutoffs.json", cutoffs)
    r, d = evaluate_scope(
        df, train, test, "temporal_holdout", ["region", "tier", "patch"]
    )
    allrows += r
    alldeltas += d
    for region in REGIONS:
        r, d = evaluate_scope(
            df,
            df.index[~df.region.eq(region)],
            df.index[df.region.eq(region)],
            "heldout_" + region,
            ["tier", "patch"],
        )
        allrows += r
        alldeltas += d
    pd.DataFrame(allrows).to_csv(
        TABLES / "prediction_all_validation_scenarios.csv", index=False
    )
    pd.DataFrame(alldeltas).to_csv(
        TABLES / "prediction_gain_all_validation_scenarios.csv", index=False
    )
    log("Prediction validation finished")


if __name__ == "__main__":
    main()
