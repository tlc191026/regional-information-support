from settings import *
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import collections


def frames(kind, suffix=""):
    base = (DATA / "aligned") if kind == "timeline" else DATA
    paths = sorted((base / (kind + "_shards")).glob("*" + suffix + ".parquet"))
    if suffix == "":
        paths = [
            p
            for p in paths
            if not p.name.endswith((".people.parquet", ".errors.parquet"))
        ]
    if kind == "timeline" and suffix == "":
        paths = [p for p in paths if pq.ParquetFile(p).metadata.num_rows > 0]
    return paths


def main():
    for kind in ["timeline", "matchdto"]:
        expected = len(
            json.loads(
                (DATA / f"{kind}_source_file_inventory.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        base = (DATA / "aligned") if kind == "timeline" else DATA
        marks = list((base / (kind + "_shards")).glob("*.complete.json"))
        assert len(marks) == expected, (kind, len(marks), expected)
    meta = pd.read_parquet(DATA / "timeline_targets_all.parquet")
    details = pd.concat(
        (pd.read_parquet(p) for p in frames("matchdto")), ignore_index=True
    )
    assert details.match_id.is_unique
    details.to_parquet(
        DATA / "match_dates_and_metadata.parquet", index=False, compression="zstd"
    )
    meta = meta.merge(details, on="match_id", how="left", validate="one_to_one")
    meta["raw_metadata_available"] = meta.queue_id_raw.notna()
    meta["metadata_valid"] = ~meta.raw_metadata_available | (
        meta.queue_id_raw.eq(420)
        & meta.game_duration_raw.ge(900)
        & meta.patch.eq(meta.patch_raw)
    )
    meta["game_date"] = pd.to_datetime(
        meta.game_start_ms, unit="ms", utc=True, errors="coerce"
    )
    meta.to_parquet(
        DATA / "canonical_match_audit.parquet", index=False, compression="zstd"
    )
    timeline_errors = pd.concat(
        (pd.read_parquet(p) for p in frames("timeline", ".errors")), ignore_index=True
    )
    known_after_identity = {
        "invalid_participant_ids",
        "invalid_participant_puuids",
        "invalid_frame_order",
        "missing_or_conflicting_game_end",
        "incomplete_participant_frame",
        "missing_frame_state",
    }
    identity_pass_errors = set(
        timeline_errors.loc[
            timeline_errors.reason.isin(known_after_identity), "match_id"
        ]
    )
    meta.groupby(["region", "tier", "patch", "status"], observed=True).size().rename(
        "n_matches"
    ).reset_index().to_csv(
        TABLES / "timeline_indexed_coverage_by_patch.csv", index=False
    )
    keep = [
        "match_id",
        "region",
        "tier",
        "patch",
        "game_duration",
        "game_date",
        "metadata_valid",
        "raw_metadata_available",
        "blue_win_raw",
    ]
    audit = []
    for window in ["0_10", "0_15", "strict_0_10", "strict_0_15"]:
        log("Assembling " + window)
        chunks = [
            pd.read_parquet(p, filters=[("window", "==", window)])
            for p in frames("timeline")
        ]
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        assert df.match_id.is_unique
        df = df.merge(meta[keep], on="match_id", how="left", validate="one_to_one")
        df["winner_agreement"] = df.blue_win_raw.isna() | df.blue_win.eq(
            df.blue_win_raw
        )
        df["analysis_eligible"] = (
            df.window_valid & df.metadata_valid & df.winner_agreement
        )
        for key in ["window_reason", "first_status"]:
            df[key] = df[key].fillna("window_invalid")
        df[
            [
                "match_id",
                "region",
                "tier",
                "patch",
                "window_valid",
                "window_reason",
                "metadata_valid",
                "winner_agreement",
                "analysis_eligible",
                "first_status",
                "state_lag_ms",
            ]
        ].to_parquet(
            DATA / f"eligibility_{window}.parquet", index=False, compression="zstd"
        )
        flow = (
            df.groupby(
                ["region", "tier", "analysis_eligible", "first_status"], observed=True
            )
            .size()
            .rename("n_matches")
            .reset_index()
        )
        flow["window"] = window
        audit.append(flow)
        stages = meta[["match_id", "region", "tier", "patch", "status"]].merge(
            df[
                [
                    "match_id",
                    "window_valid",
                    "analysis_eligible",
                    "first_post_neutral_blue",
                ]
            ],
            on="match_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        stages["canonical"] = 1
        stages["downloaded"] = stages.status.eq("downloaded").astype(int)
        stages["parsed"] = stages["_merge"].eq("both").astype(int)
        stages["identity_verified"] = (
            stages.parsed.eq(1) | stages.match_id.isin(identity_pass_errors)
        ).astype(int)
        stages["valid_window"] = stages.window_valid.eq(True).astype(int)
        stages["eligible"] = stages.analysis_eligible.eq(True).astype(int)
        stages["usable_primary_outcome"] = (
            stages.analysis_eligible.eq(True) & stages.first_post_neutral_blue.notna()
        ).astype(int)
        stage_names = [
            "canonical",
            "downloaded",
            "identity_verified",
            "parsed",
            "valid_window",
            "eligible",
            "usable_primary_outcome",
        ]
        for group, label in [
            (["region", "tier"], "region_tier"),
            (["region", "patch"], "region_patch"),
            (["region", "tier", "patch"], "region_tier_patch"),
        ]:
            stages.groupby(group, observed=True)[
                stage_names
            ].sum().reset_index().to_csv(
                TABLES / f"sample_flow_{label}_{window}.csv", index=False
            )
        del stages
        good = df[df.analysis_eligible].copy()
        for col in good.select_dtypes("float64").columns:
            if col not in [
                "game_duration",
                "state_frame_ms",
                "final_frame_ms",
                "first_post_neutral_ms",
            ]:
                good[col] = good[col].astype("float32")
        good.to_parquet(
            DATA / f"timeline_features_{window}.parquet",
            index=False,
            compression="zstd",
        )
        if window == "0_10":
            frame_audit = good[["match_id", "final_frame_ms"]].merge(
                meta[["match_id", "frames_count"]], on="match_id", validate="one_to_one"
            )
            frame_audit["count_minus_nominal_minutes"] = frame_audit.frames_count - (
                np.floor(frame_audit.final_frame_ms / 60000) + 1
            )
            frame_audit.groupby(
                "count_minus_nominal_minutes", dropna=False
            ).size().rename("n_matches").reset_index().to_csv(
                QA / "indexed_frame_count_duration_check.csv", index=False
            )
            suspicious = frame_audit[frame_audit.count_minus_nominal_minutes.lt(-1)]
            suspicious.to_parquet(
                DATA / "suspicious_frame_count_matches.parquet", index=False
            )
            write_json(
                QA / "frame_completeness_scope.json",
                {
                    "n_valid_matches": len(frame_audit),
                    "indexed_count_deficit_over_one_frame": len(suspicious),
                    "verified_in_payload": "ordered unique timestamps, complete ten-person landmark and final frames, terminal winner; gzip integrity",
                    "additional_count_check": "download-index frame count compared with elapsed nominal minutes; allows one-frame drift; not a proof of every intermediate frame state",
                },
            )
            if len(suspicious):
                raise RuntimeError(
                    "Inspect suspicious frame-count matches before model fitting"
                )
            del frame_audit, suspicious
            raw = []
            for (region, tier), g in good.groupby(["region", "tier"], observed=True):
                for component in [
                    "valid_wards_placed",
                    "valid_ward_kills",
                    "control_ward_purchases",
                    "control_ward_undo_net",
                ]:
                    values = np.r_[g["blue_" + component], g["red_" + component]]
                    raw.append(
                        {
                            "region": region,
                            "tier": tier,
                            "component": component,
                            "n_teams": len(values),
                            "mean_count": float(values.mean()),
                            "sd_count": float(values.std()),
                            "zero_fraction": float((values == 0).mean()),
                        }
                    )
            pd.DataFrame(raw).to_csv(
                TABLES / "early_components_natural_units.csv", index=False
            )
        good.groupby(["region", "tier"], observed=True).agg(
            n_matches=("match_id", "size"),
            n_primary_outcome=("first_post_neutral_blue", "count"),
            first_game=("game_date", "min"),
            last_game=("game_date", "max"),
            mean_state_lag_ms=("state_lag_ms", "mean"),
        ).reset_index().to_csv(
            TABLES / f"timeline_valid_coverage_{window}.csv", index=False
        )
        log(
            f"{window}: {len(good):,} valid matches, {good.first_post_neutral_blue.notna().sum():,} primary outcomes"
        )
        del good, df
    pd.concat(audit).to_csv(TABLES / "timeline_sample_flow.csv", index=False)
    people = pd.concat(
        (pd.read_parquet(p) for p in frames("timeline", ".people")), ignore_index=True
    )
    assert people.match_id.is_unique
    # Random-order greedy selection of matches sharing no participant (all ten IDs).
    people = people.sort_values("match_id").reset_index(drop=True)
    order = np.random.default_rng(child_seed("all_participant_disjoint")).permutation(
        len(people)
    )
    values = people[[f"person_{i}" for i in range(1, 11)]].to_numpy(dtype=np.uint64)
    used = set()
    chosen = []
    for ix in order:
        ids = values[ix].tolist()
        if not any(x in used for x in ids):
            used.update(ids)
            chosen.append(ix)
    chosen_df = people.loc[chosen, ["match_id"]]
    chosen_df.to_parquet(DATA / "all_participant_disjoint_matches.parquet", index=False)
    write_json(
        QA / "all_participant_disjoint_selection.json",
        {
            "candidate_matches": len(people),
            "selected_matches": len(chosen),
            "selected_unique_hashed_people": len(used),
            "ten_unique_people_per_match": len(used) == len(chosen) * 10,
            "seed": child_seed("all_participant_disjoint"),
            "selection": "one random-order greedy vertex-disjoint match set; different estimand, sensitivity only",
            "link_hash": "64-bit BLAKE2b with task prefix; conservative false overlap possible at negligible collision probability",
        },
    )
    log(f"All-participant-disjoint sensitivity: {len(chosen):,} matches")
    # Exact parser parity on a deterministic overlap sample, with deliberate new outcomes separated.
    cols = [
        "match_id",
        "window",
        "blue_valid_wards_placed",
        "red_valid_wards_placed",
        "blue_valid_ward_kills",
        "red_valid_ward_kills",
        "blue_control_ward_purchases",
        "red_control_ward_purchases",
        "blue_gold_at_window",
        "red_gold_at_window",
    ]
    old = pd.read_parquet(
        ROOT
        / "outputs/study1_information_support_validation/data/study1_match_difference_table_scored.parquet",
        columns=cols,
    )
    old = old[old.window.eq("0_10")].sample(n=10000, random_state=SEED)
    new = pd.read_parquet(
        DATA / "timeline_features_strict_0_10.parquet",
        columns=[c for c in cols if c != "window"],
    )
    check = old.merge(
        new, on="match_id", suffixes=("_old", "_new"), validate="one_to_one"
    )
    parity = []
    for col in cols[2:]:
        delta = check[col + "_new"] - check[col + "_old"]
        parity.append(
            {
                "variable": col,
                "overlap_matches": len(check),
                "different_matches": int((delta.abs() > 1e-5).sum()),
                "max_abs_difference": float(delta.abs().max()),
            }
        )
    pd.DataFrame(parity).to_csv(QA / "old_new_raw_feature_parity.csv", index=False)
    for kind in ["timeline", "matchdto"]:
        err = pd.concat(
            (pd.read_parquet(p) for p in frames(kind, ".errors")), ignore_index=True
        )
        err.to_parquet(DATA / f"{kind}_parse_errors.parquet", index=False)
        err.groupby("reason").size().rename("n_matches").reset_index().to_csv(
            TABLES / f"{kind}_parse_error_summary.csv", index=False
        )
        counts = collections.Counter()
        base = (DATA / "aligned") if kind == "timeline" else DATA
        for p in (base / (kind + "_shards")).glob("*.complete.json"):
            for x in json.loads(p.read_text(encoding="utf-8")).get("event_audit", []):
                counts[(x["type"], x["value"])] += x["n"]
        pd.DataFrame(
            [{"type": k[0], "value": k[1], "n_events": v} for k, v in counts.items()]
        ).to_csv(TABLES / f"{kind}_event_semantics.csv", index=False)
    log("Assembly finished")


if __name__ == "__main__":
    main()
