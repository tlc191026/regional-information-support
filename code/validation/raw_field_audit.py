"""Audit recorded signal fields and participant overlap in the archived match payloads."""

from common import *

sys.path.insert(0, str(W / "vendor"))
import orjson
import numpy as np
import pandas as pd
import gzip
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

FIELDS = [
    "enemyMissingPings",
    "enemyVisionPings",
    "commandPings",
    "needVisionPings",
    "assistMePings",
    "onMyWayPings",
    "pushPings",
    "allInPings",
    "getBackPings",
]


def worker(arg):
    path, items, num = arg
    targets = {str(x[0]): x[1:] for x in items}
    out = W / "data/raw_audit_shards"
    out.mkdir(exist_ok=True)
    groups = {}
    records = []
    errors = []
    seen = set()
    with gzip.open(path, "rb") as handle:
        for line_no, line in enumerate(handle, 1):
            j = orjson.loads(line)
            mid = str(j.get("match_id", ""))
            if mid not in targets:
                continue
            region, patch, expected = targets[mid]
            if mid in seen:
                errors.append([mid, "duplicate payload"])
                continue
            seen.add(mid)
            payload = j.get("match", {})
            meta = payload.get("metadata", {})
            info = payload.get("info", {})
            parts = info.get("participants", [])
            if meta.get("matchId") != mid or len(parts) != 10:
                errors.append([mid, "identifier or participant count"])
                continue
            if int(expected) != line_no:
                errors.append([mid, "shard line mismatch"])
            rawpatch = ".".join(str(info.get("gameVersion", "")).split(".")[:2])
            if rawpatch != patch:
                errors.append([mid, "patch mismatch"])
            key = (region, patch)
            if key not in groups:
                groups[key] = np.zeros((len(FIELDS), 5), dtype=np.int64)
            counts = groups[key]
            for p in parts:
                for k, field in enumerate(FIELDS):
                    counts[k, 0] += 1
                    if field not in p:
                        counts[k, 1] += 1
                    elif p[field] is None:
                        counts[k, 2] += 1
                    elif (
                        not isinstance(p[field], (int, float))
                        or isinstance(p[field], bool)
                        or p[field] < 0
                    ):
                        counts[k, 3] += 1
                    elif p[field] == 0:
                        counts[k, 4] += 1
            tokens = [
                int.from_bytes(
                    hashlib.blake2b(p["puuid"].encode(), digest_size=8).digest(),
                    "little",
                )
                for p in parts
            ]
            records.append([mid, *tokens])
    rows = []
    for (r, p), v in groups.items():
        for k, field in enumerate(FIELDS):
            rows.append([r, p, field, *v[k].tolist()])
    pd.DataFrame(
        records, columns=["match_id", *[f"participant_{i}" for i in range(10)]]
    ).to_parquet(out / f"{num:04d}.parquet", index=False)
    return dict(
        path=path,
        n_targets=len(targets),
        n_seen=len(seen),
        missing=sorted(set(targets) - seen),
        errors=errors,
        counts=rows,
    )


if __name__ == "__main__":
    source = ROOT / "manuscript_20260915/03_reproducibility/data"
    t = pd.read_parquet(source / "matchdto_targets_all.parquet")
    good = t[t.status.eq("downloaded") & t.shard_path.notna()].copy()
    good.region = good.region.astype(str).replace({"NA1": "NA", "EUW1": "EUW"})
    args = (
        [
            (
                str(path),
                g[["match_id", "region", "patch", "shard_line"]]
                .astype({"patch": str})
                .values.tolist(),
                i,
            )
            for i, (path, g) in enumerate(good.groupby("shard_path", sort=True))
        ]
        if "--overlap-only" not in sys.argv
        else []
    )
    log(f"Audit {len(good):,} targets in {len(args)} shards")
    results = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(worker, a) for a in args]
        for future in as_completed(futs):
            results.append(future.result())
            if len(results) % 10 == 0:
                log(f"Parsed {len(results)}/{len(args)} shards")
    if results:
        rows = [row for r in results for row in r.pop("counts")]
        columns = [
            "region",
            "patch",
            "field",
            "n_participants",
            "field_absent",
            "null_value",
            "invalid_value",
            "observed_zero",
        ]
        a = (
            pd.DataFrame(rows, columns=columns)
            .groupby(["region", "patch", "field"], as_index=False, observed=True)
            .sum()
        )
        save("signal_field_availability.csv", a)
        dump("raw_field_audit.json", results)
    log("Compute all-participant overlap for the retained prediction samples")
    split = pd.read_parquet(source / "focal_player_match_split.parquet")
    elig = pd.read_parquet(
        source / "analysis_scored_0_10.parquet",
        columns=["match_id", "first_post_neutral_blue"],
    )
    split = split.merge(
        elig[elig.first_post_neutral_blue.notna()][["match_id"]], on="match_id"
    )
    participants = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((W / "data/raw_audit_shards").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    assert participants.match_id.is_unique
    participants = participants.astype(
        {f"participant_{i}": "UInt64" for i in range(10)}
    )
    g = split.merge(participants, on="match_id", how="left", validate="one_to_one")
    cols = [f"participant_{i}" for i in range(10)]
    covered = g[cols].notna().all(axis=1)
    missing = g.loc[~covered].groupby("focal_split", observed=True).size().to_dict()
    g = g[covered]
    train = g[g.focal_split.eq("train")][cols].to_numpy(np.uint64)
    test = g[g.focal_split.eq("test")][cols].to_numpy(np.uint64)
    tr = np.unique(train)
    te = np.unique(test)
    overlap = np.intersect1d(tr, te, assume_unique=True)
    member = np.isin(test, overlap)
    result = dict(
        n_train_matches=len(train),
        n_test_matches=len(test),
        n_unique_train_participants=len(tr),
        n_unique_test_participants=len(te),
        n_shared_participants=len(overlap),
        test_participant_overlap_percent=100 * len(overlap) / len(te),
        test_matches_with_any_overlap=int(member.any(axis=1).sum()),
        test_matches_with_any_overlap_percent=100 * member.any(axis=1).mean(),
        test_participant_slots_with_overlap_percent=100 * member.mean(),
        missing_payload_by_split=missing,
    )
    dump("prediction_participant_overlap.json", result)
    log(str(result))
    log("Raw-field audit complete")
