"""Read immutable timeline payloads and reconcile minute totals to frozen features."""

from pathlib import Path
import os, sys, json, hashlib, time, argparse, concurrent.futures as cf

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent / "03_reproducibility"
DATA = BASE / "data"
sys.path.insert(0, str(OUT / "runtime_deps"))
sys.path.insert(0, str(BASE / "runtime_deps"))
sys.dont_write_bytecode = True
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
import numpy as np
import pandas as pd
import orjson

try:
    from isal import igzip as gzip

    GZIP_BACKEND = "isal.igzip"
except ImportError:
    import gzip

    GZIP_BACKEND = "stdlib.gzip"
VALID = {"YELLOW_TRINKET", "CONTROL_WARD", "SIGHT_WARD", "BLUE_TRINKET"}
COMPS = ["valid_wards_placed", "valid_ward_kills", "control_ward_purchases"]
VERSION = "minute-v1-20260919"


def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def side(pid):
    return (
        0
        if isinstance(pid, int) and 1 <= pid <= 5
        else 1 if isinstance(pid, int) and 6 <= pid <= 10 else None
    )


def minute_counts(obj, expected, cutoff, fast=True):
    tl = obj.get("timeline", obj)
    if (
        obj.get("match_id", expected) != expected
        or tl.get("metadata", {}).get("matchId") != expected
    ):
        raise ValueError("match_id_mismatch")
    info = tl.get("info", {})
    parts = info.get("participants", [])
    if sorted(p.get("participantId", 0) for p in parts) != list(range(1, 11)):
        raise ValueError("participant_ids")
    if len(set(p.get("puuid", "") for p in parts)) != 10 or any(
        not p.get("puuid") for p in parts
    ):
        raise ValueError("participant_puuids")
    counts = np.zeros((2, 4, 10), dtype=np.int32)
    for frame in info.get("frames", []):
        if fast and frame.get("timestamp", 0) > cutoff + 65000:
            continue
        for e in frame.get("events", []):
            kind = e.get("type")
            if kind not in ("WARD_PLACED", "WARD_KILL", "ITEM_PURCHASED", "ITEM_UNDO"):
                continue
            ts = e.get("timestamp")
            if not isinstance(ts, (float, int)) or not 0 <= ts <= cutoff:
                continue
            k = min(9, max(0, int((ts - 1) // 60000)))
            if kind == "WARD_PLACED" and e.get("wardType") in VALID:
                s = side(e.get("creatorId"))
                if s is not None:
                    counts[s, 0, k] += 1
            elif kind == "WARD_KILL" and e.get("wardType") in VALID:
                s = side(e.get("killerId"))
                if s is not None:
                    counts[s, 1, k] += 1
            elif kind == "ITEM_PURCHASED" and e.get("itemId") == 2055:
                s = side(e.get("participantId"))
                if s is not None:
                    counts[s, 2, k] += 1
                    counts[s, 3, k] += 1
            elif kind == "ITEM_UNDO":
                s = side(e.get("participantId"))
                if s is not None:
                    counts[s, 3, k] += int(e.get("afterId") == 2055) - int(
                        e.get("beforeId") == 2055
                    )
    return counts


def tests():
    parts = [{"participantId": i, "puuid": str(i)} for i in range(1, 11)]
    ev = [
        {"type": "WARD_PLACED", "timestamp": t, "creatorId": 1, "wardType": w}
        for t, w in [
            (0, "YELLOW_TRINKET"),
            (60000, "CONTROL_WARD"),
            (60001, "SIGHT_WARD"),
            (600239, "BLUE_TRINKET"),
            (600240, "SIGHT_WARD"),
            (1, "UNDEFINED"),
        ]
    ]
    ev += [
        {
            "type": "ITEM_PURCHASED",
            "timestamp": 120000,
            "participantId": 6,
            "itemId": 2055,
        },
        {
            "type": "ITEM_UNDO",
            "timestamp": 120001,
            "participantId": 6,
            "beforeId": 2055,
            "afterId": 0,
        },
    ]
    o = {
        "metadata": {"matchId": "TEST"},
        "info": {
            "participants": parts,
            "frames": [{"timestamp": 600239, "events": ev}],
        },
    }
    a = minute_counts(o, "TEST", 600239)
    assert (
        a[0, 0, 0] == 2 and a[0, 0, 1] == 1 and a[0, 0, 9] == 1 and a[0, 0].sum() == 4
    )
    assert a[1, 2].sum() == 1 and a[1, 3].sum() == 0 and a[1, 3, 2] == -1
    try:
        minute_counts(o, "WRONG", 600239)
    except ValueError as e:
        assert str(e) == "match_id_mismatch"
    else:
        raise AssertionError("wrong ID accepted")
    dump(
        OUT / "qa/parser_tests.json",
        {
            "boundary_0_60000_60001_and_actual_landmark": "passed",
            "undefined_exclusion": "passed",
            "purchase_undo": "passed",
            "control_ward_count_not_duplicated": "passed",
            "match_identity": "passed",
        },
    )


def worker(job):
    path, targets, meta, old_digest, eligible, signature = job
    path = Path(path)
    stem = path.parent.name + "_" + path.name.replace(".jsonl.gz", "")
    dest = OUT / "data/minute_shards" / f"{stem}.parquet"
    mark = dest.with_suffix(".complete.json")
    if mark.exists():
        prior = json.loads(mark.read_text(encoding="utf-8"))
        if (
            prior["signature"] == signature
            and dest.exists()
            and sha(dest) == prior["output_sha256"]
        ):
            return dict(prior, cached=True)
        raise RuntimeError("stale cache: " + stem)
    if (path.stat().st_size, path.stat().st_mtime_ns) != tuple(meta):
        raise RuntimeError("changed source: " + str(path))
    mapping = {int(line): mid for line, mid in targets}
    positions = {mid: i for i, mid in enumerate(eligible)}
    ids = list(eligible)
    matrix = np.zeros((len(ids), 80), dtype=np.int16)
    found = set()
    payload_hash = hashlib.sha256()
    start = time.monotonic()
    fallback = 0
    with gzip.open(path, "rb") as f:
        for line_no, line in enumerate(f, 1):
            if line_no not in mapping:
                continue
            mid = mapping[line_no]
            payload_hash.update(line)
            if mid not in eligible:
                continue
            cutoff, expected = eligible[mid]
            obj = orjson.loads(line)
            a = minute_counts(obj, mid, cutoff)
            totals = a.sum(axis=2).ravel()
            if not np.array_equal(totals, expected):
                a = minute_counts(obj, mid, cutoff, False)
                totals = a.sum(axis=2).ravel()
                fallback += 1
            if not np.array_equal(totals, expected):
                raise RuntimeError(
                    f"counts disagree {mid}: {totals.tolist()} vs {expected}"
                )
            if np.max(np.abs(a)) > 32767:
                raise RuntimeError("count exceeds int16")
            matrix[positions[mid]] = a.ravel()
            found.add(mid)
    if found != set(ids):
        raise RuntimeError("missing eligible matches " + stem)
    if payload_hash.hexdigest() != old_digest:
        raise RuntimeError("historical target payload hash changed " + stem)
    if (path.stat().st_size, path.stat().st_mtime_ns) != tuple(meta):
        raise RuntimeError("source changed during read")
    columns = [
        f"{s}_{c}_m{k:02d}"
        for s in ["blue", "red"]
        for c in [*COMPS, "control_ward_undo_net"]
        for k in range(1, 11)
    ]
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "match_id", ids)
    df.to_parquet(dest, index=False, compression="zstd")
    result = dict(
        signature=signature,
        path=str(path),
        n_matches=len(ids),
        n_indexed_targets=len(targets),
        seconds=time.monotonic() - start,
        target_payload_sha256=payload_hash.hexdigest(),
        historical_payload_identity=True,
        all_frozen_component_totals_equal=True,
        full_event_scan_fallback_matches=fallback,
        gzip_backend=GZIP_BACKEND,
        output_sha256=sha(dest),
    )
    dump(mark, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    for p in [
        "data/minute_shards",
        "source_data",
        "tables",
        "figures",
        "manuscript",
        "qa",
        "logs",
        "code",
    ]:
        (OUT / p).mkdir(parents=True, exist_ok=True)
    tests()
    cols = ["match_id", "analysis_cutoff_ms"] + [
        f"{s}_{c}" for s in ["blue", "red"] for c in [*COMPS, "control_ward_undo_net"]
    ]
    frozen = pd.read_parquet(DATA / "timeline_features_0_10.parquet", columns=cols)
    assert len(frozen) == 2223546 and frozen.match_id.is_unique
    eligible = {
        v[0]: (int(v[1]), tuple(int(x) for x in v[2:]))
        for v in frozen.itertuples(index=False, name=None)
    }
    targets = pd.read_parquet(DATA / "aligned/timeline_targets_all.parquet")
    targets = targets[targets.status.eq("downloaded")]
    inventory = {
        x["path"]: x
        for x in json.loads(
            (DATA / "aligned/timeline_source_file_inventory.json").read_text(
                encoding="utf-8"
            )
        )
    }
    signature_base = (
        VERSION + sha(Path(__file__)) + sha(DATA / "timeline_features_0_10.parquet")
    )
    jobs = []
    for path, g in targets.groupby("shard_path"):
        pp = Path(path)
        stem = pp.parent.name + "_" + pp.name.replace(".jsonl.gz", "")
        old = json.loads(
            (DATA / "aligned/timeline_shards" / f"{stem}.complete.json").read_text(
                encoding="utf-8"
            )
        )
        t = list(zip(g.shard_line.astype(int), g.match_id.astype(str)))
        e = {mid: eligible[mid] for _, mid in t if mid in eligible}
        inv = inventory[path]
        signature = hashlib.sha256(
            (signature_base + old["target_payload_sha256"] + json.dumps(t)).encode()
        ).hexdigest()
        jobs.append(
            (
                path,
                t,
                (inv["bytes"], inv["mtime_ns"]),
                old["target_payload_sha256"],
                e,
                signature,
            )
        )
    jobs.sort(key=lambda j: (Path(j[0]).name, Path(j[0]).parent.name))
    if args.smoke:
        jobs = jobs[: args.smoke]
    assert args.smoke or sum(len(j[4]) for j in jobs) == len(frozen)
    dump(
        OUT / "qa/raw_input_manifest.json",
        {
            "frozen_features_sha256": sha(DATA / "timeline_features_0_10.parquet"),
            "targets_sha256": sha(DATA / "aligned/timeline_targets_all.parquet"),
            "code_sha256": sha(Path(__file__)),
            "seed": 191026,
            "n_shards": len(jobs),
            "expected_matches": sum(len(j[4]) for j in jobs),
            "workers": args.workers,
            "gzip_backend": GZIP_BACKEND,
        },
    )
    log(
        f"Parse {len(jobs)} shards / {sum(len(j[4]) for j in jobs):,} eligible matches; {args.workers} workers; {GZIP_BACKEND}"
    )
    start = time.monotonic()
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        pending = {ex.submit(worker, j): j[0] for j in jobs}
        for future in cf.as_completed(pending):
            result = future.result()
            results.append(result)
            done = sum(r["n_matches"] for r in results)
            progress = dict(
                shards_done=len(results),
                shards_total=len(jobs),
                matches_done=done,
                elapsed_seconds=time.monotonic() - start,
            )
            dump(OUT / "logs/minute_progress.json", progress)
            log(
                f"{len(results)}/{len(jobs)} shards; {done:,} matches; {(time.monotonic()-start)/60:.1f} min"
            )
    dump(OUT / "qa/minute_parse_summary.json", results)
    log("Minute parsing complete")


if __name__ == "__main__":
    main()
