"""Parse participant-level early-game timelines for fixed-sample and held-out matches in one pass.

Minute bins, ward validity and the near-10-minute cutoff follow the frozen team-level parsers
(parse_minutes.minute_counts and parse_aligned.aligned_parser), so participant sums can be
reconciled exactly with the published team-level minute data.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (CONTROL_WARD_ITEM, DATA, LOGS, REPRO, ROOT, TABLES, VALID_WARDS, WINDOW_MS, ensure_dirs, log,
                    sha256, write_json)

sys.path.insert(0, str(REPRO / "code"))
sys.path.insert(0, str(REPRO / "runtime_deps"))
sys.path.insert(0, str(ROOT / "minute_timeline" / "runtime_deps"))
import orjson  # noqa: E402

try:
    from isal import igzip as gzip  # noqa: E402
except ImportError:
    import gzip  # noqa: E402

VERSION = "ijim-participant-timeline-v1"
METRICS = ["valid_wards_placed", "valid_ward_kills", "control_ward_purchases", "control_ward_undo_net", "control_wards_placed"]
STATE = ["x", "y", "total_gold", "xp", "level", "minions_killed", "jungle_minions_killed"]
COMBAT = ["kills_pre", "deaths_pre", "assists_pre"]
MINUTE_COLUMNS = [f"{m}_m{k:02d}" for m in METRICS for k in range(1, 11)]
KINDS = {"WARD_PLACED", "WARD_KILL", "ITEM_PURCHASED", "ITEM_UNDO", "CHAMPION_KILL"}
PIDS = set(range(1, 11))
OUT = DATA / "timeline_participants"
OLD_SHARDS = REPRO / "data" / "aligned" / "timeline_shards"
CHECK_EVERY = 100


def landmark(frames) -> tuple[int, bool]:
    stamps = [f.get("timestamp", -10**9) for f in frames]
    nearest = min(stamps, key=lambda t: abs(t - WINDOW_MS))
    return int(nearest), abs(nearest - WINDOW_MS) <= 1000


def parse_participants(obj, expected: str):
    tl = obj.get("timeline", obj)
    if obj.get("match_id", expected) != expected or tl.get("metadata", {}).get("matchId") != expected:
        raise ValueError("match_id_mismatch")
    info = tl.get("info", {})
    parts = info.get("participants", [])
    if sorted(p.get("participantId", 0) for p in parts) != list(range(1, 11)):
        raise ValueError("invalid_participant_ids")
    puuids = {p["participantId"]: p.get("puuid") for p in parts}
    if len(set(puuids.values())) != 10 or not all(puuids.values()):
        raise ValueError("invalid_participant_puuids")
    frames = info.get("frames", [])
    stamps = [f.get("timestamp", -1) for f in frames]
    if len(frames) < 2 or stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
        raise ValueError("invalid_frame_order")
    cutoff, valid = landmark(frames)
    counts = np.zeros((10, len(METRICS), 10), dtype=np.int32)
    combat = np.zeros((10, len(COMBAT)), dtype=np.int32)
    state = np.full((10, len(STATE)), np.nan)
    if valid:
        for frame in frames:
            for e in frame.get("events", ()):
                kind = e.get("type")
                if kind not in KINDS:
                    continue
                ts = e.get("timestamp")
                if not isinstance(ts, (int, float)) or not 0 <= ts <= cutoff:
                    continue
                k = min(9, max(0, int((ts - 1) // 60000)))
                if kind == "WARD_PLACED":
                    pid = e.get("creatorId")
                    if pid in PIDS and e.get("wardType") in VALID_WARDS:
                        counts[pid - 1, 0, k] += 1
                        if e.get("wardType") == "CONTROL_WARD":
                            counts[pid - 1, 4, k] += 1
                elif kind == "WARD_KILL":
                    pid = e.get("killerId")
                    if pid in PIDS and e.get("wardType") in VALID_WARDS:
                        counts[pid - 1, 1, k] += 1
                elif kind == "ITEM_PURCHASED":
                    pid = e.get("participantId")
                    if pid in PIDS and e.get("itemId") == CONTROL_WARD_ITEM:
                        counts[pid - 1, 2, k] += 1
                        counts[pid - 1, 3, k] += 1
                elif kind == "ITEM_UNDO":
                    pid = e.get("participantId")
                    delta = int(e.get("afterId") == CONTROL_WARD_ITEM) - int(e.get("beforeId") == CONTROL_WARD_ITEM)
                    if pid in PIDS and delta:
                        counts[pid - 1, 3, k] += delta
                else:
                    killer, victim = e.get("killerId"), e.get("victimId")
                    if killer in PIDS:
                        combat[killer - 1, 0] += 1
                    if victim in PIDS:
                        combat[victim - 1, 1] += 1
                    for pid in e.get("assistingParticipantIds") or ():
                        if pid in PIDS:
                            combat[pid - 1, 2] += 1
        chosen = frames[stamps.index(cutoff)].get("participantFrames", {})
        for pid in range(1, 11):
            pf = chosen.get(str(pid)) or {}
            pos = pf.get("position") or {}
            state[pid - 1] = [pos.get("x", np.nan), pos.get("y", np.nan), pf.get("totalGold", np.nan), pf.get("xp", np.nan),
                              pf.get("level", np.nan), pf.get("minionsKilled", np.nan), pf.get("jungleMinionsKilled", np.nan)]
    return cutoff, valid, puuids, counts, combat, state


class ShardRows:
    """Column buffers for one sample within one shard."""

    def __init__(self) -> None:
        self.match_ids, self.puuids, self.cutoffs, self.valid = [], [], [], []
        self.state, self.combat, self.minutes = [], [], []

    def add(self, match_id: str, parsed) -> None:
        cutoff, valid, puuids, counts, combat, state = parsed
        self.match_ids.extend([match_id] * 10)
        self.puuids.extend(puuids[p] for p in range(1, 11))
        self.cutoffs.extend([cutoff] * 10)
        self.valid.extend([valid] * 10)
        self.state.append(state)
        self.combat.append(combat)
        self.minutes.append(counts.reshape(10, -1))

    def __len__(self) -> int:
        return len(self.match_ids) // 10

    def frame(self) -> pd.DataFrame:
        n = len(self.match_ids)
        pid = np.tile(np.arange(1, 11, dtype=np.int8), n // 10)
        frame = pd.DataFrame({
            "match_id": self.match_ids,
            "participant_id": pid,
            "team_id": np.where(pid <= 5, 100, 200).astype(np.int16),
            "puuid": self.puuids,
            "cutoff_ms": np.asarray(self.cutoffs, dtype=np.int32),
            "window_valid": np.asarray(self.valid, dtype=bool),
        })
        state = np.vstack(self.state).astype(np.float32)
        combat = np.vstack(self.combat).astype(np.int16)
        minutes = np.vstack(self.minutes).astype(np.int16)
        extra = {name: state[:, j] for j, name in enumerate(STATE)}
        extra.update({name: combat[:, j] for j, name in enumerate(COMBAT)})
        extra.update({name: minutes[:, j] for j, name in enumerate(MINUTE_COLUMNS)})
        return pd.concat([frame, pd.DataFrame(extra)], axis=1)


def team_check_wanted(match_id: str) -> bool:
    return int.from_bytes(hashlib.blake2b(match_id.encode(), digest_size=4).digest(), "little") % CHECK_EVERY == 0


def worker(job) -> dict:
    import parse_aligned  # frozen team-level parser, loaded lazily inside each process

    path, targets, old_hash, signature = job
    path = Path(path)
    stem = path.parent.name + "_" + path.name.replace(".jsonl.gz", "")
    mark = OUT / "_complete" / f"{stem}.json"
    if mark.exists():
        prior = json.loads(mark.read_text(encoding="utf-8"))
        if prior.get("signature") == signature:
            return {**prior, "cached": True}
    mapping = {int(line): (mid, sample) for line, mid, sample in targets}
    stat = path.stat()
    start = time.monotonic()
    outputs = {"fixed": ShardRows(), "holdout": ShardRows()}
    team_rows = {"fixed": [], "holdout": []}
    errors = []
    fixed_hash = hashlib.sha256()
    found = set()
    with gzip.open(path, "rb") as handle:
        for line_no, line in enumerate(handle, 1):
            hit = mapping.get(line_no)
            if hit is None:
                continue
            mid, sample = hit
            found.add(mid)
            if sample == "fixed":
                fixed_hash.update(line)
            try:
                obj = orjson.loads(line)
            except Exception as exc:
                errors.append({"match_id": mid, "sample": sample, "stage": "json", "reason": str(exc)})
                continue
            try:
                outputs[sample].add(mid, parse_participants(obj, mid))
            except Exception as exc:
                errors.append({"match_id": mid, "sample": sample, "stage": "participant", "reason": str(exc)})
            if sample == "holdout" or team_check_wanted(mid):
                try:
                    rows, _people, _audit = parse_aligned.aligned_parser(obj, mid)
                    team_rows[sample].extend(rows)
                except Exception as exc:
                    errors.append({"match_id": mid, "sample": sample, "stage": "team", "reason": str(exc)})
    for line_no, (mid, sample) in mapping.items():
        if mid not in found:
            errors.append({"match_id": mid, "sample": sample, "stage": "index", "reason": "indexed_line_absent"})
    if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(f"source changed during read: {path}")
    written = {}
    for sample, buffer in outputs.items():
        if len(buffer):
            buffer.frame().to_parquet(OUT / sample / f"{stem}.parquet", index=False, compression="zstd")
            written[sample] = len(buffer)
        if team_rows[sample]:
            folder = DATA / ("timeline_team_holdout" if sample == "holdout" else "timeline_team_fixed_check")
            pd.DataFrame(team_rows[sample]).to_parquet(folder / f"{stem}.parquet", index=False, compression="zstd")
    if errors:
        pd.DataFrame(errors).to_parquet(OUT / "_errors" / f"{stem}.parquet", index=False)
    result = {
        "signature": signature, "path": str(path), "targets": len(mapping), "parsed": written, "errors": len(errors),
        "fixed_payload_sha256": fixed_hash.hexdigest(), "historical_fixed_payload_sha256": old_hash,
        "historical_payload_identity": (old_hash == fixed_hash.hexdigest()) if old_hash else None,
        "seconds": time.monotonic() - start,
    }
    write_json(mark, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    ensure_dirs()
    for folder in [OUT / "fixed", OUT / "holdout", OUT / "_complete", OUT / "_errors",
                   DATA / "timeline_team_holdout", DATA / "timeline_team_fixed_check"]:
        folder.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(DATA / "timeline_targets_ijim.parquet")
    code_hash = sha256(Path(__file__))
    jobs = []
    for path, group in targets.groupby("shard_path"):
        pp = Path(path)
        stem = pp.parent.name + "_" + pp.name.replace(".jsonl.gz", "")
        old_mark = OLD_SHARDS / f"{stem}.complete.json"
        old_hash = json.loads(old_mark.read_text(encoding="utf-8"))["target_payload_sha256"] if old_mark.exists() else None
        rows = list(zip(group["shard_line"].astype(int), group["match_id"].astype(str), group["sample"].astype(str)))
        signature = hashlib.sha256((VERSION + code_hash + json.dumps(rows)).encode()).hexdigest()
        jobs.append((path, rows, old_hash, signature))
    jobs.sort(key=lambda j: (Path(j[0]).name, Path(j[0]).parent.name))
    if args.smoke:
        jobs = jobs[: args.smoke]
    log(f"{len(jobs)} shards; {sum(len(j[1]) for j in jobs):,} targets; {args.workers} workers")
    start = time.monotonic()
    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job[0] for job in jobs}
        for future in cf.as_completed(futures):
            results.append(future.result())
            parsed = sum(sum(r["parsed"].values()) for r in results)
            progress = {"shards_done": len(results), "shards_total": len(jobs), "matches_parsed": parsed,
                        "errors": sum(r["errors"] for r in results), "elapsed_seconds": time.monotonic() - start}
            write_json(LOGS / "timeline_participants_progress.json", progress)
            if len(results) % 10 == 0 or len(results) == len(jobs):
                log(f"{len(results)}/{len(jobs)} shards; {parsed:,} matches; {progress['errors']} errors")
    write_json(TABLES / "qa" / ("timeline_participants_smoke.json" if args.smoke else "timeline_participants_parse.json"), {
        "version": VERSION, "code_sha256": code_hash, "shards": len(results),
        "parsed_by_sample": {s: sum(r["parsed"].get(s, 0) for r in results) for s in ["fixed", "holdout"]},
        "errors": sum(r["errors"] for r in results),
        "historical_payload_identity_false": [r["path"] for r in results if r["historical_payload_identity"] is False],
        "seconds": time.monotonic() - start,
    })
    log("participant timeline parsing complete")


if __name__ == "__main__":
    main()
