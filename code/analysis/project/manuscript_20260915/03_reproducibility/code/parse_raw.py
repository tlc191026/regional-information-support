"""Read indexed raw payloads once; validate identity and isolate all new caches."""

from settings import *
import argparse, gzip, time, collections, concurrent.futures as cf
import numpy as np
import pandas as pd

try:
    import orjson

    loads = orjson.loads
except ImportError:
    loads = json.loads

METRICS = [
    "valid_wards_placed",
    "valid_ward_kills",
    "control_ward_purchases",
    "control_wards_placed",
    "control_ward_undo_net",
    "raw_wards_placed",
    "undefined_wards_placed",
    "kills_pre",
    "early_assists_pre",
    "neutral_objectives_pre",
    "turret_plates_pre",
    "building_objectives_pre",
    "kills_post",
    "neutral_objectives_post",
    "building_objectives_post",
    "gold_at_window",
    "xp_at_window",
    "level_at_window",
    "damage_at_window",
    "gold_final",
    "damage_final",
]


def team_of(pid):
    if isinstance(pid, int):
        if 1 <= pid <= 5:
            return 100
        if 6 <= pid <= 10:
            return 200
    return None


def frame_sums(frame):
    pfs = frame.get("participantFrames", {})
    ids = [int(p.get("participantId", 0)) for p in pfs.values()]
    if sorted(ids) != list(range(1, 11)):
        raise ValueError("incomplete_participant_frame")
    result = {100: {}, 200: {}}
    for key, source in [("gold", "totalGold"), ("xp", "xp"), ("level", "level")]:
        for team in [100, 200]:
            vals = [
                p.get(source)
                for p in pfs.values()
                if team_of(p.get("participantId")) == team
            ]
            if len(vals) != 5 or any(v is None for v in vals):
                raise ValueError("missing_frame_state")
            result[team][key] = float(sum(vals))
    for team in [100, 200]:
        vals = [
            (p.get("damageStats") or {}).get("totalDamageDoneToChampions")
            for p in pfs.values()
            if team_of(p.get("participantId")) == team
        ]
        result[team]["damage"] = (
            float(sum(vals)) if all(v is not None for v in vals) else np.nan
        )
    return result


def parse_timeline(obj, expected):
    tl = obj.get("timeline", obj)
    if (
        obj.get("match_id", expected) != expected
        or tl.get("metadata", {}).get("matchId") != expected
    ):
        raise ValueError("match_id_mismatch")
    info = tl.get("info", {})
    parts = info.get("participants", [])
    if sorted(p.get("participantId", 0) for p in parts) != list(range(1, 11)):
        raise ValueError("invalid_participant_ids")
    if len(set(p.get("puuid", "") for p in parts)) != 10 or any(
        not p.get("puuid") for p in parts
    ):
        raise ValueError("invalid_participant_puuids")
    frames = info.get("frames", [])
    stamps = [f.get("timestamp", -1) for f in frames]
    if len(frames) < 2 or stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
        raise ValueError("invalid_frame_order")
    events = [e for f in frames for e in f.get("events", [])]
    ends = [e for e in events if e.get("type") == "GAME_END"]
    winners = {e.get("winningTeam") for e in ends}
    if len(winners) != 1 or not winners.issubset({100, 200}):
        raise ValueError("missing_or_conflicting_game_end")
    winner = next(iter(winners))
    final = frame_sums(frames[-1])
    people = {"match_id": expected, "blue_win": int(winner == 100)}
    for p in parts:
        people[f'person_{p["participantId"]}'] = int.from_bytes(
            hashlib.blake2b(
                ("lol20260915:" + p["puuid"]).encode(), digest_size=8
            ).digest(),
            "little",
        )
    audit = collections.Counter()
    for e in events:
        if e.get("type") == "ELITE_MONSTER_KILL":
            audit[("monster_type", str(e.get("monsterType", "MISSING")))] += 1
            audit[("monster_team", str(e.get("killerTeamId", "MISSING")))] += 1
        if e.get("type") in ("WARD_PLACED", "WARD_KILL"):
            audit[(e["type"], str(e.get("wardType", "MISSING")))] += 1
        if "PING" in str(e.get("type", "")):
            audit[("ping_event", e["type"])] += 1
    rows = []
    for window, limit in WINDOWS.items():
        before = [f for f in frames if f["timestamp"] <= limit]
        row = {
            "match_id": expected,
            "window": window,
            "blue_win": people["blue_win"],
            "window_valid": False,
            "window_reason": "",
        }
        if stamps[-1] <= limit or not before:
            row["window_reason"] = "no_post_window_observation"
            rows.append(row)
            continue
        chosen = before[-1]
        lag = limit - chosen["timestamp"]
        if lag > 65000:
            row["window_reason"] = "state_frame_too_old"
            rows.append(row)
            continue
        try:
            state = frame_sums(chosen)
        except ValueError as e:
            row["window_reason"] = str(e)
            rows.append(row)
            continue
        row.update(
            window_valid=True,
            state_frame_ms=chosen["timestamp"],
            state_lag_ms=lag,
            final_frame_ms=stamps[-1],
        )
        values = {100: {k: 0.0 for k in METRICS}, 200: {k: 0.0 for k in METRICS}}
        post_neutral = []
        for team in values:
            for k in ["gold", "xp", "level", "damage"]:
                values[team][k + "_at_window"] = state[team][k]
            for k in ["gold", "damage"]:
                values[team][k + "_final"] = final[team][k]
        for e in events:
            ts = e.get("timestamp")
            if not isinstance(ts, (int, float)) or ts < 0:
                audit[("invalid", "event_timestamp")] += 1
                continue
            pre = ts <= limit
            kind = e.get("type")
            if kind == "WARD_PLACED" and pre:
                team = team_of(e.get("creatorId"))
                ward = e.get("wardType")
                if team in values:
                    v = values[team]
                    v["raw_wards_placed"] += 1
                    if ward in VALID_WARDS:
                        v["valid_wards_placed"] += 1
                        v["control_wards_placed"] += int(ward == "CONTROL_WARD")
                    v["undefined_wards_placed"] += int(ward == "UNDEFINED")
            elif kind == "WARD_KILL" and pre and e.get("wardType") in VALID_WARDS:
                team = team_of(e.get("killerId"))
                if team in values:
                    values[team]["valid_ward_kills"] += 1
            elif kind == "ITEM_PURCHASED" and pre and e.get("itemId") == 2055:
                team = team_of(e.get("participantId"))
                if team in values:
                    values[team]["control_ward_purchases"] += 1
                    values[team]["control_ward_undo_net"] += 1
            elif kind == "ITEM_UNDO" and pre:
                team = team_of(e.get("participantId"))
                if team in values:
                    values[team]["control_ward_undo_net"] += int(
                        e.get("afterId") == 2055
                    ) - int(e.get("beforeId") == 2055)
            elif kind == "CHAMPION_KILL":
                team = team_of(e.get("killerId"))
                if team in values:
                    values[team]["kills_pre" if pre else "kills_post"] += 1
                    if pre:
                        for pid in e.get("assistingParticipantIds", []) or []:
                            at = team_of(pid)
                            if at in values:
                                values[at]["early_assists_pre"] += 1
            elif kind == "ELITE_MONSTER_KILL":
                team = e.get("killerTeamId")
                if not pre:
                    post_neutral.append((ts, team, e.get("monsterType", "MISSING")))
                if team in values:
                    values[team][
                        "neutral_objectives_pre" if pre else "neutral_objectives_post"
                    ] += 1
            elif kind in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
                destroyed = e.get("teamId")
                team = 300 - destroyed if destroyed in (100, 200) else None
                if team in values:
                    if kind == "BUILDING_KILL":
                        values[team][
                            (
                                "building_objectives_pre"
                                if pre
                                else "building_objectives_post"
                            )
                        ] += 1
                    elif pre:
                        values[team]["turret_plates_pre"] += 1
        row["first_status"] = "no_event"
        row["first_post_neutral_blue"] = np.nan
        row["first_legal_post_neutral_blue"] = np.nan
        row["first_post_neutral_ms"] = np.nan
        row["first_monster_type"] = "NONE"
        if post_neutral:
            earliest = min(x[0] for x in post_neutral)
            first = [x for x in post_neutral if x[0] == earliest]
            row["first_post_neutral_ms"] = earliest
            row["first_monster_type"] = "|".join(sorted(set(x[2] for x in first)))
            teams = set(x[1] for x in first)
            if not teams.issubset({100, 200}):
                row["first_status"] = "unassigned"
            elif len(teams) > 1:
                row["first_status"] = "tied_opposing_teams"
            else:
                row["first_status"] = "observed"
                row["first_post_neutral_blue"] = int(next(iter(teams)) == 100)
            legal = [x for x in post_neutral if x[1] in (100, 200)]
            if legal:
                first_legal = min(x[0] for x in legal)
                lt = {x[1] for x in legal if x[0] == first_legal}
                if len(lt) == 1:
                    row["first_legal_post_neutral_blue"] = int(next(iter(lt)) == 100)
        for team, prefix in [(100, "blue"), (200, "red")]:
            if values[team]["control_ward_undo_net"] < 0:
                audit[("invalid", "negative_undo_net")] += 1
            for k, v in values[team].items():
                row[prefix + "_" + k] = v
        rows.append(row)
    return rows, people, audit


def parse_matchdto(obj, expected):
    match = obj.get("match", obj.get("match_detail", obj))
    if (
        obj.get("match_id", expected) != expected
        or match.get("metadata", {}).get("matchId") != expected
    ):
        raise ValueError("match_id_mismatch")
    info = match.get("info", {})
    parts = info.get("participants", [])
    if len(parts) != 10 or collections.Counter(p.get("teamId") for p in parts) != {
        100: 5,
        200: 5,
    }:
        raise ValueError("invalid_matchdto_teams")
    if any(team_of(p.get("participantId")) != p.get("teamId") for p in parts):
        raise ValueError("participant_team_mapping_mismatch")
    teams = info.get("teams", [])
    wins = [t["teamId"] for t in teams if t.get("win")]
    if len(wins) != 1:
        raise ValueError("invalid_matchdto_win")
    version = str(info.get("gameVersion", ""))
    row = {
        "match_id": expected,
        "game_creation_ms": info.get("gameCreation"),
        "game_start_ms": info.get("gameStartTimestamp", info.get("gameCreation")),
        "game_end_ms": info.get("gameEndTimestamp"),
        "game_duration_raw": info.get("gameDuration"),
        "queue_id_raw": info.get("queueId"),
        "patch_raw": ".".join(version.split(".")[:2]),
        "game_version_raw": version,
        "blue_win_raw": int(wins[0] == 100),
        "surrender": any(p.get("gameEndedInSurrender", False) for p in parts),
        "early_surrender": any(
            p.get("gameEndedInEarlySurrender", False) for p in parts
        ),
    }
    return row


def parse_shard(job):
    kind, path, targets, expected_meta, limit = job
    path = Path(path)
    out = DATA / (kind + "_shards")
    out.mkdir(exist_ok=True)
    stem = path.parent.name + "_" + path.name.replace(".jsonl.gz", "")
    dest = out / (stem + ".parquet")
    mark = out / (stem + ".complete.json")
    sig = hashlib.sha256(
        (PARSER_VERSION + kind + str(expected_meta) + json.dumps(targets)).encode()
    ).hexdigest()
    if mark.exists():
        previous = loads(mark.read_bytes())
        if previous.get("signature") == sig and dest.exists():
            return {**previous, "cached": True}
        raise RuntimeError("Stale cache fingerprint: " + stem)
    mapping = {int(line): mid for line, mid in targets}
    if len(mapping) != len(targets):
        raise RuntimeError("duplicate_shard_lines:" + stem)
    start = time.monotonic()
    rows = []
    people = []
    errors = []
    audit = collections.Counter()
    found = set()
    payload_hash = hashlib.sha256()
    total = 0
    try:
        with gzip.open(path, "rb") as f:
            for line_no, line in enumerate(f, 1):
                total = line_no
                if line_no not in mapping:
                    continue
                expected = mapping[line_no]
                found.add(expected)
                payload_hash.update(line)
                try:
                    obj = loads(line)
                    if kind == "timeline":
                        r, p, a = parse_timeline(obj, expected)
                        rows.extend(r)
                        people.append(p)
                        audit.update(a)
                    else:
                        rows.append(parse_matchdto(obj, expected))
                except Exception as e:
                    errors.append(
                        {"match_id": expected, "reason": str(e), "line": line_no}
                    )
                if limit and len(found) >= limit:
                    break
    except Exception as e:
        errors.append(
            {
                "match_id": "SHARD",
                "reason": type(e).__name__ + ":" + str(e),
                "line": total,
            }
        )
        # A gzip integrity failure invalidates the entire shard, not just its tail.
        raise RuntimeError(f"Failed source integrity: {path}: {e}")
    if not limit:
        for mid in set(mapping.values()) - found:
            errors.append(
                {"match_id": mid, "reason": "indexed_line_absent", "line": -1}
            )
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != tuple(expected_meta):
        raise RuntimeError("Source changed during read: " + str(path))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(dest, index=False, compression="zstd")
    else:
        pd.DataFrame({"match_id": pd.Series(dtype="str")}).to_parquet(dest, index=False)
    if kind == "timeline" and people:
        pf = pd.DataFrame(people)
        for col in pf.columns:
            if col.startswith("person_"):
                pf[col] = pf[col].astype("uint64")
        pf.to_parquet(out / (stem + ".people.parquet"), index=False, compression="zstd")
    pd.DataFrame(errors, columns=["match_id", "reason", "line"]).to_parquet(
        out / (stem + ".errors.parquet"), index=False
    )
    result = {
        "signature": sig,
        "path": str(path),
        "kind": kind,
        "target_count": len(targets),
        "indexed_rows_found": len(found),
        "parsed_matches": len(people) if kind == "timeline" else len(rows),
        "output_rows": len(rows),
        "errors": len(errors),
        "lines_read": total,
        "target_payload_sha256": payload_hash.hexdigest(),
        "seconds": time.monotonic() - start,
        "event_audit": [
            {"type": k[0], "value": k[1], "n": v} for k, v in audit.items()
        ],
    }
    write_json(mark, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["timeline", "matchdto"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    targets = pd.read_parquet(DATA / f"{args.kind}_targets_all.parquet")
    targets = targets[targets.status.eq("downloaded")]
    inventories = {
        p["path"]: p
        for p in loads((DATA / f"{args.kind}_source_file_inventory.json").read_bytes())
    }
    jobs = []
    for path, g in targets.groupby("shard_path"):
        meta = inventories[path]
        jobs.append(
            (
                args.kind,
                path,
                list(zip(g.shard_line.astype(int), g.match_id.astype(str))),
                (meta["bytes"], meta["mtime_ns"]),
                0,
            )
        )
    jobs.sort(key=lambda j: (Path(j[1]).name, Path(j[1]).parent.name))
    if args.smoke:
        jobs = jobs[: args.smoke]
    log(
        f"{args.kind}: {len(jobs)} shards, {sum(len(j[2]) for j in jobs):,} indexed targets, {args.workers} workers"
    )
    results = []
    start = time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(parse_shard, j): j[1] for j in jobs}
        for future in cf.as_completed(futures):
            result = future.result()
            results.append(result)
            done = sum(x["parsed_matches"] for x in results)
            log(
                f'{args.kind}: {len(results)}/{len(jobs)} shards; {done:,} matches; {sum(x["errors"] for x in results)} errors; {(time.monotonic()-start)/60:.1f} min'
            )
            write_json(
                LOGS / f"{args.kind}_progress.json",
                {
                    "completed": len(results),
                    "total": len(jobs),
                    "parsed_matches": done,
                    "errors": sum(x["errors"] for x in results),
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
    write_json(QA / f"{args.kind}_parse_summary.json", results)
    log("Parsing complete")


if __name__ == "__main__":
    main()
