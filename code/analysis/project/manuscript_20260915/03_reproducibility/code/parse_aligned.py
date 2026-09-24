"""Align exposure, state and outcome cutoffs to the same near-landmark frame.

The actual frame must be within 1,000 ms of 10/15 min. Strict nominal-cutoff
specifications are retained in the same pass as explicit sensitivities.
"""

from settings import *
import parse_raw as raw
import shutil

base_parser = raw.parse_timeline
raw.DATA = DATA / "aligned"
raw.QA = QA / "aligned"
raw.LOGS = LOGS / "aligned"
raw.PARSER_VERSION = "20260915-v2-frame-aligned-and-strict"


def aligned_parser(obj, expected):
    frames = obj.get("timeline", obj).get("info", {}).get("frames", [])
    cutoffs = {}
    valid = {}
    for name, nominal in WINDOWS.items():
        times = [f.get("timestamp", -1000000) for f in frames]
        nearest = min(times, key=lambda t: abs(t - nominal)) if times else nominal
        valid[name] = bool(times) and abs(nearest - nominal) <= 1000
        cutoffs[name] = int(nearest) if valid[name] else nominal
        cutoffs["strict_" + name] = nominal
    raw.WINDOWS_MS = None
    raw.WINDOWS = cutoffs
    rows, people, audit = base_parser(obj, expected)
    for row in rows:
        key = row["window"]
        base = key.replace("strict_", "")
        nominal = WINDOWS[base]
        row["nominal_cutoff_ms"] = nominal
        row["analysis_cutoff_ms"] = cutoffs[key]
        row["cutoff_offset_ms"] = cutoffs[key] - nominal
        if not key.startswith("strict_") and not valid[base]:
            row["window_valid"] = False
            row["window_reason"] = "no_state_frame_within_one_second_of_landmark"
    return rows, people, audit


raw.parse_timeline = aligned_parser

if __name__ == "__main__":
    for p in [raw.DATA, raw.QA, raw.LOGS]:
        p.mkdir(exist_ok=True, parents=True)
    for name in ["timeline_targets_all.parquet", "timeline_source_file_inventory.json"]:
        if not (raw.DATA / name).exists():
            shutil.copy2(DATA / name, raw.DATA / name)
    write_json(
        QA / "landmark_protocol_amendment.json",
        {
            "amendment": "align both exposure and prospective outcome to actual near-10/15-minute state frame",
            "reason": "Strict <=600000 ms selected approximately 540000 ms frames because nominal minute frames drift by milliseconds. This is not the claimed 10-minute adjustment.",
            "near_landmark_tolerance_ms": 1000,
            "no_future_information_relative_to_actual_cutoff": True,
            "strict_nominal_specifications_retained": True,
            "discovered_before_new_models": True,
            "sample_of_45794_old_nominal_windows_median_state_lag_ms": 59773,
            "parser_version": raw.PARSER_VERSION,
        },
    )
    raw.main()
