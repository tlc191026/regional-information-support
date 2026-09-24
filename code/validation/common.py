"""Shared paths and numerical helpers for the additional validation analyses."""

from pathlib import Path
import os, sys, json, hashlib, time

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "4"
sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
if "RESEARCH_INPUT_ROOT" not in os.environ:
    raise RuntimeError(
        "Set RESEARCH_INPUT_ROOT to the directory containing the record-level analytical inputs."
    )
ROOT = Path(os.environ["RESEARCH_INPUT_ROOT"]).resolve()
W = Path(
    os.environ.get("VALIDATION_OUTPUT_ROOT", str(P / "outputs/validation"))
).resolve()
for name in ["qa", "tables", "data", "logs"]:
    (W / name).mkdir(parents=True, exist_ok=True)
SEED = 191026
REG = ["NA", "EUW", "KR"]
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
POS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
M = "shared_information_maintenance"
S = "team_signalling"
PINGS = [
    "enemy_missing_pings_pm",
    "enemy_vision_pings_pm",
    "command_pings_pm",
    "need_vision_pings_pm",
    "assist_me_pings_pm",
    "on_my_way_pings_pm",
    "push_pings_pm",
    "all_in_pings_pm",
    "get_back_pings_pm",
]


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def dump(name, obj):
    (W / "qa" / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        "utf-8",
    )


def save(name, df):
    df.to_csv(W / "tables" / name, index=False, encoding="utf-8-sig")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()
