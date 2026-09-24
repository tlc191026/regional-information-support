"""Paths, event definitions and random seeds for timeline analyses."""

from pathlib import Path
import os, sys, json, hashlib, datetime

NEW = Path(__file__).resolve().parents[2]
ROOT = NEW.parent
REPRO = NEW / "03_reproducibility"
DATA = REPRO / "data"
TABLES = NEW / "02_timeline" / "tables"
QA = REPRO / "qa"
LOGS = REPRO / "logs"

os.environ.setdefault("MPLCONFIGDIR", str(LOGS / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
SEED = 191026
DB = Path(
    os.environ.get("RESEARCH_DATABASE", str(ROOT / "restricted_inputs/lol_data.db"))
)
REGIONS = ["NA1", "EUW1", "KR"]
LABELS = {"NA1": "NA", "EUW1": "EUW", "KR": "KR"}
TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
VALID_WARDS = {"YELLOW_TRINKET", "CONTROL_WARD", "SIGHT_WARD", "BLUE_TRINKET"}
WINDOWS = {"0_10": 600000, "0_15": 900000}
PARSER_VERSION = "20260915-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )


def child_seed(name):
    return int.from_bytes(
        hashlib.blake2b(f"{SEED}:{name}".encode(), digest_size=8).digest(), "little"
    ) % (2**32 - 1)


def log(message):
    print(datetime.datetime.now().isoformat(timespec="seconds"), message, flush=True)
