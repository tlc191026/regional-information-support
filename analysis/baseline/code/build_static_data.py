"""Download Data Dragon champion metadata and archived public rank-distribution pages."""
from __future__ import annotations

import json
import re
import urllib.request

import pandas as pd

from config import DATA, TABLES, ensure_dirs, log, write_json

STATIC = DATA / "static"
SNAPSHOTS = {
    "KR": "http://web.archive.org/web/20251112154023/https://www.leagueofgraphs.com/rankings/rank-distribution/kr",
    "EUW1": "http://web.archive.org/web/20251030144408/https://www.leagueofgraphs.com/rankings/rank-distribution/euw",
    "NA1": "http://web.archive.org/web/20250423084021/https://www.leagueofgraphs.com/rankings/rank-distribution/na",
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-script"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        return resp.read()


def champions() -> pd.DataFrame:
    versions = json.loads(fetch("https://ddragon.leagueoflegends.com/api/versions.json"))
    version = versions[0]
    payload = json.loads(fetch(f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"))
    (STATIC / f"champion_{version}.json").write_text(json.dumps(payload), encoding="utf-8")
    rows = [{
        "champion_id": int(c["key"]), "champion_name": c["id"], "tags": "|".join(c.get("tags", [])),
        "primary_tag": (c.get("tags") or ["None"])[0], "partype": c.get("partype", ""),
    } for c in payload["data"].values()]
    frame = pd.DataFrame(rows).sort_values("champion_id")
    frame.to_csv(TABLES / "static_champions.csv", index=False)
    log(f"Data Dragon {version}: {len(frame)} champions")
    return frame


def snapshots() -> None:
    for region, url in SNAPSHOTS.items():
        html = fetch(url)
        (STATIC / f"rank_distribution_{region}.html").write_bytes(html)
        text = html.decode("utf-8", errors="replace")
        hits = re.findall(r"(Iron|Bronze|Silver|Gold|Platinum|Emerald|Diamond|Master|Grandmaster|Challenger)"
                          r"(?:\s+(IV|III|II|I))?[^%]{0,400}?(\d+(?:\.\d+)?)%", text)
        log(f"{region}: {len(html):,} bytes; first matches {hits[:6]}")


def main() -> None:
    ensure_dirs()
    STATIC.mkdir(parents=True, exist_ok=True)
    champs = champions()
    snapshots()
    write_json(TABLES / "qa" / "static_sources.json", {"champions": int(len(champs)), "rank_snapshots": SNAPSHOTS})


if __name__ == "__main__":
    main()
