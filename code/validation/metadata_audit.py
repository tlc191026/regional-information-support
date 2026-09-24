"""Summarize match dates, version coverage and stored player-record timestamps."""

from common import *
import sqlite3
import numpy as np
import pandas as pd

db = sqlite3.connect(
    Path(os.environ["RESEARCH_DATABASE"]).resolve().as_uri() + "?mode=ro", uri=True
)
db.execute("PRAGMA query_only=ON")
source = ROOT / "manuscript_20260915/03_reproducibility/data"
f = pd.read_parquet(source / "canonical_focal_links.parquet")
t = pd.read_parquet(source / "matchdto_targets_all.parquet")
dates = pd.read_parquet(source / "match_dates_and_metadata.parquet")
players = pd.read_sql_query("SELECT puuid,region,tier,created_at FROM players", db)
assert not players.puuid.duplicated().any()
sample = (
    f[["puuid", "region", "tier"]]
    .drop_duplicates("puuid")
    .merge(
        players[["puuid", "region", "tier", "created_at"]],
        on="puuid",
        how="left",
        suffixes=("_sample", "_db"),
        validate="one_to_one",
    )
)
assert len(sample) == 126000 and sample.created_at.notna().all()
rows = []
for (r, tier), g in sample.groupby(["region_sample", "tier_sample"], observed=True):
    rows.append(
        dict(
            region=r,
            tier=tier,
            n_players=len(g),
            record_created_first=g.created_at.min(),
            record_created_last=g.created_at.max(),
            tier_disagreement=int(g.tier_sample.ne(g.tier_db).sum()),
            region_disagreement=int(g.region_sample.ne(g.region_db).sum()),
        )
    )
save("player_timestamp_provenance.csv", pd.DataFrame(rows))
match = t.merge(dates, on="match_id", how="left", validate="one_to_one")
match["date"] = pd.to_datetime(
    match.game_start_ms, unit="ms", utc=True, errors="coerce"
)
rows = []
for (r, patch), g in match.groupby(["region", "patch"], observed=True):
    known = g.date.notna()
    rows.append(
        dict(
            region=r,
            patch=patch,
            n_unique_matches=len(g),
            n_dates_available=int(known.sum()),
            first_match_utc=str(g.date.min()),
            last_match_utc=str(g.date.max()),
            queue_420=int(g.queue_id_raw.eq(420).sum()),
            queue_other=int((g.queue_id_raw.notna() & g.queue_id_raw.ne(420)).sum()),
        )
    )
save("match_dates_by_region_patch.csv", pd.DataFrame(rows))
joined = f.merge(
    dates[["match_id", "game_start_ms"]],
    on="match_id",
    how="left",
    validate="many_to_one",
).merge(
    players[["puuid", "created_at"]], on="puuid", how="left", validate="many_to_one"
)
# SQLite CURRENT_TIMESTAMP uses UTC; compare match and collection calendar dates in UTC.
md = (
    pd.to_datetime(joined.game_start_ms, unit="ms", utc=True, errors="coerce")
    .dt.tz_localize(None)
    .dt.normalize()
)
cd = pd.to_datetime(joined.created_at).dt.normalize()
joined["calendar_day_offset"] = (md - cd).dt.days
rows = []
for r, g in [("All", joined), *list(joined.groupby("region", observed=True))]:
    age = g.calendar_day_offset.dropna()
    q = age.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).to_numpy()
    rows.append(
        dict(
            region=r,
            n_player_match_records=len(g),
            n_dates_available=len(age),
            min_days=q[0],
            p05_days=q[1],
            p25_days=q[2],
            median_days=q[3],
            p75_days=q[4],
            p95_days=q[5],
            max_days=q[6],
            matches_before_record_date=int((age < 0).sum()),
            matches_same_record_date=int((age == 0).sum()),
            matches_after_record_date=int((age > 0).sum()),
        )
    )
save("match_player_record_date_offsets.csv", pd.DataFrame(rows))
prefix = match.match_id.str.split("_").str[0]
match["platform_prefix"] = prefix
save(
    "match_platform_prefixes.csv",
    match.groupby(["region", "platform_prefix"], observed=True)
    .size()
    .reset_index(name="n_unique_matches"),
)
dump(
    "metadata_summary.json",
    dict(
        n_players=len(sample),
        player_record_first=sample.created_at.min(),
        player_record_last=sample.created_at.max(),
        n_unique_matches=len(match),
        n_match_dates_available=int(match.date.notna().sum()),
        first_match_utc=str(match.date.min()),
        last_match_utc=str(match.date.max()),
        n_versions=int(match.patch.nunique()),
        versions=sorted(
            match.patch.astype(str).unique(),
            key=lambda s: tuple(map(int, s.split("."))),
        ),
        timestamp_definition="Stored players.created_at is an insertion timestamp. Collection code and logs independently corroborate player-list collection on 4 March 2026 UTC; it is not historical match-time tier.",
        schema=db.execute(
            "SELECT sql FROM sqlite_master WHERE name='players'"
        ).fetchone()[0],
    ),
)
log("Metadata audit complete")
