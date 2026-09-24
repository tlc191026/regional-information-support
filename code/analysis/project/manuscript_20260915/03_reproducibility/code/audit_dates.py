from settings import *
import pandas as pd

paths = [
    p
    for p in sorted((DATA / "matchdto_shards").glob("*.parquet"))
    if ".errors." not in p.name
]
df = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
assert df.match_id.is_unique
df["region"] = df.match_id.str.split("_").str[0]
df["date_utc"] = pd.to_datetime(df.game_start_ms, unit="ms", utc=True, errors="coerce")
df["calendar_month"] = df.date_utc.dt.strftime("%Y-%m")
df.groupby(["region", "calendar_month", "patch_raw"], observed=True).size().rename(
    "n_matches"
).reset_index().to_csv(TABLES / "raw_match_calendar_patch_coverage.csv", index=False)
out = {
    "n": len(df),
    "first_game": str(df.date_utc.min()),
    "last_game": str(df.date_utc.max()),
    "missing_date": int(df.date_utc.isna().sum()),
    "years": {
        str(k): int(v)
        for k, v in df.date_utc.dt.year.value_counts().sort_index().items()
    },
    "queues": {str(k): int(v) for k, v in df.queue_id_raw.value_counts().items()},
}
write_json(QA / "raw_game_dates.json", out)
print(json.dumps(out, indent=2))
