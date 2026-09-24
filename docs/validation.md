# Additional validation analyses

These analyses use the fixed study sample and descriptive scoring parameters.
They examine model specification, signal weighting, temporal proximity, objective timing, bootstrap approximation and participant overlap.
The checks were specified after the main findings were known.

## Running the checks

Install the packages in `requirements.txt` and run from the repository root.
Set `RESEARCH_INPUT_ROOT`, `RESEARCH_DATABASE` and `VALIDATION_OUTPUT_ROOT` to the analytical input directory, collection database and a separate output directory.

For PowerShell:

```powershell
$env:RESEARCH_INPUT_ROOT = '/path/to/analytical_inputs'
$env:RESEARCH_DATABASE = '/path/to/collection.db'
$env:VALIDATION_OUTPUT_ROOT = '/path/to/validation_outputs'
python code/validation/metadata_audit.py
python code/validation/regional_checks.py
python code/validation/proximate_sample_check.py
python code/validation/timeline_checks.py
python code/validation/raw_field_audit.py
```

For Bash, set the same variables with `export NAME=/path/to/location` before running the Python commands.
The database connections are read only. Raw shard paths are read from the retained manifest.
No API calls or credentials are required. The raw-field audit scans all 475 indexed gzip shards.

## Inputs

- `exp_20260717/data/context_unresidualized_participant_scores.parquet`
- `exp_20260717/data/participant_metrics_transformed_z.parquet`
- `exp_20260717/data/bootstrap_weights_primary_region_tier_1000.npy`
- Under `manuscript_20260915/03_reproducibility/data`: `canonical_focal_links.parquet`, `matchdto_targets_all.parquet`, `match_dates_and_metadata.parquet`, `analysis_scored_0_10.parquet` and `focal_player_match_split.parquet`.
- The `players` table of the collection database and the raw shards indexed by `matchdto_targets_all.parquet`.

## Supplementary results

| Script | Supplementary tables | Purpose |
|---|---|---|
| `metadata_audit.py` | 24a,d | Match dates, stored collection timestamps and platform labels |
| `regional_checks.py` | 25a,b; 27a | Regional contrasts under interactions, equal-ping endpoints and paired OLS bootstrap refits |
| `proximate_sample_check.py` | 25c–e | Matches within 30 calendar days of player-list collection and on the corresponding platform |
| `timeline_checks.py` | 26a,b; 27b | Original first-objective composition, exclusion of events within 60 seconds and paired logistic bootstrap refits |
| `raw_field_audit.py` | 24b,c; 28 | Present versus missing signal fields and participant overlap in focal-player holdout |

Published estimates are available under [`data/supplementary/results`](../data/supplementary/results).
Table titles, notes and source files are indexed in [`table_index.csv`](../data/metadata/table_index.csv).

Sampling tier refers to the focal player's tier at player-list collection.
The 60-second sensitivity analysis excludes matches whose original first subsequent objective occurs within that interval; it does not select a later event instead.
Its intervals use match-level HC0 standard errors on the log-odds scale.
The paired 300-resample checks compare the interval approximation with full refits in fixed subsamples.
Participant overlap is evaluated among all participants of matches with retrievable raw records.

Record-level intermediates, including participant hashes, are written only to the chosen output directory and are not part of the public dataset.
