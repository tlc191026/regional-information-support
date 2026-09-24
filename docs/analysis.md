# Reproducing the analyses

## Scope

The repository provides the code used for the manuscript's indicator relationships, regional comparisons, contextual analyses and timeline results.
Use `code/render_figures.py` to recreate the publication figures from the public aggregate data.
Statistical estimation requires the fixed record-level analytical inputs described below.

## Run an analysis stage

Run commands from the repository root. First check that the required inputs are present:

```bash
python code/analysis/run_analysis.py --stage regional --input-root /path/to/analytical_inputs --check-only
```

Then select an empty output directory:

```bash
python code/analysis/run_analysis.py --stage regional --input-root /path/to/analytical_inputs --output-root outputs/regional
```

The runner copies the selected inputs and code into an isolated working directory before execution.
It writes a run log, input checksums and a completion record. The supplied input directory is read only during this operation.
Directory names within `code/analysis/project` retain the layout used when generating the fixed analytical datasets.
The exact required paths are listed in [`analysis_inputs.json`](analysis_inputs.json).

| Stage | Analysis | Required record-level inputs |
|---|---|---|
| `relationships` | Player-level correlations and exhaustive indicator-group assignments | Player metric and process scores |
| `regional` | Adjusted regional means, joint process contrasts and score distributions | Participant scores, transformed metrics and fixed bootstrap weights |
| `signals` | Individual pings, occurrence proportions and conditional frequency | Regional inputs, canonical player–match sample and adjusted player scores |
| `context` | Tier and position profiles and interactions | Participant scores, canonical sample and fixed bootstrap weights |
| `supplement_additions` | Equal-ping weighting and early-tier comparisons | Regional inputs and scored 10-minute matches |
| `minute_profiles` | Minute trajectories and regional contrasts | Parsed minute shards, valid timeline features and participant-disjoint match selection |
| `objective_curves` | Standardized probability curves and regional probability changes | Scored 10-minute matches and fitted maintenance scale |
| `timeline_models` | Early regional comparisons and subsequent-objective associations | 10-minute and 15-minute match features |
| `timeline_sensitivity` | Alternative windows, components, samples and outcomes | Timeline features, sample linkage and historical comparison inputs |
| `prediction` | Focal-player, temporal and regional holdout checks | Scored matches, fixed player split and focal-player linkage |
| `tests` | Parser, time-boundary and statistical utility checks | Included synthetic fixtures |
| `raw_prepare` | Raw-file manifests and focal-player split | Canonical match metadata, focal-player linkage, earlier comparison sample and download database |

The runner also stages aggregate reference tables used to verify scoring and compare reproduced estimates.
These reference files must be present in the corresponding directories of the input root.
The preflight check tests the stage's listed primary inputs; the analysis scripts validate the additional reference tables when they are read.

## Raw event processing

Raw reconstruction requires the original read-only download database and indexed gzip shards.
Set `RESEARCH_DATABASE` to the database location before running `raw_prepare`.
Paths to raw shards are stored in the database and must resolve on the machine performing the reconstruction.

In the staged analysis directory, run the following modules from `manuscript_20260915/03_reproducibility/code` in order:

```bash
python parse_raw.py --kind matchdto
python parse_aligned.py --kind timeline
python assemble_data.py
python audit_dates.py
python run_models.py
```

Minute-level processing uses `parse_minutes.py` and `analyse_minutes.py` in `manuscript_20260915/13_timeline_dynamics_20260919/code`.
The parser worker count can be set with `--workers`.
No API credentials are required to process the retained files.

Upstream sample-construction and scoring modules are retained with their original relative paths under `code/analysis/project`.
They document the creation of fixed analytical inputs. The stage runner selects the analyses reported in the current manuscript.
These upstream modules also contain earlier methodological diagnostics; running their complete standalone workflows is not required for the current results.

## Numerical conventions

- Use `keep_default_na=False` when reading region labels from CSV files.
- Read game versions as strings and preserve the order of fixed resampling weights.
- The fixed sample contains 126,000 focal players, 2,520,000 player–match records and 2,240,354 unique matches.
- Regional and contextual regressions retain 2,519,994 records from 2,240,349 unique matches after position filtering.
- Valid 10-minute features cover 2,223,546 matches. The primary subsequent-objective sample contains 2,210,921 matches.
- Minute profiles contain 2,221,452 matches. Probability standardization uses a shared reference of 2,208,868 matches.
- The probability contrast from −1 to +1 common standard deviation spans two standard deviations.
- The principal analysis seed is 191026; the original fixed sampling seed is 20260601. Named child seeds are defined in the code.
- Descriptive analyses retain fixed scoring parameters. Predictive transformations are fitted using the training data only.

## Verification

The release was checked by running the parser, alignment and statistical utility suites and regenerating all 11 publication figures.
The reproduced PNG files were compared with the manuscript figure files at the pixel level.
The large record-level model stages were not rerun as part of the code release.

To compare rerun tables with the corresponding retained results:

```bash
python code/analysis/verify_reproduced_results.py --input-root /path/to/analytical_inputs --runs-root outputs/analysis_runs --report outputs/comparison.json
```

The comparison checks labels and numerical agreement using absolute tolerance `1e-10` and relative tolerance `1e-8`.
