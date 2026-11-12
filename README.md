# Information work in temporary digital teams

Analysis code and aggregate source data for **Beyond usage intensity: How temporary digital teams organise information work**.

The study examines regional configurations of information maintenance and team signalling, their distribution across roles and competitive tiers, associations among participants, and early maintenance in relation to the next contested objective. The discovery sample comprises 126,000 focal players and 2,240,354 unique matches from the Korea, Western Europe and North America sampling rosters.

## Current release

The November 2026 analysis package corresponds to manuscript **V15** and its supplementary information. Start with [`analysis/README.md`](analysis/README.md) for the research-question map, workflow order and interpretation of the outputs.

```bash
python -m pip install -r analysis/requirements.txt
python analysis/run.py --stage figures --execute
```

This command reconstructs the five main figures and eleven supplementary figures from the included aggregate estimates. Outputs are written to `outputs/current_figures/`. Panel descriptions are provided in the manuscript captions rather than repeated above the plots.

| Directory | Contents |
| --- | --- |
| [`analysis/baseline/`](analysis/baseline/) | Sampling, extraction, scoring and initial analysis routines |
| [`analysis/revision/code/`](analysis/revision/code/) | Common-scale peer analyses and map-adjusted objective models |
| [`analysis/revision/theory/`](analysis/revision/theory/) | Regional configuration and role-pair analyses |
| [`analysis/revision/extensions/`](analysis/revision/extensions/) | Cross-activity, whole-team and time-window analyses, current figures and aggregate source data |
| [`analysis/timeline/`](analysis/timeline/) and [`analysis/minute_timeline/`](analysis/minute_timeline/) | Timestamped event parsing |
| [`code/`](code/), [`data/`](data/), [`docs/`](docs/) | September release, retained for earlier analyses and traceability |

## Data and reproducibility

The repository distributes analysis code, aggregate results, plotting data and variable documentation. Individual-level Riot API records, player identifiers and match identifiers are not included. Figure reconstruction uses the public aggregates. Full model re-estimation requires the individual-level inputs described in [`analysis/INPUTS.md`](analysis/INPUTS.md).

The two holdouts are re-estimation samples from the same collection; the strict holdout is nested within the player holdout. Supplementary prediction results concern the first post-cutoff neutral objective, not the final match winner. Regional configuration, peer and timeline analyses use different observational units; their counts and scales are recorded with the outputs.

[`analysis/provenance.json`](analysis/provenance.json) records source and release hashes. [`analysis/release_checks.json`](analysis/release_checks.json) records the packaging checks. Earlier files and their Git history remain available; the [September release guide](docs/september_release.md) documents their original structure.

## Licence

Code is available under the [MIT licence](LICENSE). Data-access and reuse conditions for Riot Games records remain those of the source platform. The repository does not include an API key or provide access to restricted source records.
