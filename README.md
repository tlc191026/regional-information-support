# Regional information support

Analysis code and aggregate source data for *Regional variation in information maintenance and team signalling in online cooperation*.
The study examines information practices among 126,000 League of Legends players across North American, Western European and Korean servers, covering 2,240,354 unique matches.

## Getting started

Use Python 3.14 and install the analysis packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

Recreate the four main figures and seven supplementary figures from the included data:

```bash
python code/render_figures.py --output-dir outputs/figures
```

The command saves PNG, PDF, SVG and 600 dpi TIFF files under `outputs/figures/main` and `outputs/figures/supplementary`.
It also records the input files and their checksums. Choose a new output directory when repeating a run.
The figures use Arial; font substitution on other systems may change text placement.

Run the event-parser, time-window and statistical utility checks:

```bash
python code/analysis/run_analysis.py --stage tests --output-root outputs/tests
```

These checks use synthetic fixtures included in the code and require no API credentials or match records.

## Repository contents

| Location | Contents |
|---|---|
| [`code/analysis`](code/analysis) | Sample construction, behavioural scoring, regional and contextual comparisons, timeline analyses and sensitivity checks |
| [`code/figures`](code/figures) | Figure layouts, shared visual settings and source-data mappings |
| [`code/validation`](code/validation) | Additional checks of model specification, temporal proximity, bootstrap approximation and participant overlap |
| [`data`](data) | Aggregate source data for 11 figures, the main table and 74 supplementary subtables |
| [`docs`](docs) | Analysis instructions and input specifications |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies |

## Data and reproducibility

The public data contain unrounded estimates, variable definitions, figure captions and table notes.
They correspond to the manuscript and Supplementary Information dated 23 September 2026.

- [Data guide and import examples](data/README.md)
- [Figure index](data/metadata/figure_index.csv)
- [Table index](data/metadata/table_index.csv)
- [Variable dictionary](data/dictionaries/variables.csv)
- [Record-level analysis guide](docs/analysis.md)
- [Additional validation guide](docs/validation.md)

The included data support reconstruction of the reported summaries and figures.
Refitting the statistical models requires the underlying player-level and match-level records, which are not included in this repository.
The [analysis guide](docs/analysis.md) identifies the required inputs and provides commands for researchers who have access to them.
Redistribution of data obtained through the Riot Games API is governed by [Riot Games' API terms](https://developer.riotgames.com/terms).

The fixed sample includes 2,520,000 player–match records. Six records with nonstandard positions are excluded from the regional and contextual regressions.
The primary timeline outcome analysis includes 2,210,921 matches; sample sizes for other analyses are given in the data guide and table notes.

When reading CSV files, preserve `NA` as the North America label and read game versions as strings.
For example, versions `15.1` and `15.10` are distinct.

## Software and licence

The code was tested with Python 3.14.3 on Windows. The analysis uses CPU computation and does not require a GPU.
Record-level models may require substantial memory because they analyse several million observations and repeated resamples.

The source code is distributed under the [MIT licence](LICENSE).
This licence does not grant rights to redistribute the underlying Riot Games API records.

This research is not endorsed by Riot Games and does not represent its views.
Riot Games, League of Legends and associated properties are trademarks or registered trademarks of Riot Games, Inc.
