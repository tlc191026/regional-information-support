# Data for regional information maintenance and team signalling

This dataset accompanies *Regional variation in information maintenance and team signalling in online cooperation* (manuscript version 23 September 2026).
It contains aggregate source data for four main figures, seven supplementary figures, the main table and all 74 supplementary subtables.

## Find the data

The [figure index](metadata/figure_index.csv) and [table index](metadata/table_index.csv) link each display item to its data and include the complete captions and notes.
All indexed paths are relative to this `data` directory.

| Folder | Contents |
| --- | --- |
| `main/figures/` | Data for Figures 1–4, including distributions, adjusted means, confidence intervals and probability changes |
| `main/tables/` | Table 1 as displayed in the manuscript, plus unrounded estimates |
| `supplementary/figures/` | Data for Supplementary Figures 1–7 |
| `supplementary/tables/` | The 74 subtables within Supplementary Tables 1–28, as displayed in the Supplementary Information |
| `supplementary/results/` | Unrounded estimates and counts supporting the supplementary tables |
| `dictionaries/` | Variable definitions, column descriptions and category labels |
| `metadata/` | Figure and table indexes, file descriptions and version information |

The [variable dictionary](dictionaries/variables.csv) defines the behavioural indicators, process scores, timeline events and game-state variables.
The [column dictionary](dictionaries/columns.csv) explains each field in the numerical source files, including its units.
The [file manifest](metadata/file_manifest.csv) records what each row represents, file dimensions and numerical precision.

## Read the CSV files

Files use UTF-8 with a byte-order mark, comma separators and one header row.
Read the files as text first, then convert the numerical columns you need.
This preserves region labels, game versions and very small P values.

| Value | Meaning and import convention |
| --- | --- |
| `NA` or `NA1` | North America; preserve as text when importing |
| `EUW` or `EUW1` | Western Europe |
| `KR` | Korea |
| Game patch | Preserve as text; `15.1` and `15.10` identify different versions |
| Empty numeric cell | Unavailable or not applicable, depending on the column |

Run these examples from the repository root, which contains the `data/` folder.

### Python

```python
import pandas as pd

data = pd.read_csv(
    "data/main/figures/Figure2b.csv",
    encoding="utf-8-sig",
    keep_default_na=False,
    dtype=str,
)
for column in ["estimate", "ci_low", "ci_high"]:
    data[column] = pd.to_numeric(data[column])
```

### R

```r
data <- read.csv(
  "data/main/figures/Figure2b.csv",
  fileEncoding = "UTF-8-BOM",
  colClasses = "character",
  na.strings = "",
  check.names = FALSE
)
for (column in c("estimate", "ci_low", "ci_high")) {
  data[[column]] <- as.numeric(data[[column]])
}
```

### Numerical precision

Table CSVs preserve the manuscript's display format, including confidence intervals, thousands separators and inequalities such as `<0.001`.
Unrounded result files provide separate columns for estimates and interval bounds.
Use the table notes to identify the estimator, adjustment variables and uncertainty method.

Some P values are smaller than ordinary floating-point numbers can represent.
The files preserve values such as `3.3e-3665` as scientific-notation text.
Use the supplied `log10_p` or `log10_p_holm` columns, or software that supports arbitrary precision.
A zero in a numerical P-value column may reflect this computational limit.

## Samples and measurement units

Sample sizes depend on the observation unit and the records required for each analysis.

| Sample or analysis | Observation unit | Number |
| --- | --- | ---: |
| Fixed sample | Focal players | 126,000 |
| Fixed sample | Player–match records | 2,520,000 |
| Fixed sample | Unique matches | 2,240,354 |
| Regional and contextual regressions | Player–match records | 2,519,994 |
| Regional and contextual regressions | Unique matches | 2,240,349 |
| Minute profiles under common tier and patch support | Unique matches | 2,221,452 |
| Primary subsequent-objective analysis | Unique matches | 2,210,921 |
| Common reference for Figure 4d probabilities | Unique matches | 2,208,868 |

Regional and contextual regressions exclude six records with nonstandard team positions.

| Measure | Scale and interpretation |
| --- | --- |
| Full-match maintenance and signalling | Scores standardized across the full sample |
| Overall information investment | Equal-weight mean of maintenance and signalling, without further standardization |
| Natural ping frequency | Pings per player per 10 min, calculated over the full match |
| Ping occurrence | Fraction of valid player–match records containing at least one occurrence |
| Conditional ping frequency | Mean frequency among records containing the specified ping |
| Minute-specific maintenance | Events per team per minute |
| Adjusted probability | Probability on a 0–1 scale |
| Probability difference | Multiply by 100 to express the difference in percentage points |

The primary signalling score gives equal weight to three categories of pings.
Equal weighting of all nine individual ping types is a sensitivity analysis.
Performance indicators and the vision-score diagnostic do not enter either process score.

For the minute profiles, interval 1 covers the first minute, 0–1 min.
The final interval uses its actual cut-off time and duration.

### Probability curves and uncertainty

In Figure 4d, `maintenance_difference_sd` is the blue-team score minus the red-team score, divided by the common standard deviation (SD).
The difference is not centred, so zero represents equal team scores.
The model uses the centred predictor `model_predictor_z`, expressed in the same SD units.

Odds ratios describe a one-SD increase in the relevant predictor.
The probability difference compares +1 with −1 common SD, spanning two SDs.
Multiply `probability_difference` and its interval bounds by 100 to obtain the manuscript's percentage-point values.

In logistic result files, `ci_low` and `ci_high` generally describe the log-odds coefficient.
The columns `or_low` and `or_high` describe its odds-ratio interval.
Pointwise and simultaneous confidence intervals have separate columns in the relevant files.
The column dictionary and table notes specify interval methods and adjustments for multiple comparisons.

## Empirical distributions and data scope

Supplementary Figure 2a–c provides a separate score-frequency table for each measure.
Within each region, identical scores are combined, with their frequencies and cumulative counts retained.
These counts reproduce the original empirical cumulative distribution function (ECDF) exactly for all 42,000 players in each region.
Some score values have a frequency of one in these tables.
The measures are sorted separately and have no shared player key.

The dataset supports inspection and reconstruction of the reported summaries and figures.
Refitting the statistical models requires the underlying records, which are outside this aggregate dataset.
The files contain no player or match identifiers, participant-linkage tables, chat records, API credentials or raw event payloads.

File checksums are recorded in the manifest and in [checksums.sha256](checksums.sha256).
The latter covers every other file in this data directory.
