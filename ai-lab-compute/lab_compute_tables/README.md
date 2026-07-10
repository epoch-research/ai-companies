# Frontier Lab Compute — Data Generator

Point-in-time estimates of the total compute rented or used by frontier AI
labs — Google DeepMind, Meta Superintelligence Labs, OpenAI, Anthropic — in
H100-equivalents (H100e), with P5 / median / P95 Monte Carlo uncertainty.
Output is one pandas DataFrame with a row per (lab × end-of-year) snapshot,
following the [ai-chip-components](https://github.com/epoch-research/ai-chip-components)
generator pattern: import the module and call getter functions.

No model structure lives here. The module runs the lab Monte Carlos from
[`../frontier_lab_compute_model.py`](../frontier_lab_compute_model.py) (priors
in [`../lab_model_params.csv`](../lab_model_params.csv), detailed walkthroughs
in the per-lab notebooks) and shapes their sample arrays into the export table.

## Usage

```python
from generate_tables import get_all_tables

tables = get_all_tables()          # dict of two DataFrames
tables["year_end_by_lab"]          # (end-of-year × lab) headline estimates
tables["intermediates_by_lab"]     # the quantities behind each estimate
```

Or the individual getters:

```python
from generate_tables import get_year_end_by_lab, get_intermediates_by_lab

df = get_year_end_by_lab()
```

From the command line (writes both `data/lab_compute_*.csv` files):

```
python3.11 generate_tables.py
```

Results are deterministic: each lab model reseeds 42 internally so it
reproduces its canonical notebook run — hence no seed parameter.

## Schema — `year_end_by_lab`

| Column | Type | Notes |
|--------|------|-------|
| `Name` | str | `"{Lab} end-{Year}"`, e.g. `"OpenAI end-2025"` |
| `Lab` | str | `Google DeepMind` / `Meta Superintelligence Labs` / `OpenAI` / `Anthropic` |
| `Year` | int | Calendar year of the snapshot |
| `Date` | str | `YYYY-12-31` — the point in time the estimate refers to |
| `h100e_p5` | float | 5th percentile, total H100e |
| `h100e_med` | float | Median, total H100e |
| `h100e_p95` | float | 95th percentile, total H100e |
| `Notes` | str | Generation timestamp |

H100e converts each chip at its dense 8-bit peak FLOP/s divided by the H100's
1979 TFLOP/s. Estimates cover compute **rented or used** by each lab (not
owned), at the stated moment in time — they are operational-stock snapshots,
not flows, so consecutive years must not be summed.

The 2024 rows for Google DeepMind and Meta are backcasts. They keep the same
`Lab` labels for continuity, but "the lab" in 2024 means *frontier-AI compute
at the company*: Meta Superintelligence Labs did not exist in 2024 (its
predecessor was Meta AI / GenAI plus FAIR), and the backcast share priors are
for those predecessor scopes — see `../lab_2024_backcasts.ipynb`.

## Schema — `intermediates_by_lab`

How each lab's final distribution is computed: one row per intermediate
quantity (owned fleets, deployment ratios, shares, power, chip counts, ...),
in model order. The traces come from `MODEL_STEPS` in
[`../frontier_lab_compute_model.py`](../frontier_lab_compute_model.py) — each
model records its steps as pure bookkeeping, so adding a `step(...)` entry
there is all it takes to extend this table (and the walkthrough page below).

| Column | Type | Notes |
|--------|------|-------|
| `Name` | str | `"{Lab} end-{Year} · {Label}"` |
| `Lab` / `Year` | str / int | Same conventions as `year_end_by_lab`; one trace per modelled (lab, year) snapshot |
| `Step` | int | 1-based position in the model's computation |
| `Variable` | str | Machine name; sheet-prior steps match their row name in `lab_model_params.csv` (the 2024 backcasts' fleet inputs come from the dashboard CSVs instead) |
| `Label` | str | Human-readable name |
| `Kind` | str | `input` (sampled prior) / `constant` (fixed scalar) / `derived` / `final` |
| `Units` | str | `H100e`, `MW`, `share`, `ratio`, `quarters`, `chips`, `USD B/yr`, ... |
| `Expression` | str | For derived/final rows: how the step combines earlier ones, by `Variable` name |
| `value_p5` / `value_med` / `value_p95` | float | Percentiles in the row's own units; constants repeat the same value |
| `Notes` | str | Generation timestamp |

Each snapshot's `final` row equals its `year_end_by_lab` row (tested).

## Coverage

| Lab | Year-ends |
|-----|-----------|
| OpenAI | 2023, 2024, 2025 |
| Anthropic | 2025 |
| Google DeepMind | 2024, 2025 |
| Meta Superintelligence Labs | 2024, 2025 |

OpenAI's power model yields a snapshot per disclosed year-end; DeepMind and
Meta add end-2024 backcast models (`model_deepmind_2024`, `model_msl_2024` in
the frontier script, promoted from `../lab_2024_backcasts.ipynb`). Anthropic's
end-2024 backcast lives in `../anthropic_2024_backcast.ipynb` and is
deliberately not exported here. Rows are omitted (not zero-padded) where no
model exists — treat a missing (lab, year) as "no estimate", not zero. xAI is
not covered: it has no Monte Carlo model in `frontier_lab_compute_model.py`
(only point estimates via `xai_aggregate.py` at the repo root).

## Consolidated page (draft)

```
python3.11 build_compute_page.py
```

regenerates [`index.html`](index.html), one self-contained page in the Epoch
website style holding both views of the data:

- **The chart**: x-axis = year, one bar per lab per year colored by lab
  (the site's `organizationColors` palette), bar height = `h100e_med`,
  whiskers from `h100e_p5` to `h100e_p95`, plus a collapsible data table.
  Hovering a bar shows median/CI and a link to that estimate's walkthrough
  section; clicking the bar (or Enter on it) jumps there. OpenAI's 2023–24
  bars link to the OpenAI end-2025 section, since they are per-year outputs
  of the same power model.
- **The walkthroughs**: one section per (lab × year-end) snapshot, one row
  per intermediate quantity in model order — a median dot with a capped
  90%-CI whisker on a full-width track (the same form as the "factors behind
  the total" summary charts in the lab notebooks), with the expression
  combining earlier steps under each derived row. Consecutive rows with the
  same kind of unit are grouped into panels, and all panels of a unit family
  within a section share one scale.

The page is a pure view over the two tables from `get_all_tables()` — the
same data as the CSVs — so new steps or snapshots added to `MODEL_STEPS` in
the frontier script appear on the page (and in the intermediates CSV) with no
changes to the viz code (a new snapshot also needs its (lab, year) →
trace-key entry in `LAB_YEAR_KEYS` in `generate_tables.py`). Re-run it
whenever the models or priors change. It has a dark theme keyed off
`prefers-color-scheme` (a `data-theme` attribute on `<html>` overrides it);
the lab colors are CVD-validated for both surfaces, with the dark theme
using a slightly adjusted purple and orange.

Two caveats for downstream rendering:

- **Don't derive cross-row totals or ratios from the percentile columns.**
  The lab models share one RNG stream (each reseeds 42), so per-sample draws
  are artificially aligned across labs and years: per-row percentiles are
  valid, but a stacked "all labs" bar with a credible interval — or a
  year-over-year growth CI — is not supported by this table. Stacking the
  medians for display is fine.
- Missing (lab, year) rows mean "no estimate", so a grouped (not stacked)
  layout reads best for the earlier years with partial lab coverage.

## Tests

```
python3.11 -m pytest tests/
```
