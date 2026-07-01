# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (Plotly)
#     language: python
#     name: plotly_kernel
# ---

# %% [markdown]
# # OpenAI compute Monte Carlo
#
# Turns the single medians of `openai_power_model.ipynb` into full distributions
# by sampling the four main sources of uncertainty together:
#
# 1. **`new_chip_share`** — the chip-mix knob. The share of OpenAI's power placed
#    on the *newest* year's deployment mix; the rest follows the *default*
#    vintage-layered mix. At 0 the fleet is the default mix; at 1 it is entirely
#    the latest mix.
# 2. **Deployment lag** — Nvidia books revenue when chips ship; they go live some
#    quarters later. The lag controls which Microsoft snapshot each year reads.
# 3. **Systematic power factor** — one shared multiplier on every disclosed
#    megawatt figure, for the gap between disclosed and effective IT power.
# 4. **Rounding jitter** — each disclosure is reported to the nearest 0.1 GW, so
#    the true value sits uniformly within ±50 MW. Same band for every year, so
#    2025 (1.9 GW) is the most precise in relative terms and 2023 (0.2 GW) the
#    least.
#
# Watts per GPU and H100e per GPU are held fixed for now (watts per GPU is the
# sensible next thing to vary). With `new_chip_share = 0`, lag = 0, and nominal
# power, the model reproduces `openai_power_model.ipynb` exactly.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squigglepy as sq

N_SAMPLES = 5000
sq.set_seed(42)  # squigglepy has its own RNG; np.random.seed does not affect it

CHIP_TYPES = ['A100', 'H100/H200', 'B200', 'B300']
CHIP_COLORS = {'A100': '#8b5cf6', 'H100/H200': '#76b900', 'B200': '#1a73e8', 'B300': '#e8710a'}
CHIP_LABELS = {'A100': 'A100', 'H100/H200': 'H100/H200', 'B200': 'B200 (GB200)', 'B300': 'B300 (GB300)'}


def fmt(value):
    """Format an H100e or chip count as a short string (millions or thousands)."""
    if abs(value) >= 1e6:
        return f'{value / 1e6:.2f}M'
    return f'{value / 1e3:,.0f}k'


def percentiles(samples):
    """Return (5th, 50th, 95th) percentiles of a sample array."""
    p = sq.get_percentiles(samples, percentiles=[5, 50, 95])
    return p[5], p[50], p[95]


# %% [markdown]
# ## 1. Load the data and derive model inputs
#
# Three CSVs (the same ones the main model uses) give us: OpenAI's disclosed
# power per year, Microsoft's fleet over time, and IT power per chip.

# %%
data_dir = Path('ai-lab-compute') if Path('ai-lab-compute/IT power by chip.csv').exists() else Path('.')

owners_df = pd.read_csv(data_dir / 'nvidia_owners_cumulative_by_chip.csv')
chip_power_df = pd.read_csv(data_dir / 'IT power by chip.csv')
openai_df = pd.read_csv(data_dir / 'lab IT power.csv')
openai_df['Date'] = pd.to_datetime(openai_df['Date'])

# IT power per GPU. The power CSV names Blackwell parts "GB200"/"GB300"; the
# fleet data calls them "B200"/"B300", so we translate.
power_csv_to_chip_name = {'A100': 'A100', 'H100': 'H100/H200', 'GB200': 'B200', 'GB300': 'B300'}
watts_per_gpu = {
    power_csv_to_chip_name[row['Chip type']]: row['IT power per GPU (W)']
    for _, row in chip_power_df.iterrows()
    if row['Chip type'] in power_csv_to_chip_name
}

# Microsoft's cumulative fleet, every quarter, as (date x chip type) tables.
microsoft = owners_df[
    (owners_df['Owner'] == 'Microsoft') & (owners_df['Chip type'].isin(CHIP_TYPES))
].copy()
microsoft['End date'] = pd.to_datetime(microsoft['End date'])

microsoft_units = (
    microsoft.pivot_table(index='End date', columns='Chip type', values='Number of Units', aggfunc='first')
    .reindex(columns=CHIP_TYPES).fillna(0).sort_index()
)
microsoft_h100e = (
    microsoft.pivot_table(index='End date', columns='Chip type',
                          values='Compute estimate in H100e (median)', aggfunc='first')
    .reindex(columns=CHIP_TYPES).fillna(0).sort_index()
)

# The OpenAI disclosure dates (year-ends) are the model's time steps.
OPENAI_DATES = list(pd.to_datetime(openai_df['Date'].sort_values().unique()))

# H100e per GPU is a hardware ratio (roughly constant over time); read it off
# Microsoft's latest snapshot. Held fixed throughout.
last_date = OPENAI_DATES[-1]
h100e_per_gpu = {
    chip: (microsoft_h100e.loc[last_date, chip] / microsoft_units.loc[last_date, chip])
    if microsoft_units.loc[last_date, chip] > 0 else 0.0
    for chip in CHIP_TYPES
}

# Microsoft's cumulative IT power (MW) by chip, across every quarter.
microsoft_power = (microsoft_units * pd.Series(watts_per_gpu) / 1e6).sort_index()

# OpenAI's disclosed power per year, and the new power added each year.
disclosed_power_mw = {d: float(p) for d, p in zip(OPENAI_DATES, openai_df.sort_values('Date')['Total IT power (MW)'])}
power_added_mw = {}
previous = 0.0
for date in OPENAI_DATES:
    power_added_mw[date] = disclosed_power_mw[date] - previous
    previous = disclosed_power_mw[date]

print('Watts per GPU (fixed): ', {c: f'{w:,.0f} W' for c, w in watts_per_gpu.items()})
print('H100e per GPU (fixed): ', {c: round(v, 2) for c, v in h100e_per_gpu.items()})
print('Disclosed power (MW):  ', {d.strftime('%Y'): int(p) for d, p in disclosed_power_mw.items()})

# %% [markdown]
# ## 2. The model
#
# The whole model rests on Microsoft's deployment mix — the share of newly added
# power going to each chip type each year. We build two versions of OpenAI's
# fleet from it:
#
# - **default** — vintage-layered: each year's added power keeps the mix Microsoft
#   was deploying then, and carries forward. This is the main model.
# - **newest** — the entire fleet placed on the most recent year's mix.
#
# `new_chip_share` blends the two. The chosen mix is then scaled by OpenAI's
# total power to get megawatts per chip, then chip counts, then H100e.
#
# The **deployment lag** shifts the dates at which we read Microsoft's fleet.
# Because Microsoft's data is quarterly, we just interpolate its cumulative power
# to whatever (possibly fractional) date a sampled lag implies — `np.interp`
# does this for the entire sample array at once, so no lookup grid is needed.

# %%
QUARTER_DAYS = 365.25 / 4
series_start = microsoft_power.index[0]
microsoft_day = np.array([(d - series_start).days for d in microsoft_power.index], dtype=float)
openai_day = {date: (date - series_start).days for date in OPENAI_DATES}


def microsoft_power_on(day, chip):
    """Microsoft's cumulative IT power (MW) for one chip, interpolated to any day.
    Accepts a scalar or an array of days. Lags before the data start clamp to the
    earliest snapshot."""
    return np.interp(day, microsoft_day, microsoft_power[chip].values)


def chip_power_shares(lag_quarters):
    """For each OpenAI year, the share of total power on each chip type, under a
    given deployment lag. Returns (default_shares, newest_shares), each a
    {date: {chip: share}} dict whose chip shares sum to 1.

    lag_quarters may be a scalar or an array (one value per Monte Carlo sample);
    every share comes back the same shape.
    """
    lag_days = lag_quarters * QUARTER_DAYS

    # Microsoft's cumulative power by chip, as seen `lag` quarters before each date.
    cumulative = {
        date: {chip: microsoft_power_on(openai_day[date] - lag_days, chip) for chip in CHIP_TYPES}
        for date in OPENAI_DATES
    }

    # The mix of power Microsoft *added* in each step: cumulative for the first
    # step (so legacy A100s are counted), incremental afterward.
    added_mix = {}
    for i, date in enumerate(OPENAI_DATES):
        if i == 0:
            added = {chip: np.maximum(cumulative[date][chip], 0.0) for chip in CHIP_TYPES}
        else:
            prev = OPENAI_DATES[i - 1]
            added = {chip: np.maximum(cumulative[date][chip] - cumulative[prev][chip], 0.0) for chip in CHIP_TYPES}
        total = sum(added.values())
        added_mix[date] = {chip: added[chip] / total for chip in CHIP_TYPES}

    # Default carries each year's additions forward at their own mix; newest uses
    # the latest year's mix for the whole fleet.
    default_shares, newest_shares = {}, {}
    carried_power = {chip: 0.0 for chip in CHIP_TYPES}
    for date in OPENAI_DATES:
        for chip in CHIP_TYPES:
            carried_power[chip] = carried_power[chip] + power_added_mw[date] * added_mix[date][chip]
        carried_total = sum(carried_power.values())
        default_shares[date] = {chip: carried_power[chip] / carried_total for chip in CHIP_TYPES}
        newest_shares[date] = added_mix[date]
    return default_shares, newest_shares


def run_monte_carlo(new_chip_share, lag_quarters, total_power):
    """Per-year, per-chip chip counts and H100e for the given parameter samples
    (each argument is a length-N array; total_power is a {date: array} dict)."""
    default_shares, newest_shares = chip_power_shares(lag_quarters)
    results = {}
    for date in OPENAI_DATES:
        counts, h100e = {}, {}
        for chip in CHIP_TYPES:
            # Blend the two mixes, then size the result by OpenAI's total power.
            share = (1 - new_chip_share) * default_shares[date][chip] + new_chip_share * newest_shares[date][chip]
            megawatts = total_power[date] * share
            counts[chip] = megawatts * 1e6 / watts_per_gpu[chip]
            h100e[chip] = counts[chip] * h100e_per_gpu[chip]
        results[date] = {
            'counts': counts,
            'h100e': h100e,
            'total_counts': sum(counts.values()),
            'total_h100e': sum(h100e.values()),
        }
    return results


# How the newest-year power mix shifts as the lag grows (more lag -> older mix).
print('Newest-year power mix at', last_date.strftime('%Y'), 'by deployment lag:')
for lag in [0.0, 1.0, 2.0]:
    _, newest = chip_power_shares(lag)
    shares = ', '.join(f'{c} {newest[last_date][c] * 100:.0f}%' for c in CHIP_TYPES if newest[last_date][c] > 0.005)
    print(f'   {lag:.0f}Q lag: {shares}')

# %% [markdown]
# ## 3. Parameters and sampling
#
# - **`new_chip_share`** — baseline prior `Beta(2,6)` (mean 0.25): most capacity
#   sits on longer contracts that don't refresh to the newest chips. Section 6
#   sweeps alternatives.
# - **Deployment lag** — lognormal, 90% range 0.5–2 quarters (median 1).
# - **Systematic power factor** — lognormal, 90% range 0.85–1.15, shared across
#   years.
# - **Rounding jitter** — uniform ±50 MW per year (the 0.1 GW rounding step).

# %%
new_chip_share_prior = sq.beta(2, 6)
lag_prior = sq.to(0.5, 2.0)
systematic_power_prior = sq.to(0.85, 1.15)
ROUNDING_MW = 50.0


def sample_total_power(n):
    """Total power per year: one shared systematic factor times each disclosure
    plus its own independent rounding jitter."""
    systematic_factor = systematic_power_prior @ n
    return {
        date: (disclosed_power_mw[date] + (sq.uniform(-ROUNDING_MW, ROUNDING_MW) @ n)) * systematic_factor
        for date in OPENAI_DATES
    }


# Central values, used to hold a parameter still while isolating another (section 5).
new_chip_share_central = float(np.median(new_chip_share_prior @ 20000))
lag_central = 1.0  # geometric mean of the 0.5–2.0 quarter range

print(f'new_chip_share central (median): {new_chip_share_central:.2f}')
print(f'deployment lag central (median): {lag_central:.2f} quarters')

# %% [markdown]
# ## 4. Run the Monte Carlo
#
# All four inputs vary together, so each sample is one coherent scenario.

# %%
new_chip_share = new_chip_share_prior @ N_SAMPLES
lag_quarters = lag_prior @ N_SAMPLES
total_power = sample_total_power(N_SAMPLES)

mc = run_monte_carlo(new_chip_share, lag_quarters, total_power)

# Deterministic reference: new_chip_share 0, no lag, nominal power = the main model.
reference = run_monte_carlo(
    np.array([0.0]), np.array([0.0]), {d: np.array([disclosed_power_mw[d]]) for d in OPENAI_DATES})
reference_h100e = {d: reference[d]['total_h100e'][0] for d in OPENAI_DATES}

print('Total H100e by year (5th / median / 95th), and the main-model reference:')
for date in OPENAI_DATES:
    lo, mid, hi = percentiles(mc[date]['total_h100e'])
    print(f'   {date.strftime("%Y")}: {fmt(lo)} / {fmt(mid)} / {fmt(hi)}   (reference {fmt(reference_h100e[date])})')

# Chip-mix composition at the latest date.
a100_share = mc[last_date]['counts']['A100'] / mc[last_date]['total_counts']
blackwell_share = (mc[last_date]['counts']['B200'] + mc[last_date]['counts']['B300']) / mc[last_date]['total_counts']
print(f'\n{last_date.strftime("%Y")} chip-count shares (5th / median / 95th):')
print(f'   A100:      {" / ".join(f"{v:.0%}" for v in percentiles(a100_share))}')
print(f'   Blackwell: {" / ".join(f"{v:.0%}" for v in percentiles(blackwell_share))}')

# %% [markdown]
# ## 5. Results

# %%
fig, (ax_years, ax_hist) = plt.subplots(1, 2, figsize=(15, 5.5))

# Left: total H100e by year, with its 90% interval and the main-model reference.
years = [d.year for d in OPENAI_DATES]
median_by_year = [np.median(mc[d]['total_h100e']) / 1e6 for d in OPENAI_DATES]
low_by_year = [percentiles(mc[d]['total_h100e'])[0] / 1e6 for d in OPENAI_DATES]
high_by_year = [percentiles(mc[d]['total_h100e'])[2] / 1e6 for d in OPENAI_DATES]
reference_by_year = [reference_h100e[d] / 1e6 for d in OPENAI_DATES]

ax_years.fill_between(years, low_by_year, high_by_year, color='#1a73e8', alpha=0.18, label='90% interval')
ax_years.plot(years, median_by_year, color='#1a73e8', lw=2.4, marker='o', label='median')
ax_years.plot(years, reference_by_year, color='#888780', lw=1.6, ls='--', marker='s',
              markerfacecolor='white', label='main-model reference')
ax_years.set_title('OpenAI compute with uncertainty', fontsize=12)
ax_years.set_ylabel('Total H100e (millions)')
ax_years.set_xticks(years)
ax_years.legend()
ax_years.grid(True, alpha=0.3)

# Right: distribution of the latest-year total.
latest_samples = mc[last_date]['total_h100e'] / 1e6
ax_hist.hist(latest_samples, bins=60, color='#1a73e8', alpha=0.85, edgecolor='white')
for value, style in zip(percentiles(mc[last_date]['total_h100e']), ['--', '-', '--']):
    ax_hist.axvline(value / 1e6, color='#0c447c', ls=style, lw=1.5)
ax_hist.set_title(f'{last_date.strftime("%Y")} total H100e', fontsize=12)
ax_hist.set_xlabel('Total H100e (millions)')
ax_hist.set_ylabel('Monte Carlo samples')
ax_hist.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. What drives the uncertainty
#
# Each source is turned on alone (the others held at their central values), so
# the spread in the latest-year total belongs to just that source.

# %%
central_share = np.full(N_SAMPLES, new_chip_share_central)
central_lag = np.full(N_SAMPLES, lag_central)
nominal_power = {d: np.full(N_SAMPLES, disclosed_power_mw[d]) for d in OPENAI_DATES}

decomposition = {
    'new_chip_share only': run_monte_carlo(new_chip_share, central_lag, nominal_power),
    'deployment lag only': run_monte_carlo(central_share, lag_quarters, nominal_power),
    'power (factor + jitter) only': run_monte_carlo(central_share, central_lag, total_power),
    'all combined': mc,
}

fig, ax = plt.subplots(figsize=(11, 4.5))
sources = list(decomposition)
for row, name in enumerate(sources):
    lo, mid, hi = percentiles(decomposition[name][last_date]['total_h100e'])
    ax.barh(row, (hi - lo) / 1e6, left=lo / 1e6, color='#1a73e8', alpha=0.55, height=0.5)
    ax.plot(mid / 1e6, row, marker='|', color='#0c447c', markersize=18, markeredgewidth=2)
    ax.text((hi + 20000) / 1e6, row, f'90% width {fmt(hi - lo)}', va='center', fontsize=9, color='#444441')
ax.axvline(reference_h100e[last_date] / 1e6, color='#888780', ls='--', lw=1.4)
ax.set_yticks(range(len(sources)))
ax.set_yticklabels(sources)
ax.invert_yaxis()
ax.set_xlabel(f'{last_date.strftime("%Y")} total H100e (millions)')
ax.set_title('Uncertainty contribution by source', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print('Latest-year total H100e by source (5th / median / 95th):')
for name in sources:
    lo, mid, hi = percentiles(decomposition[name][last_date]['total_h100e'])
    print(f'   {name:30s}: {fmt(lo)} / {fmt(mid)} / {fmt(hi)}')

# %% [markdown]
# ## 7. Sensitivity to the `new_chip_share` prior
#
# Re-run the full Monte Carlo (lag and power still varying, reusing the same
# samples) under each candidate prior.

# %%
priors = {
    'Beta(2,18) — mean 0.10': sq.beta(2, 18),
    'Beta(2,6) — mean 0.25': sq.beta(2, 6),
    'Beta(2,2) — mean 0.50': sq.beta(2, 2),
    'Uniform — mean 0.50': sq.uniform(0, 1),
    'Beta(6,2) — mean 0.75': sq.beta(6, 2),
}

prior_h100e = {
    name: run_monte_carlo(prior @ N_SAMPLES, lag_quarters, total_power)[last_date]['total_h100e']
    for name, prior in priors.items()
}

fig, ax = plt.subplots(figsize=(11, 5))
ax.boxplot([prior_h100e[name] / 1e6 for name in priors], showfliers=False, widths=0.55)
ax.set_xticks(range(1, len(priors) + 1))
ax.set_xticklabels([n.replace(' — ', '\n') for n in priors], fontsize=8)
ax.axhline(reference_h100e[last_date] / 1e6, color='#888780', ls='--', lw=1.4, label='main-model reference')
ax.set_ylabel(f'{last_date.strftime("%Y")} total H100e (millions)')
ax.set_title('Latest-year compute by new_chip_share prior (full uncertainty)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print('Latest-year total H100e by new_chip_share prior (5th / median / 95th):')
for name in priors:
    lo, mid, hi = percentiles(prior_h100e[name])
    print(f'   {name:24s}: {fmt(lo)} / {fmt(mid)} / {fmt(hi)}')

# %% [markdown]
# ### Takeaways
#
# - The biggest driver of total-compute uncertainty is the **systematic power
#   factor** — it scales the whole fleet, so its ±15% flows almost directly into
#   the H100e total.
# - **`new_chip_share`** and the **deployment lag** move the total only modestly
#   (the fleet is already mostly newest-vintage), but they are what reshape the
#   **chip composition** — A100 vs. Blackwell share.
# - Even sweeping the `new_chip_share` prior across its full range leaves the
#   total in a fairly narrow band, so the headline number is robust; the chip mix
#   is where the assumptions bite.
# - Natural next addition: let watts per GPU vary, which feeds both the mix and
#   the power-to-chip conversion.
