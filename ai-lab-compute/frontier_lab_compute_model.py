"""Consolidated frontier-lab compute Monte Carlo.

One master script holding the model structure and results for four labs'
end-2025 compute, in H100-equivalents (H100e):

    Google DeepMind, Meta Superintelligence Labs (MSL), OpenAI, Anthropic

The sampled priors live in lab_model_params.csv (loaded via
lab_compute_utils.load_lab_params) — the single source of truth shared with the
lab notebooks. This file holds only model structure. Each lab keeps the
canonical model from its own notebook; the detailed walkthroughs,
visualizations, and sensitivity sweeps live in the individual notebooks
(deepmind_compute_model, msl_compute_model, openai_compute_monte_carlo,
anthropic_compute_monte_carlo) and are intentionally not reproduced here.

For OpenAI and Anthropic the *power-based* model is canonical; the cloud-spend
analyses are deliberately excluded.

Besides the end-2025 headline models, section 5 holds end-2024 backcasts for
Google DeepMind and Meta AI (pre-MSL), promoted from the lab_2024_backcasts
notebook. OpenAI's end-2024 falls out of model_openai() (its power series is
per-year); Anthropic's backcast stays in anthropic_2024_backcast and is
deliberately not promoted here.

Each model also records the intermediate quantities behind its final
distribution in MODEL_STEPS (pure bookkeeping, no effect on results), which
lab_compute_tables/ exports as a table and a walkthrough page.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squigglepy as sq

N_SAMPLES = 5000
H100_FLOPS = 1.979e15  # H100 dense 8-bit FLOP/s, the H100e denominator

# Data lives at repo root (csv_export/) and in ai-lab-compute/; resolve either
# whether the script is run from the repo root or from ai-lab-compute/.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

sys.path.insert(0, str(HERE))  # so lab_compute_utils resolves from any cwd
from lab_compute_utils import load_lab_params

# Each model reseeds 42 so it reproduces its canonical notebook run. Side
# effect: the labs share one RNG stream, so per-sample values are artificially
# aligned across labs — fine for the per-lab percentiles reported here, but do
# not sum or ratio samples ACROSS labs without reseeding each model differently.


def fmt(x):
    """Format an H100e / chip count as a short string (millions or thousands)."""
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}M"
    return f"{x / 1e3:,.0f}k"


def pctiles(samples):
    """Return (5th, 50th, 95th) percentiles of a sample array."""
    p = sq.get_percentiles(samples, percentiles=[5, 50, 95])
    return p[5], p[50], p[95]


# Each model_*() call refreshes its lab's entry here with an ordered list of
# the intermediate quantities behind its final distribution, so downstream
# exports (lab_compute_tables/) can show how each estimate is built. Pure
# bookkeeping: recording steps draws no samples and changes no results. Read
# a lab's entry right after calling its model.
MODEL_STEPS = {}


def step(name, label, samples, units, kind, expression=""):
    """One named quantity in a model's decomposition. kind is 'input' (sampled
    prior), 'constant' (fixed scalar), 'derived', or 'final'; samples is a
    sample array (or a scalar for constants); expression says how a derived
    quantity combines earlier steps, referring to them by name."""
    return dict(name=name, label=label, samples=samples, units=units,
                kind=kind, expression=expression)


# ---------------------------------------------------------------------------
# 1. Google DeepMind
# ---------------------------------------------------------------------------
# DeepMind H100e = total_owned x deployment_lag x DeepMind_fraction, where the
# DeepMind fraction blends its share of Google's cloud half and its share of the
# internal (non-cloud) half:
#   fraction = cloud_share * dm_cloud_share + (1 - cloud_share) * dm_noncloud_share
# The two DeepMind shares are drawn with a modest positive correlation (from the
# sheet); the owned fleet = Nvidia GPUs + Google TPUs (drawn independently).

def model_deepmind():
    sq.set_seed(42)
    P = load_lab_params()["deepmind"]
    nvidia_owned = P["nvidia_owned"] @ N_SAMPLES
    google_owned = P["google_owned"] @ N_SAMPLES  # TPU fleet
    total_owned = nvidia_owned + google_owned

    # Only part of the owned fleet is online at any moment.
    deployment_lag = P["deployment_lag"] @ N_SAMPLES
    operational = total_owned * deployment_lag

    # CFO's "around half" cloud vs internal split, and DeepMind's slice of each.
    cloud_share = P["cloud_share"] @ N_SAMPLES
    # The two DeepMind shares plausibly move together, so draw them with a modest
    # positive correlation (from the sheet) rather than independently — independent
    # draws let a high cloud share offset a low non-cloud share and artificially
    # narrow the DeepMind CI. P holds fresh dist objects, safe for sq.correlate to tie.
    dm_cloud_dist, dm_noncloud_dist = sq.correlate(
        (P["dm_cloud_share"], P["dm_noncloud_share"]), P["dm_share_correlation"])
    dm_cloud_share = dm_cloud_dist @ N_SAMPLES      # enterprise Gemini + external rentals
    dm_noncloud_share = dm_noncloud_dist @ N_SAMPLES  # consumer Gemini + DM R&D
    dm_fraction = cloud_share * dm_cloud_share + (1 - cloud_share) * dm_noncloud_share
    dm_h100e = operational * dm_fraction

    MODEL_STEPS["deepmind"] = [
        step("nvidia_owned", "Google-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("google_owned", "Google TPU fleet", google_owned, "H100e", "input"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "nvidia_owned + google_owned"),
        step("deployment_lag", "Operational share of owned", deployment_lag, "ratio", "input"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "total_owned × deployment_lag"),
        step("cloud_share", "Cloud share of Google ML compute", cloud_share, "share", "input"),
        step("dm_cloud_share", "DeepMind share of the cloud half", dm_cloud_share, "share", "input"),
        step("dm_noncloud_share", "DeepMind share of the internal half", dm_noncloud_share,
             "share", "input"),
        step("dm_fraction", "DeepMind fraction of the operational fleet", dm_fraction,
             "share", "derived",
             "cloud_share × dm_cloud_share + (1 − cloud_share) × dm_noncloud_share"),
        step("total_h100e", "DeepMind compute, end-2025", dm_h100e, "H100e", "final",
             "operational × dm_fraction"),
    ]
    return dm_h100e


# ---------------------------------------------------------------------------
# 2. Meta Superintelligence Labs (MSL)
# ---------------------------------------------------------------------------
# MSL H100e = total_owned x deployment_lag x MSL_share + rented_cloud. Meta's
# fleet is almost entirely internal at end-2025; the MSL share is sampled
# directly (frontier AI work vs core-business recommenders). The rented term
# covers the CoreWeave/Google/Oracle deals signed Sept-Oct 2025: a
# zero-inflated spend run rate converted at GB200/GB300 3-year rental prices
# (SemiAnalysis InferenceX Aug 2025) -- see msl_compute_model section 4.
# Owned fleet = Nvidia GPUs + AMD Instinct (drawn independently); MTIA excluded.

def model_msl():
    sq.set_seed(42)
    P = load_lab_params()["msl"]
    nvidia_owned = P["nvidia_owned"] @ N_SAMPLES
    amd_owned = P["amd_owned"] @ N_SAMPLES
    total_owned = nvidia_owned + amd_owned

    # Slightly higher lag ratios than DeepMind (Meta's fleet ramped fast in 2025).
    deployment_lag = P["deployment_lag"] @ N_SAMPLES
    operational = total_owned * deployment_lag

    # Frontier vs core-business split — highly uncertain, see the sheet's notes.
    msl_share = P["msl_share"] @ N_SAMPLES
    msl_owned = operational * msl_share

    # Rented cloud, no deployment-lag haircut (billed as delivered). Spend run
    # rate ($B/yr) buys H100e at an uncertain price per H100e-hour: GB200-GB300
    # 3-year rental range, both ~2.5 H100e per GPU (see the notebook's caveats
    # on 3- vs 5-6-year pricing and the spring-2026 large-deal repricing).
    cloud_spend = sq.zero_inflated(P["cloud_p_nothing_online"],
                                   P["cloud_spend_run_rate"]) @ N_SAMPLES
    price_per_h100e_hour = sq.to(3.30 / 2.5, 3.96 / 2.5) @ N_SAMPLES
    rented_h100e = cloud_spend * 1e9 / (price_per_h100e_hour * 8760)
    msl_h100e = msl_owned + rented_h100e

    MODEL_STEPS["msl"] = [
        step("nvidia_owned", "Meta-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("amd_owned", "Meta-owned AMD Instinct fleet", amd_owned, "H100e", "input"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "nvidia_owned + amd_owned"),
        step("deployment_lag", "Operational share of owned", deployment_lag, "ratio", "input"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "total_owned × deployment_lag"),
        step("msl_share", "MSL share vs core-business recommenders", msl_share, "share", "input"),
        step("msl_owned", "MSL slice of the owned fleet", msl_owned, "H100e", "derived",
             "operational × msl_share"),
        step("cloud_spend", "Cloud rental spend run rate", cloud_spend, "USD B/yr", "input"),
        step("rental_price", "Rental price per H100e-hour", price_per_h100e_hour,
             "USD/H100e-hr", "input"),
        step("rented_h100e", "Rented cloud compute", rented_h100e, "H100e", "derived",
             "cloud_spend ÷ (rental_price × 8760 h)"),
        step("total_h100e", "MSL compute, end-2025", msl_h100e, "H100e", "final",
             "msl_owned + rented_h100e"),
    ]
    return msl_h100e


# ---------------------------------------------------------------------------
# 3. OpenAI (power-based model)
# ---------------------------------------------------------------------------
# Turns OpenAI's disclosed IT power per year into H100e via Microsoft's chip
# deployment mix. The fleet is built two ways from that mix -- "default"
# (vintage-layered: each year's added power keeps the mix deployed then) and
# "newest" (whole fleet on the latest year's mix) -- and new_chip_share blends
# them. Six sampled inputs: new_chip_share, deployment lag (which Microsoft
# snapshot each year reads), a power definition factor (IT vs gross), a
# figure-accuracy factor (is the internal number itself right), rounding jitter,
# and an IT overhead factor (server power -> IT power per GPU).

QUARTER_DAYS = 365.25 / 4
OAI_CHIP_TYPES = ["A100", "H100/H200", "B200", "B300"]


def _load_openai_data():
    """OpenAI's disclosed power, Microsoft's cumulative fleet, and per-chip specs."""
    owners_df = pd.read_csv(HERE / "data" / "nvidia_owners_cumulative_by_chip.csv")
    chip_power_df = pd.read_csv(HERE / "data" / "IT power by chip.csv")
    openai_df = pd.read_csv(HERE / "data" / "lab IT power.csv")
    openai_df["Date"] = pd.to_datetime(openai_df["Date"], format="%m/%d/%y")

    # IT watts per GPU. The power CSV names Blackwell "GB200"/"GB300"; the fleet
    # data calls them "B200"/"B300", so translate.
    rename = {"A100": "A100", "H100": "H100/H200", "GB200": "B200", "GB300": "B300"}
    # Median IT watts per GPU (drives the mix shares, which a shared overhead leaves
    # unchanged) and server power per GPU (scaled by the sampled overhead below).
    watts_per_gpu = {
        rename[r["Chip type"]]: r["IT power per GPU (W)"]
        for _, r in chip_power_df.iterrows()
        if r["Chip type"] in rename
    }
    server_power_per_gpu = {
        rename[r["Chip type"]]: r["Server power per GPU (W)"]
        for _, r in chip_power_df.iterrows()
        if r["Chip type"] in rename
    }

    microsoft = owners_df[
        (owners_df["Owner"] == "Microsoft") & (owners_df["Chip type"].isin(OAI_CHIP_TYPES))
    ].copy()
    microsoft["End date"] = pd.to_datetime(microsoft["End date"])
    ms_units = (microsoft.pivot_table(index="End date", columns="Chip type",
                                       values="Number of Units", aggfunc="first")
                .reindex(columns=OAI_CHIP_TYPES).fillna(0).sort_index())
    ms_h100e = (microsoft.pivot_table(index="End date", columns="Chip type",
                                       values="Compute estimate in H100e (median)", aggfunc="first")
                .reindex(columns=OAI_CHIP_TYPES).fillna(0).sort_index())

    dates = list(pd.to_datetime(openai_df["Date"].sort_values().unique()))
    last_date = dates[-1]

    # H100e per GPU is a (roughly constant) hardware ratio; read off the latest snapshot.
    h100e_per_gpu = {
        c: (ms_h100e.loc[last_date, c] / ms_units.loc[last_date, c])
        if ms_units.loc[last_date, c] > 0 else 0.0
        for c in OAI_CHIP_TYPES
    }
    # Microsoft's cumulative IT power (MW) by chip across every quarter.
    ms_power = (ms_units * pd.Series(watts_per_gpu) / 1e6).sort_index()

    # OpenAI's disclosed power per year, and the power added each year.
    disclosed = {d: float(p) for d, p in
                 zip(dates, openai_df.sort_values("Date")["Total IT power (MW)"])}
    added, prev = {}, 0.0
    for d in dates:
        added[d] = disclosed[d] - prev
        prev = disclosed[d]

    return dict(dates=dates, last_date=last_date, watts_per_gpu=watts_per_gpu,
                server_power_per_gpu=server_power_per_gpu,
                h100e_per_gpu=h100e_per_gpu, ms_power=ms_power,
                disclosed=disclosed, added=added)


def _chip_power_shares(data, lag_quarters):
    """For each OpenAI year, the share of power on each chip under a deployment lag.
    Returns (default_shares, newest_shares) as {date: {chip: share}}; lag_quarters
    may be a scalar or a per-sample array (shares come back the same shape)."""
    dates, ms_power = data["dates"], data["ms_power"]
    added_power = data["added"]
    start = ms_power.index[0]
    ms_day = np.array([(d - start).days for d in ms_power.index], dtype=float)
    lag_days = lag_quarters * QUARTER_DAYS

    # Microsoft's cumulative power by chip as seen `lag` quarters before each date,
    # interpolated to the (possibly fractional) day.
    cumulative = {
        d: {c: np.interp((d - start).days - lag_days, ms_day, ms_power[c].values)
            for c in OAI_CHIP_TYPES}
        for d in dates
    }
    # Mix of power Microsoft *added* each step (cumulative for the first step so
    # legacy A100s count; incremental afterward).
    added_mix = {}
    for i, d in enumerate(dates):
        if i == 0:
            add = {c: np.maximum(cumulative[d][c], 0.0) for c in OAI_CHIP_TYPES}
        else:
            prev = dates[i - 1]
            add = {c: np.maximum(cumulative[d][c] - cumulative[prev][c], 0.0)
                   for c in OAI_CHIP_TYPES}
        tot = sum(add.values())
        added_mix[d] = {c: add[c] / tot for c in OAI_CHIP_TYPES}

    # Default carries each year's additions forward at their own mix; newest puts
    # the whole fleet on the latest year's mix.
    default_shares, newest_shares = {}, {}
    carried = {c: 0.0 for c in OAI_CHIP_TYPES}
    for d in dates:
        for c in OAI_CHIP_TYPES:
            carried[c] = carried[c] + added_power[d] * added_mix[d][c]
        carried_tot = sum(carried.values())
        default_shares[d] = {c: carried[c] / carried_tot for c in OAI_CHIP_TYPES}
        newest_shares[d] = added_mix[d]
    return default_shares, newest_shares


def model_openai():
    """Returns the end-2025 total H100e samples (plus per-year totals for every
    disclosed year-end) and the per-chip specs and chip counts that the
    Anthropic model borrows."""
    sq.set_seed(42)
    data = _load_openai_data()
    watts, h100e_per_gpu = data["watts_per_gpu"], data["h100e_per_gpu"]
    server_ppg = data["server_power_per_gpu"]
    dates, last_date = data["dates"], data["last_date"]

    # Sampled inputs. new_chip_share blends the vintage-layered mix toward the
    # newest year's mix; the lag shifts which Microsoft snapshot each year reads.
    _params = load_lab_params()
    P = _params["openai"]
    new_chip_share = P["new_chip_share"] @ N_SAMPLES
    lag_quarters = P["lag_quarters"] @ N_SAMPLES
    # Watts per GPU = server power x a shared IT overhead factor (server -> IT power);
    # a higher overhead means fewer chips per disclosed MW. Its low end is the upside.
    it_overhead = _params["chip_specs"]["nvidia_it_overhead"] @ N_SAMPLES

    # Total power per year: one shared power-definition factor times each
    # disclosure, plus independent rounding jitter. Is each disclosed figure IT
    # power or gross? With probability p_gross it's gross and gets divided by a
    # datacenter PUE (a downward haircut); otherwise it's already IT.
    if_gross_power = 1 / P["gross_pue"]
    definition_factor = sq.mixture([P["if_it_power"], if_gross_power],
                                   [1 - P["p_gross"], P["p_gross"]]) @ N_SAMPLES
    # Is OpenAI's internal figure itself right, aside from rounding and IT-vs-gross?
    # Undercounted providers push up, overstatement down (median ~1.04).
    accuracy_factor = P["figure_accuracy"] @ N_SAMPLES
    # Rounding jitter: the disclosures' 0.1 GW rounding half-step, triangular
    # (edges less likely).
    rounding_mw = P["rounding_mw"]
    total_power = {
        d: (data["disclosed"][d] + (sq.triangular(-rounding_mw, 0.0, rounding_mw) @ N_SAMPLES))
        * definition_factor * accuracy_factor
        for d in dates
    }

    default_shares, newest_shares = _chip_power_shares(data, lag_quarters)
    # Chip counts at every disclosed year-end, not just the latest: each year
    # blends that year's two mixes and sizes the result by that year's power.
    counts_by_date = {}
    for d in dates:
        counts_by_date[d] = {}
        for c in OAI_CHIP_TYPES:
            share = (1 - new_chip_share) * default_shares[d][c] + new_chip_share * newest_shares[d][c]
            megawatts = total_power[d] * share
            counts_by_date[d][c] = megawatts * 1e6 / (server_ppg[c] * it_overhead)
    counts = counts_by_date[last_date]
    total_h100e_by_date = {
        d: sum(counts_by_date[d][c] * h100e_per_gpu[c] for c in OAI_CHIP_TYPES)
        for d in dates
    }

    MODEL_STEPS["openai"] = [
        step("disclosed_power", "Disclosed end-2025 power", data["disclosed"][last_date],
             "MW", "constant"),
        step("definition_factor", "Power-definition factor (IT vs gross)", definition_factor,
             "ratio", "input", "mixture(if_it_power, 1 / gross_pue; p_gross)"),
        step("accuracy_factor", "Figure-accuracy factor", accuracy_factor, "ratio", "input"),
        step("total_power", "Modelled end-2025 IT power", total_power[last_date],
             "MW", "derived",
             "(disclosed_power + rounding jitter) × definition_factor × accuracy_factor"),
        step("new_chip_share", "Fleet share on the newest chip mix", new_chip_share,
             "share", "input"),
        step("lag_quarters", "Deployment lag behind Microsoft's mix", lag_quarters,
             "quarters", "input"),
        step("it_overhead", "Server-to-IT power overhead", it_overhead, "ratio", "input"),
    ] + [
        step(c.lower().replace("/", "_").replace(" ", "_") + "_count", f"{c} chips",
             counts[c], "chips", "derived",
             "total_power × mix share ÷ (server watts × it_overhead)")
        for c in OAI_CHIP_TYPES
    ] + [
        step("total_h100e", "OpenAI compute, end-2025", total_h100e_by_date[last_date],
             "H100e", "final", "Σ chip count × H100e per chip"),
    ]

    return dict(total_h100e=total_h100e_by_date[last_date],
                total_h100e_by_date=total_h100e_by_date, watts_per_gpu=watts,
                h100e_per_gpu=h100e_per_gpu, counts=counts, last_date=last_date)


# ---------------------------------------------------------------------------
# 4. Anthropic (power-based model)
# ---------------------------------------------------------------------------
# Anthropic H100e = total_power_mw x blended_H100e_per_mw, where the blend is set
# by the Trainium2 share of power. At a fixed power budget the Nvidia mix and the
# TPU mix buy about the same H100e per watt, while Trainium2 buys ~0.75x as much
# (per the New Carlisle equivalency in the sheet's chip_specs rows); so the fleet
# collapses to two buckets and the Trainium2 power share is the lever.
# Nvidia specs and the H100:Blackwell ratio are borrowed from the OpenAI model.

# Shared hardware constants from the params sheet's chip_specs rows: TPU TDPs,
# the IT-power overhead, and the supplied Trainium2 equivalency (a fleet worth a
# known H100e draws a known IT power, which fixes the watts per chip).
CHIP_SPECS = load_lab_params()["chip_specs"]

TRAINIUM2_H100E = 1299 / 1979  # Trainium2 dense 8-bit throughput relative to an H100
TRAINIUM2_IT_WATTS = TRAINIUM2_H100E / (
    CHIP_SPECS["trainium2_ref_h100e"] / CHIP_SPECS["trainium2_ref_it_mw"]) * 1e6
IT_OVERHEAD = CHIP_SPECS["tpu_it_overhead"]  # IT power per chip / TDP, for TPUs (no public server specs)


def _tpu_mix_per_mw():
    """Google's real v5+ TPU fleet efficiency (H100e per MW), scored on native
    8-bit peak, weighted by chip count x IT power -- the OpenAI methodology."""
    tpu_tdp_w = {
        "TPU v5e": CHIP_SPECS["tpu_v5e_tdp"],
        "TPU v5p": CHIP_SPECS["tpu_v5p_tdp"],
        "TPU v6e": CHIP_SPECS["tpu_v6e_tdp"],
        "TPU v7": CHIP_SPECS["tpu_v7_tdp"],
    }
    tpu_8bit = {"TPU v5e": 3.93e14, "TPU v5p": 9.18e14, "TPU v6e": 1.836e15, "TPU v7": 4.614e15}
    it_watts = {c: tpu_tdp_w[c] * IT_OVERHEAD for c in tpu_tdp_w}
    h100e_per_chip = {c: tpu_8bit[c] / H100_FLOPS for c in tpu_tdp_w}

    df = pd.read_csv(ROOT / "csv_export" / "tpu_cumulative_by_chip.csv")
    df["End date"] = pd.to_datetime(df["End date"])
    snap = df[df["End date"] == pd.Timestamp("2025-12-31")]
    units = {c: float(snap.loc[snap["Chip type"] == c, "Number of units (median)"].iloc[0])
             for c in tpu_tdp_w}
    total_it_mw = sum(units[c] * it_watts[c] / 1e6 for c in tpu_tdp_w)
    return sum(units[c] * h100e_per_chip[c] for c in tpu_tdp_w) / total_it_mw


def model_anthropic(openai_result):
    sq.set_seed(42)
    watts = openai_result["watts_per_gpu"]
    h100e_per_gpu = openai_result["h100e_per_gpu"]
    counts = openai_result["counts"]

    # OpenAI's end-2025 Hopper:Blackwell count ratio (A100 dropped, GB300 folded in).
    hopper = float(np.median(counts["H100/H200"]))
    blackwell = float(np.median(counts["B200"] + counts["B300"]))
    h100_count_fraction = hopper / (hopper + blackwell)

    # Per-chip H100e per MW for the modelled chips.
    h100e_per_mw = {
        "H100": h100e_per_gpu["H100/H200"] / watts["H100/H200"] * 1e6,
        "GB200": h100e_per_gpu["B200"] / watts["B200"] * 1e6,
        "Trainium2": TRAINIUM2_H100E / TRAINIUM2_IT_WATTS * 1e6,
    }
    # Nvidia mix: H100 + GB200 in OpenAI's count ratio, weighted by watts per chip.
    h100_per_gb200 = h100_count_fraction / (1 - h100_count_fraction)
    h100_power_share = (h100_per_gb200 * watts["H100/H200"]) / (
        h100_per_gb200 * watts["H100/H200"] + watts["B200"])
    nvidia_mix_per_mw = (h100_power_share * h100e_per_mw["H100"]
                         + (1 - h100_power_share) * h100e_per_mw["GB200"])

    # Nvidia and TPU are within a few percent, so treat non-Trainium as one bucket.
    nontrainium_per_mw = (nvidia_mix_per_mw + _tpu_mix_per_mw()) / 2
    trainium2_per_mw = h100e_per_mw["Trainium2"]

    # Total power: OpenAI-memo mainline 1.4 GW, upper bound ~1.8 GW (just under
    # OpenAI's 1.9), lower bound set so the lognormal median lands on 1.4 GW.
    P = load_lab_params()["anthropic"]
    power_mw = (P["lab_power_gw"] @ N_SAMPLES) * 1000.0

    # Trainium2 share of IT power: ~normal, median ~0.52, anchored on New Carlisle
    # + Madison site power (Epoch's data-center directory) and ceiling-checked
    # against Amazon's ~1.4M deployed chips. Sampled independently of power.
    trainium_share = P["trainium_share"] @ N_SAMPLES

    blended_per_mw = trainium_share * trainium2_per_mw + (1 - trainium_share) * nontrainium_per_mw
    anthropic_h100e = power_mw * blended_per_mw

    MODEL_STEPS["anthropic"] = [
        step("power_mw", "Total IT power", power_mw, "MW", "input", "lab_power_gw × 1000"),
        step("trainium_share", "Trainium2 share of IT power", trainium_share, "share", "input"),
        step("trainium2_per_mw", "Trainium2 fleet efficiency", trainium2_per_mw,
             "H100e/MW", "constant"),
        step("nontrainium_per_mw", "Nvidia + TPU fleet efficiency", nontrainium_per_mw,
             "H100e/MW", "constant"),
        step("blended_per_mw", "Blended fleet efficiency", blended_per_mw, "H100e/MW", "derived",
             "trainium_share × trainium2_per_mw + (1 − trainium_share) × nontrainium_per_mw"),
        step("total_h100e", "Anthropic compute, end-2025", anthropic_h100e, "H100e", "final",
             "power_mw × blended_per_mw"),
    ]
    return anthropic_h100e


# ---------------------------------------------------------------------------
# 5. End-2024 backcasts: Google DeepMind and Meta AI (pre-MSL)
# ---------------------------------------------------------------------------
# Same top-down shape as the end-2025 models (owned fleet x operational ratio x
# lab share), promoted from the lab_2024_backcasts notebook, with two changes
# in how the first two factors are obtained:
#
#  - Owned fleets are read from the quarterly dashboard CSVs at end-2024, as
#    lognormals through the summed per-chip 5th/95th columns. (Summing per-chip
#    percentile bounds treats chips as perfectly correlated, so the CIs are on
#    the generous side -- the convention the end-2025 sheet rows effectively
#    used.)
#  - The operational/owned ratio is computed from the owned-stock trajectory
#    under a sampled deployment lag instead of hand-derived: fleets grew
#    ~3.5-4.5x during 2024, so the 2025 ratios would overstate early years.
#
# "The lab" in 2024 means frontier-AI compute at the company: MSL did not exist
# (its predecessor was Meta AI / GenAI plus FAIR), and the share priors are for
# those predecessor scopes.

END_2024 = pd.Timestamp("2024-12-31")
OWNERS_CSV = HERE / "data" / "nvidia_owners_cumulative_by_chip.csv"
TPU_CSV = ROOT / "csv_export" / "tpu_cumulative_by_chip.csv"
AMD_CSV = ROOT / "csv_export" / "amd_cumulative_by_chip.csv"


def owner_quarterly_h100e_medians(csv_path, owner=None):
    """Quarterly cumulative H100e medians (summed across chip types) as a
    Series indexed by quarter-end date; optionally one owner's slice."""
    df = pd.read_csv(csv_path)
    if owner is not None:
        df = df[df["Owner"] == owner]
    df["End date"] = pd.to_datetime(df["End date"])
    return df.groupby("End date")["Compute estimate in H100e (median)"].sum()


def end_2024_fleet_dist(csv_path, owner=None):
    """Owned-fleet H100e at end-2024, as a lognormal through the dashboard's
    summed per-chip 5th/95th columns. The owners CSV and the TPU/AMD exports
    name those columns differently."""
    df = pd.read_csv(csv_path)
    if owner is not None:
        df = df[df["Owner"] == owner]
        lo_col, hi_col = "H100e (5th percentile)", "H100e (95th percentile)"
    else:
        lo_col, hi_col = ("Compute estimate in H100e (5th percentile)",
                          "Compute estimate in H100e (95th percentile)")
    df["End date"] = pd.to_datetime(df["End date"])
    snap = df[df["End date"] == END_2024]
    return sq.to(snap[lo_col].sum(), snap[hi_col].sum())


def operational_ratio_2024(stock_series, lag_quarters):
    """Owned stock `lag_quarters` before end-2024, as a fraction of the
    end-2024 stock, interpolated along the quarterly median trajectory. The
    trajectory's shape is treated as data; the stock's level uncertainty is
    sampled separately (the fleet lognormals)."""
    window = stock_series.loc["2023-12-31":END_2024]
    days = np.array([(d - window.index[0]).days for d in window.index], dtype=float)
    target = days[-1] - lag_quarters * QUARTER_DAYS
    return np.interp(target, days, window.values) / window.values[-1]


def model_deepmind_2024():
    sq.set_seed(42)
    P = load_lab_params()["deepmind"]
    nvidia_owned = end_2024_fleet_dist(OWNERS_CSV, "Google") @ N_SAMPLES
    google_owned = end_2024_fleet_dist(TPU_CSV) @ N_SAMPLES  # TPU fleet
    total_owned = nvidia_owned + google_owned

    # Operational share of owned: sample the install lag, then read the owned
    # stock that many quarters before end-2024 off the trajectory.
    lag_quarters = P["lag_quarters_2024"] @ N_SAMPLES
    stock = (owner_quarterly_h100e_medians(OWNERS_CSV, "Google")
             + owner_quarterly_h100e_medians(TPU_CSV)).dropna()
    deployment_lag = operational_ratio_2024(stock, lag_quarters)
    operational = total_owned * deployment_lag

    # One overall DeepMind share: Google gave no cloud/internal split for 2024,
    # so the 2025 model's two-sub-share blend has nothing to anchor on.
    dm_share = P["dm_share_2024"] @ N_SAMPLES
    dm_h100e = operational * dm_share

    MODEL_STEPS["deepmind_2024"] = [
        step("nvidia_owned", "Google-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("google_owned", "Google TPU fleet", google_owned, "H100e", "input"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "nvidia_owned + google_owned"),
        step("lag_quarters_2024", "Deployment lag", lag_quarters, "quarters", "input"),
        step("deployment_lag", "Operational share of owned", deployment_lag, "ratio", "derived",
             "owned stock lag_quarters_2024 before end-2024 ÷ end-2024 stock"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "total_owned × deployment_lag"),
        step("dm_share_2024", "DeepMind share of Google ML compute", dm_share, "share", "input"),
        step("total_h100e", "DeepMind compute, end-2024", dm_h100e, "H100e", "final",
             "operational × dm_share_2024"),
    ]
    return dm_h100e


def model_msl_2024():
    """Meta AI / GenAI (the pre-MSL frontier org) at end-2024. Owned fleet =
    Meta's Nvidia GPUs plus a sampled slice of the all-owner AMD Instinct
    fleet (the dashboards don't split AMD by owner); MTIA excluded."""
    sq.set_seed(42)
    P = load_lab_params()["msl"]
    nvidia_owned = end_2024_fleet_dist(OWNERS_CSV, "Meta") @ N_SAMPLES
    amd_all_owners = end_2024_fleet_dist(AMD_CSV) @ N_SAMPLES
    meta_amd_share = P["meta_amd_share_2024"] @ N_SAMPLES
    amd_owned = amd_all_owners * meta_amd_share
    total_owned = nvidia_owned + amd_owned

    # Operational share of owned, as in the DeepMind backcast. The trajectory
    # uses the median AMD share; only the level uncertainty is sampled.
    lag_quarters = P["lag_quarters_2024"] @ N_SAMPLES
    nvidia_stock = owner_quarterly_h100e_medians(OWNERS_CSV, "Meta")
    amd_stock = (owner_quarterly_h100e_medians(AMD_CSV)
                 .reindex(nvidia_stock.index).fillna(0.0))
    stock = (nvidia_stock + float(np.median(meta_amd_share)) * amd_stock).dropna()
    deployment_lag = operational_ratio_2024(stock, lag_quarters)
    operational = total_owned * deployment_lag

    # Frontier vs core-business split for the predecessor org, see the sheet.
    meta_ai_share = P["meta_ai_share_2024"] @ N_SAMPLES
    meta_h100e = operational * meta_ai_share

    MODEL_STEPS["msl_2024"] = [
        step("nvidia_owned", "Meta-owned Nvidia fleet", nvidia_owned, "H100e", "input"),
        step("amd_all_owners", "AMD Instinct fleet, all owners", amd_all_owners,
             "H100e", "input"),
        step("meta_amd_share_2024", "Meta share of the AMD fleet", meta_amd_share,
             "share", "input"),
        step("amd_owned", "Meta-owned AMD Instinct fleet", amd_owned, "H100e", "derived",
             "amd_all_owners × meta_amd_share_2024"),
        step("total_owned", "Total owned fleet", total_owned, "H100e", "derived",
             "nvidia_owned + amd_owned"),
        step("lag_quarters_2024", "Deployment lag", lag_quarters, "quarters", "input"),
        step("deployment_lag", "Operational share of owned", deployment_lag, "ratio", "derived",
             "owned stock lag_quarters_2024 before end-2024 ÷ end-2024 stock"),
        step("operational", "Operational fleet", operational, "H100e", "derived",
             "total_owned × deployment_lag"),
        step("meta_ai_share_2024", "Meta AI (pre-MSL) frontier share", meta_ai_share,
             "share", "input"),
        step("total_h100e", "Meta AI frontier compute, end-2024", meta_h100e, "H100e", "final",
             "operational × meta_ai_share_2024"),
    ]
    return meta_h100e


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def main():
    openai_result = model_openai()
    labs = {
        "Google DeepMind": model_deepmind(),
        "Meta SL": model_msl(),
        "OpenAI": openai_result["total_h100e"],
        "Anthropic": model_anthropic(openai_result),
    }

    print("End-2025 compute, H100-equivalents (5th / median / 95th):\n")
    print(f"  {'Lab':<18}{'5th':>10}{'median':>10}{'95th':>10}")
    for name, samples in labs.items():
        lo, mid, hi = pctiles(samples)
        print(f"  {name:<18}{fmt(lo):>10}{fmt(mid):>10}{fmt(hi):>10}")

    openai_2024 = next(arr for d, arr in openai_result["total_h100e_by_date"].items()
                       if d.year == 2024)
    backcasts = {
        "Google DeepMind": model_deepmind_2024(),
        "Meta AI (pre-MSL)": model_msl_2024(),
        "OpenAI": openai_2024,
    }
    print("\nEnd-2024 backcasts (Anthropic's lives in anthropic_2024_backcast):\n")
    print(f"  {'Lab':<18}{'5th':>10}{'median':>10}{'95th':>10}")
    for name, samples in backcasts.items():
        lo, mid, hi = pctiles(samples)
        print(f"  {name:<18}{fmt(lo):>10}{fmt(mid):>10}{fmt(hi):>10}")

    # One comparison chart: median bar per lab with a 90% CI error bar.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.13)
    colors = {"Google DeepMind": "#2B8C86", "Meta SL": "#2B6CB8",
              "OpenAI": "#1a73e8", "Anthropic": "#e8710a"}
    names = list(labs)
    highest = 0.0
    for i, name in enumerate(names):
        y = len(names) - 1 - i
        lo, mid, hi = pctiles(labs[name])
        highest = max(highest, hi / 1e6)
        ax.barh(y, mid / 1e6, height=0.5, color=colors[name], alpha=0.85)
        ax.errorbar(mid / 1e6, y, xerr=[[(mid - lo) / 1e6], [(hi - mid) / 1e6]],
                    fmt="none", ecolor="#333333", elinewidth=1.5, capsize=5, capthick=1.5)
        ax.text(hi / 1e6 + 0.04, y, f"{fmt(mid)}  (90% CI {fmt(lo)}-{fmt(hi)})",
                va="center", fontsize=9, color="#444444")
    ax.set_xlim(0, highest * 1.55)  # headroom so the annotations don't clip
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_xlabel("End-2025 compute (H100e, millions)")
    ax.set_title("Frontier-lab compute at end-2025", loc="left", weight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.show()

    return labs


if __name__ == "__main__":
    main()
