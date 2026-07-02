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


# ---------------------------------------------------------------------------
# 1. Google DeepMind
# ---------------------------------------------------------------------------
# DeepMind H100e = total_owned x deployment_lag x DeepMind_fraction, where the
# DeepMind fraction blends its share of Google's cloud half and its share of the
# internal (non-cloud) half:
#   fraction = cloud_share * dm_cloud_share + (1 - cloud_share) * dm_noncloud_share
# Owned fleet = Nvidia GPUs + Google TPUs (drawn independently).

def model_deepmind():
    sq.set_seed(42)
    P = load_lab_params()["deepmind"]
    nvidia_owned = P["nvidia_owned"] @ N_SAMPLES
    google_owned = P["google_owned"] @ N_SAMPLES  # TPU fleet
    total_owned = nvidia_owned + google_owned

    # Only part of the owned fleet is online at any moment.
    operational = total_owned * (P["deployment_lag"] @ N_SAMPLES)

    # CFO's "around half" cloud vs internal split, and DeepMind's slice of each.
    cloud_share = P["cloud_share"] @ N_SAMPLES
    dm_cloud_share = P["dm_cloud_share"] @ N_SAMPLES      # enterprise Gemini + external rentals
    dm_noncloud_share = P["dm_noncloud_share"] @ N_SAMPLES  # consumer Gemini + DM R&D
    dm_fraction = cloud_share * dm_cloud_share + (1 - cloud_share) * dm_noncloud_share

    return operational * dm_fraction


# ---------------------------------------------------------------------------
# 2. Meta Superintelligence Labs (MSL)
# ---------------------------------------------------------------------------
# MSL H100e = total_owned x deployment_lag x MSL_share. Meta's fleet is almost
# entirely internal at end-2025, so there is no cloud/non-cloud split: the MSL
# share is sampled directly (frontier AI work vs core-business recommenders).
# Owned fleet = Nvidia GPUs + AMD Instinct (drawn independently); MTIA excluded.

def model_msl():
    sq.set_seed(42)
    P = load_lab_params()["msl"]
    nvidia_owned = P["nvidia_owned"] @ N_SAMPLES
    amd_owned = P["amd_owned"] @ N_SAMPLES
    total_owned = nvidia_owned + amd_owned

    # Slightly higher lag ratios than DeepMind (Meta's fleet ramped fast in 2025).
    operational = total_owned * (P["deployment_lag"] @ N_SAMPLES)

    # Frontier vs core-business split — highly uncertain, see the sheet's notes.
    msl_share = P["msl_share"] @ N_SAMPLES

    return operational * msl_share


# ---------------------------------------------------------------------------
# 3. OpenAI (power-based model)
# ---------------------------------------------------------------------------
# Turns OpenAI's disclosed IT power per year into H100e via Microsoft's chip
# deployment mix. The fleet is built two ways from that mix -- "default"
# (vintage-layered: each year's added power keeps the mix deployed then) and
# "newest" (whole fleet on the latest year's mix) -- and new_chip_share blends
# them. Four sampled inputs: new_chip_share, deployment lag (which Microsoft
# snapshot each year reads), a power definition factor (IT vs gross), and rounding jitter.

QUARTER_DAYS = 365.25 / 4
OAI_CHIP_TYPES = ["A100", "H100/H200", "B200", "B300"]


def _load_openai_data():
    """OpenAI's disclosed power, Microsoft's cumulative fleet, and per-chip specs."""
    owners_df = pd.read_csv(HERE / "nvidia_owners_cumulative_by_chip.csv")
    chip_power_df = pd.read_csv(HERE / "IT power by chip.csv")
    openai_df = pd.read_csv(HERE / "lab IT power.csv")
    openai_df["Date"] = pd.to_datetime(openai_df["Date"], format="%m/%d/%y")

    # IT watts per GPU. The power CSV names Blackwell "GB200"/"GB300"; the fleet
    # data calls them "B200"/"B300", so translate.
    rename = {"A100": "A100", "H100": "H100/H200", "GB200": "B200", "GB300": "B300"}
    watts_per_gpu = {
        rename[r["Chip type"]]: r["IT power per GPU (W)"]
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
    """Returns the end-2025 total H100e samples plus the per-chip specs and chip
    counts that the Anthropic model borrows."""
    sq.set_seed(42)
    data = _load_openai_data()
    watts, h100e_per_gpu = data["watts_per_gpu"], data["h100e_per_gpu"]
    dates, last_date = data["dates"], data["last_date"]

    # Sampled inputs. new_chip_share blends the vintage-layered mix toward the
    # newest year's mix; the lag shifts which Microsoft snapshot each year reads.
    P = load_lab_params()["openai"]
    new_chip_share = P["new_chip_share"] @ N_SAMPLES
    lag_quarters = P["lag_quarters"] @ N_SAMPLES

    # Total power per year: one shared power-definition factor times each
    # disclosure, plus independent rounding jitter. Is each disclosed figure IT
    # power or gross? With probability p_gross it's gross and gets divided by a
    # datacenter PUE (a downward haircut); otherwise it's already IT.
    if_gross_power = 1 / P["gross_pue"]
    definition_factor = sq.mixture([P["if_it_power"], if_gross_power],
                                   [1 - P["p_gross"], P["p_gross"]]) @ N_SAMPLES
    # Rounding jitter: nearest 0.1 GW => within +/-50 MW, triangular (edges less likely).
    total_power = {
        d: (data["disclosed"][d] + (sq.triangular(-50.0, 0.0, 50.0) @ N_SAMPLES)) * definition_factor
        for d in dates
    }

    default_shares, newest_shares = _chip_power_shares(data, lag_quarters)
    counts = {}
    for c in OAI_CHIP_TYPES:
        share = (1 - new_chip_share) * default_shares[last_date][c] + new_chip_share * newest_shares[last_date][c]
        megawatts = total_power[last_date] * share
        counts[c] = megawatts * 1e6 / watts[c]
    total_h100e = sum(counts[c] * h100e_per_gpu[c] for c in OAI_CHIP_TYPES)

    return dict(total_h100e=total_h100e, watts_per_gpu=watts,
                h100e_per_gpu=h100e_per_gpu, counts=counts, last_date=last_date)


# ---------------------------------------------------------------------------
# 4. Anthropic (power-based model)
# ---------------------------------------------------------------------------
# Anthropic H100e = total_power_mw x blended_H100e_per_mw, where the blend is set
# by the Trainium2 share of power. At a fixed power budget the Nvidia mix and the
# TPU mix buy about the same H100e per watt, while Trainium2 buys ~0.6x as much;
# so the fleet collapses to two buckets and the Trainium2 power share is the lever.
# Nvidia specs and the H100:Blackwell ratio are borrowed from the OpenAI model.

TRAINIUM2_H100E = 1299 / 1979  # Trainium2 dense 8-bit throughput relative to an H100
TRAINIUM2_REF_H100E = 300e3    # supplied equivalency: a 300k-H100e Trainium2 fleet...
TRAINIUM2_REF_IT_MW = 478.0    # ...draws 478 MW of IT power, which fixes watts/chip
TRAINIUM2_IT_WATTS = TRAINIUM2_H100E / (TRAINIUM2_REF_H100E / TRAINIUM2_REF_IT_MW) * 1e6
IT_OVERHEAD = 1.742            # IT power per chip / TDP, for TPUs (no public server specs)


def _tpu_mix_per_mw():
    """Google's real v5+ TPU fleet efficiency (H100e per MW), scored on native
    8-bit peak, weighted by chip count x IT power -- the OpenAI methodology."""
    tpu_tdp_w = {"TPU v5e": 225, "TPU v5p": 540, "TPU v6e": 380, "TPU v7": 960}
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

    # Trainium2 share of IT power: ~normal, median ~0.6, anchored on Rainier's
    # ~700k chips and Amazon's 1.4M-chip ceiling. Sampled independently of power.
    trainium_share = P["trainium_share"] @ N_SAMPLES

    blended_per_mw = trainium_share * trainium2_per_mw + (1 - trainium_share) * nontrainium_per_mw
    return power_mw * blended_per_mw


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
