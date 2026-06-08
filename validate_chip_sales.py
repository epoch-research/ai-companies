#!/usr/bin/env python3
"""Validate local AI chip-sales model outputs against Epoch's published website data.

The notebooks in this repo (e.g. nvidia_estimates.ipynb) run a Monte Carlo model and write
per-chip and cumulative estimates to csv_export/. Those CSVs are what ultimately feed the
public dataset at https://epoch.ai/data/ai-chip-sales. This script downloads the currently
published dataset and compares it, cell by cell, against the local CSVs so we can catch
drift, stale uploads, or accidental changes before/after publishing.

For each designer it lines up three published tables with their local counterparts:

    Published (in the zip)                local CSV (csv_export/)
    ------------------------------------  --------------------------------------------
    timelines_by_chip                     <designer>_calendar_quarter_chip_timelines.csv
    cumulative_timelines                  <designer>_cumulative_by_chip.csv
    cumulative_timelines_by_designer      <designer>_cumulative_totals.csv

What it checks per table:
  1. Row coverage  - rows present locally but not on the website, and vice versa.
  2. Per-cell diff - percent difference for every numeric metric (units, H100e, power),
                     with each cell classified ok / minor / major against tolerances.
  3. Timestamp     - the "Estimates generated on" stamp in each file, to explain diffs
                     (e.g. the website table was generated from an older model run).
  4. Visuals       - local-vs-website scatter, percent-diff distributions, cumulative
                     time series, and a per-quarter heatmap (written as PNGs).
  5. Artifacts     - detailed diff CSVs and a markdown summary in the output directory.

Only Nvidia is wired up today. To extend to Google (TPU), Amazon (Trainium), AMD, etc.,
add an entry to DESIGNER_CONFIGS: the three published tables share the same layout across
designers, so only the local file paths, the website "Chip manufacturer" label, and any
chip-name aliases need to be supplied.

Usage:
    python validate_chip_sales.py                 # validate Nvidia, download fresh data
    python validate_chip_sales.py --designer all  # validate every configured designer
    python validate_chip_sales.py --no-download    # reuse previously downloaded data
    python validate_chip_sales.py --data-dir DIR   # use already-extracted website CSVs
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
import urllib.request
import urllib.error
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt


WEBSITE_ZIP_URL = "https://epoch.ai/data/ai_chip_sales.zip"

# Cells whose percent difference is within TOL_PCT are considered a match; between TOL_PCT
# and WARN_PCT is "minor" drift; above WARN_PCT is "major" drift worth investigating.
DEFAULT_TOL_PCT = 1.0
DEFAULT_WARN_PCT = 10.0


# ---------------------------------------------------------------------------
# Table layouts (shared across all designers)
# ---------------------------------------------------------------------------
# Each entry describes one published table, the local file role it maps to, whether rows
# are keyed by chip type, and the list of metrics to compare. Each metric is a triple of
# (local column name, website column name, short label). The column names differ between
# the two sides in places - most notably the cumulative-by-chip H100e columns - so they
# are spelled out explicitly rather than assumed equal.
TABLE_SPECS = {
    "timelines_by_chip": {
        "title": "Quarterly timelines by chip",
        "web_file": "timelines_by_chip.csv",
        "has_chip": True,
        "metrics": [
            ("Number of Units", "Number of Units", "Units median"),
            ("Number of Units (5th percentile)", "Number of Units (5th percentile)", "Units p5"),
            ("Number of Units (95th percentile)", "Number of Units (95th percentile)", "Units p95"),
            ("Compute estimate in H100e (median)", "Compute estimate in H100e (median)", "H100e median"),
            ("H100e (5th percentile)", "H100e (5th percentile)", "H100e p5"),
            ("H100e (95th percentile)", "H100e (95th percentile)", "H100e p95"),
        ],
    },
    "cumulative_by_chip": {
        "title": "Cumulative timelines by chip",
        "web_file": "cumulative_timelines.csv",
        "has_chip": True,
        "metrics": [
            ("Number of units (median)", "Number of units (median)", "Units median"),
            ("Number of units (5th percentile)", "Number of units (5th percentile)", "Units p5"),
            ("Number of units (95th percentile)", "Number of units (95th percentile)", "Units p95"),
            ("Compute estimate in H100e (median)", "H100e compute power (median)", "H100e median"),
            ("Compute estimate in H100e (5th percentile)", "H100e compute power (5th percentile)", "H100e p5"),
            ("Compute estimate in H100e (95th percentile)", "H100e compute power (95th percentile)", "H100e p95"),
        ],
    },
    "cumulative_by_designer": {
        "title": "Cumulative totals by designer",
        "web_file": "cumulative_timelines_by_designer.csv",
        "has_chip": False,
        "metrics": [
            ("Number of units (median)", "Number of units (median)", "Units median"),
            ("Number of units (5th percentile)", "Number of units (5th percentile)", "Units p5"),
            ("Number of units (95th percentile)", "Number of units (95th percentile)", "Units p95"),
            ("Compute estimate in H100e (median)", "Compute estimate in H100e (median)", "H100e median"),
            ("Compute estimate in H100e (5th percentile)", "Compute estimate in H100e (5th percentile)", "H100e p5"),
            ("Compute estimate in H100e (95th percentile)", "Compute estimate in H100e (95th percentile)", "H100e p95"),
            ("Power in MW (median)", "Power in MW (median)", "Power MW median"),
            ("Power in MW (5th percentile)", "Power in MW (5th percentile)", "Power MW p5"),
            ("Power in MW (95th percentile)", "Power in MW (95th percentile)", "Power MW p95"),
        ],
    },
}

# Logical metric groups, used purely for coloring/structuring the visualizations.
METRIC_GROUPS = {"Units": "Units", "H100e": "H100e", "Power": "Power"}


# ---------------------------------------------------------------------------
# Per-designer wiring
# ---------------------------------------------------------------------------
# To add a designer, copy the Nvidia block and fill in:
#   web_name      - the label used in the website's "Chip manufacturer" column
#   chip_aliases  - any chip names that differ between website and local exports
#   local_files   - the three local CSVs (see csv_export/ for the naming conventions)
DESIGNER_CONFIGS = {
    "nvidia": {
        "web_name": "Nvidia",
        # Chip naming isn't yet consistent across published files: the per-quarter table and
        # the local exports use "H100/H200", but the published cumulative_timelines.csv still
        # says "H100". Normalize both sides so rows line up regardless.
        "chip_aliases": {"H100/H200": "H100"},
        "local_files": {
            "timelines_by_chip": "csv_export/nvidia_calendar_quarter_chip_timelines.csv",
            "cumulative_by_chip": "csv_export/nvidia_cumulative_by_chip.csv",
            "cumulative_by_designer": "csv_export/nvidia_cumulative_totals.csv",
        },
    },
    # Example for a future designer (paths are illustrative, uncomment + verify to enable):
    # "google": {
    #     "web_name": "Google",
    #     "chip_aliases": {},
    #     "local_files": {
    #         "timelines_by_chip": "csv_export/tpu_calendar_quarter_chip_timelines.csv",
    #         "cumulative_by_chip": "csv_export/tpu_cumulative_by_chip.csv",
    #         "cumulative_by_designer": "csv_export/tpu_cumulative_totals.csv",
    #     },
    # },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_TS_RE = re.compile(r"Estimates generated on:\s*([0-9]{1,2}-[0-9]{1,2}-[0-9]{4} [0-9]{1,2}:[0-9]{2})")


def _flatten_key(key_tuple):
    """Join a tuple row key into a single string for use as a flat DataFrame index."""
    return "\x1f".join(str(p) for p in key_tuple)


def _to_float(value):
    """Coerce a cell to float, treating blanks/None/non-numeric as NaN."""
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _iso_dates(series):
    """Parse a date column (either M/D/YYYY or YYYY-MM-DD) to canonical ISO strings."""
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def _quarter_label(iso_date):
    """Turn an ISO start date into a calendar-quarter label like 'Q1 2024'."""
    dt = pd.to_datetime(iso_date)
    return f"Q{(dt.month - 1) // 3 + 1} {dt.year}"


def extract_timestamps(notes_series):
    """Return a Counter of the 'Estimates generated on' stamps found in a Notes column."""
    stamps = []
    for note in notes_series.dropna():
        match = _TS_RE.search(str(note))
        if match:
            stamps.append(match.group(1))
    return Counter(stamps)


def _format_timestamps(counter):
    """Render a timestamp Counter as a compact human-readable string."""
    if not counter:
        return "(no timestamp found)"
    parts = [f"{ts} (x{n})" for ts, n in counter.most_common()]
    return ", ".join(parts)


def percent_diff(local_v, web_v):
    """Percent difference of local relative to the published website value.

    Returns NaN if either side is missing, 0 if both are exactly zero, and +/-inf if the
    website value is zero but the local value is not (an unbounded relative change).
    """
    if np.isnan(local_v) or np.isnan(web_v):
        return np.nan
    if web_v == 0:
        return 0.0 if local_v == 0 else np.inf
    return (local_v - web_v) / web_v * 100.0


def classify(pct, tol_pct, warn_pct):
    """Bucket a percent difference into ok / minor / major / nodata."""
    if np.isnan(pct):
        return "nodata"
    ap = abs(pct)
    if np.isinf(ap):
        return "major"
    if ap <= tol_pct:
        return "ok"
    if ap <= warn_pct:
        return "minor"
    return "major"


# ---------------------------------------------------------------------------
# Fetching the published data
# ---------------------------------------------------------------------------
def fetch_website_data(url, cache_dir, download=True):
    """Ensure the published CSVs are available in cache_dir, downloading if needed.

    Returns the path to the directory holding the extracted CSVs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    needed = [spec["web_file"] for spec in TABLE_SPECS.values()]
    have_all = all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)

    if download or not have_all:
        print(f"Downloading published dataset from {url} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "epoch-chip-validation/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SystemExit(
                f"Could not download {url}: {exc}\n"
                f"If you are offline, download the zip manually and pass its extracted folder "
                f"via --data-dir, or reuse a previous download with --no-download."
            )
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(cache_dir)
        print(f"Extracted {len(payload):,} bytes to {cache_dir}")
    else:
        print(f"Using cached published data in {cache_dir}")

    missing = [f for f in needed if not os.path.exists(os.path.join(cache_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Expected published files not found in {cache_dir}: {missing}. "
            f"Re-run without --no-download, or point --data-dir at a valid extraction."
        )
    return cache_dir


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------
def compare_table(local_path, web_path, spec, web_name, chip_aliases, tol_pct, warn_pct):
    """Compare one local CSV against its published counterpart.

    Returns a dict with the long-format per-cell diffs, the unmatched row keys on each
    side, the run timestamps, and bookkeeping needed for reporting and plots.
    """
    local = pd.read_csv(local_path)
    web = pd.read_csv(web_path)
    web = web[web["Chip manufacturer"] == web_name].copy()

    # Canonicalize dates and (optionally) chip names so the two sides line up on a key.
    for df in (local, web):
        df["_start"] = _iso_dates(df["Start date"])
        df["_end"] = _iso_dates(df["End date"])
    if spec["has_chip"]:
        local["_chip"] = local["Chip type"].map(lambda c: chip_aliases.get(str(c), str(c)))
        web["_chip"] = web["Chip type"].map(lambda c: chip_aliases.get(str(c), str(c)))
        key_cols = ["_chip", "_start", "_end"]
    else:
        local["_chip"] = "(total)"
        web["_chip"] = "(total)"
        key_cols = ["_start", "_end"]

    # Keep a tuple key for set math and display, and a flat string key for fast row lookup
    # (a tuple passed to .loc gets misread as multi-axis indexing on a plain index).
    local["_key"] = local[key_cols].apply(tuple, axis=1)
    web["_key"] = web[key_cols].apply(tuple, axis=1)
    local["_skey"] = local["_key"].map(_flatten_key)
    web["_skey"] = web["_key"].map(_flatten_key)

    # Guard against accidental duplicate keys so set_index lookups stay unambiguous.
    for label, df in (("local", local), ("website", web)):
        n_dup = int(df["_skey"].duplicated().sum())
        if n_dup:
            print(f"  warning: {n_dup} duplicate row key(s) in {label} {os.path.basename(web_path)}; keeping first")
    local = local.drop_duplicates("_skey")
    web = web.drop_duplicates("_skey")

    local_ts = extract_timestamps(local["Notes"]) if "Notes" in local.columns else Counter()
    web_ts = extract_timestamps(web["Notes"]) if "Notes" in web.columns else Counter()

    lkeys, wkeys = set(local["_key"]), set(web["_key"])
    matched = sorted(lkeys & wkeys)
    local_only = sorted(lkeys - wkeys)
    web_only = sorted(wkeys - lkeys)

    li = local.set_index("_skey")
    wi = web.set_index("_skey")

    # Skip any metric whose column is absent on either side (keeps the script robust as
    # the published schema evolves), and note what was skipped.
    usable_metrics, skipped_metrics = [], []
    for local_col, web_col, mlabel in spec["metrics"]:
        if local_col in li.columns and web_col in wi.columns:
            usable_metrics.append((local_col, web_col, mlabel))
        else:
            skipped_metrics.append(mlabel)

    rows = []
    for key in matched:
        skey = _flatten_key(key)
        lrow, wrow = li.loc[skey], wi.loc[skey]
        chip = lrow["_chip"]
        start, end = lrow["_start"], lrow["_end"]
        for local_col, web_col, mlabel in usable_metrics:
            lv = _to_float(lrow[local_col])
            wv = _to_float(wrow[web_col])
            pct = percent_diff(lv, wv)
            rows.append({
                "chip": chip,
                "start": start,
                "end": end,
                "quarter": _quarter_label(start),
                "metric": mlabel,
                "metric_group": _metric_group(mlabel),
                "local": lv,
                "website": wv,
                "abs_diff": abs(lv - wv) if not (np.isnan(lv) or np.isnan(wv)) else np.nan,
                "pct_diff": pct,
                "status": classify(pct, tol_pct, warn_pct),
            })

    diffs = pd.DataFrame(rows)

    return {
        "table": spec_key_from_web_file(spec["web_file"]),
        "title": spec["title"],
        "has_chip": spec["has_chip"],
        "local_path": local_path,
        "web_path": web_path,
        "diffs": diffs,
        "n_matched": len(matched),
        "local_only": local_only,
        "web_only": web_only,
        "local_ts": local_ts,
        "web_ts": web_ts,
        "skipped_metrics": skipped_metrics,
        "metrics": [m[2] for m in usable_metrics],
    }


def _metric_group(metric_label):
    """Map a metric label to its logical group for plotting."""
    for token, group in METRIC_GROUPS.items():
        if metric_label.startswith(token):
            return group
    return "Other"


def spec_key_from_web_file(web_file):
    """Reverse-lookup the TABLE_SPECS key for a given published filename."""
    for key, spec in TABLE_SPECS.items():
        if spec["web_file"] == web_file:
            return key
    return web_file


def summarize_metrics(diffs):
    """Per-metric summary: counts by status plus bias, typical magnitude, and worst cell."""
    if diffs.empty:
        return pd.DataFrame()
    out = []
    for metric, grp in diffs.groupby("metric", sort=False):
        finite = grp["pct_diff"].replace([np.inf, -np.inf], np.nan).dropna()
        out.append({
            "metric": metric,
            "n": len(grp),
            "ok": int((grp["status"] == "ok").sum()),
            "minor": int((grp["status"] == "minor").sum()),
            "major": int((grp["status"] == "major").sum()),
            "mean_signed_pct": finite.mean() if len(finite) else np.nan,
            "mean_abs_pct": finite.abs().mean() if len(finite) else np.nan,
            "max_abs_pct": finite.abs().max() if len(finite) else np.nan,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table_report(result, tol_pct, warn_pct, top_n=8):
    """Print a per-table console summary in the style of the repo's other check output."""
    print("\n" + "=" * 78)
    print(f"  {result['title']}  [{result['table']}]")
    print("=" * 78)
    print(f"  local:   {result['local_path']}")
    print(f"           run {_format_timestamps(result['local_ts'])}")
    print(f"  website: {os.path.basename(result['web_path'])}")
    print(f"           run {_format_timestamps(result['web_ts'])}")

    # Flag stale uploads: if the dominant timestamps differ, diffs are expected.
    local_dom = result["local_ts"].most_common(1)
    web_dom = result["web_ts"].most_common(1)
    if local_dom and web_dom and local_dom[0][0] != web_dom[0][0]:
        print("  ** timestamps differ - the published table is from a different model run **")

    print(f"\n  rows: {result['n_matched']} matched, "
          f"{len(result['local_only'])} local-only, {len(result['web_only'])} website-only")
    if result["skipped_metrics"]:
        print(f"  skipped metrics (column absent on one side): {', '.join(result['skipped_metrics'])}")

    _print_unmatched(result["local_only"], "local-only (in csv_export, not on website)")
    _print_unmatched(result["web_only"], "website-only (published, missing locally)")

    summary = summarize_metrics(result["diffs"])
    if summary.empty:
        print("  no comparable cells")
        return

    print()
    header = f"  {'metric':<16}{'n':>5}{'ok':>5}{'minor':>7}{'major':>7}{'bias%':>9}{'|mean|%':>9}{'max%':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in summary.iterrows():
        flag = "  <-- check" if r["major"] > 0 else ""
        print(f"  {r['metric']:<16}{int(r['n']):>5}{int(r['ok']):>5}{int(r['minor']):>7}{int(r['major']):>7}"
              f"{_fmt_pct(r['mean_signed_pct']):>9}{_fmt_pct(r['mean_abs_pct']):>9}{_fmt_pct(r['max_abs_pct']):>9}{flag}")

    # Show the largest individual cell discrepancies for quick eyeballing.
    worst = result["diffs"].copy()
    worst["abs_pct"] = worst["pct_diff"].abs()
    worst = worst[worst["status"].isin(["minor", "major"])].sort_values("abs_pct", ascending=False)
    if not worst.empty:
        print(f"\n  largest discrepancies (top {min(top_n, len(worst))}):")
        for _, r in worst.head(top_n).iterrows():
            loc_lab = "" if r["chip"] == "(total)" else f"{r['chip']} "
            print(f"    {loc_lab}{r['quarter']:<8} {r['metric']:<14} "
                  f"local={_fmt_num(r['local'])}  web={_fmt_num(r['website'])}  ({_fmt_pct(r['pct_diff'])})")


def _print_unmatched(keys, label):
    if not keys:
        return
    shown = ", ".join("/".join(str(p) for p in k) for k in keys[:10])
    more = f" (+{len(keys) - 10} more)" if len(keys) > 10 else ""
    print(f"  {label}: {shown}{more}")


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if np.isinf(x):
        return "inf"
    return f"{x:+.1f}%" if x else "0.0%"


def _fmt_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def write_diff_csvs(designer, results, outdir):
    """Write the full per-cell diff table for each comparison to CSV."""
    written = []
    for res in results:
        if res["diffs"].empty:
            continue
        path = os.path.join(outdir, f"{designer}_{res['table']}_diffs.csv")
        out = res["diffs"].copy()
        out["abs_pct"] = out["pct_diff"].abs()
        out = out.sort_values("abs_pct", ascending=False).drop(columns="abs_pct")
        out.to_csv(path, index=False)
        written.append(path)
    return written


def write_markdown_report(designer, results, outdir, tol_pct, warn_pct, figures):
    """Write a human-readable markdown summary of the whole validation run."""
    path = os.path.join(outdir, f"{designer}_validation_report.md")
    lines = [
        f"# Chip-sales validation report - {designer}",
        "",
        f"Comparison of local `csv_export/` outputs against the published Epoch dataset "
        f"({WEBSITE_ZIP_URL}).",
        f"Percent differences are local relative to website. Tolerance: ok <= {tol_pct}%, "
        f"minor <= {warn_pct}%, major > {warn_pct}%.",
        "",
    ]
    for res in results:
        lines.append(f"## {res['title']} (`{res['table']}`)")
        lines.append("")
        lines.append(f"- Local: `{res['local_path']}` - run {_format_timestamps(res['local_ts'])}")
        lines.append(f"- Website: `{os.path.basename(res['web_path'])}` - run {_format_timestamps(res['web_ts'])}")
        lines.append(f"- Rows: {res['n_matched']} matched, {len(res['local_only'])} local-only, "
                     f"{len(res['web_only'])} website-only")
        local_dom = res["local_ts"].most_common(1)
        web_dom = res["web_ts"].most_common(1)
        if local_dom and web_dom and local_dom[0][0] != web_dom[0][0]:
            lines.append("- **Note:** timestamps differ; the published table is from a different model run.")
        lines.append("")
        summary = summarize_metrics(res["diffs"])
        if not summary.empty:
            lines.append("| metric | n | ok | minor | major | bias% | mean abs% | max abs% |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for _, r in summary.iterrows():
                lines.append(f"| {r['metric']} | {int(r['n'])} | {int(r['ok'])} | {int(r['minor'])} | "
                             f"{int(r['major'])} | {_fmt_pct(r['mean_signed_pct'])} | "
                             f"{_fmt_pct(r['mean_abs_pct'])} | {_fmt_pct(r['max_abs_pct'])} |")
            lines.append("")
    if figures:
        lines.append("## Figures")
        lines.append("")
        for fig in figures:
            lines.append(f"- `{os.path.basename(fig)}`")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def make_visualizations(designer, results, outdir):
    """Render the validation figures and return the list of written PNG paths."""
    figures = []
    figures += _plot_scatter(designer, results, outdir)
    figures += _plot_pct_distributions(designer, results, outdir)
    figures += _plot_cumulative_timeseries(designer, results, outdir)
    figures += _plot_timelines_heatmap(designer, results, outdir)
    return [f for f in figures if f]


def _group_colors(groups):
    cmap = plt.get_cmap("tab10")
    uniq = list(dict.fromkeys(groups))
    return {g: cmap(i % 10) for i, g in enumerate(uniq)}


def _plot_scatter(designer, results, outdir):
    """Local vs website scatter (log-log) per table; points on the diagonal agree."""
    usable = [r for r in results if not r["diffs"].empty]
    if not usable:
        return []
    fig, axes = plt.subplots(1, len(usable), figsize=(6 * len(usable), 5.5), squeeze=False)
    axes = axes[0]
    colors = _group_colors([g for r in usable for g in r["diffs"]["metric_group"]])

    for ax, res in zip(axes, usable):
        d = res["diffs"]
        d = d[(d["local"] > 0) & (d["website"] > 0)]
        for group, sub in d.groupby("metric_group"):
            ax.scatter(sub["website"], sub["local"], s=22, alpha=0.6,
                       color=colors[group], label=group, edgecolors="none")
        if not d.empty:
            lo = min(d["website"].min(), d["local"].min())
            hi = max(d["website"].max(), d["local"].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, label="y = x")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel("website value")
        ax.set_ylabel("local value")
        ax.set_title(res["title"], fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle(f"{designer}: local vs published (each point is one cell)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, f"{designer}_scatter_local_vs_website.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def _plot_pct_distributions(designer, results, outdir):
    """Box-and-strip of percent differences per metric, one panel per table."""
    usable = [r for r in results if not r["diffs"].empty]
    if not usable:
        return []
    fig, axes = plt.subplots(1, len(usable), figsize=(6 * len(usable), 5.5), squeeze=False)
    axes = axes[0]

    for ax, res in zip(axes, usable):
        d = res["diffs"].copy()
        d["pct_diff"] = d["pct_diff"].replace([np.inf, -np.inf], np.nan)
        metrics = list(dict.fromkeys(d["metric"]))
        data = [d.loc[d["metric"] == m, "pct_diff"].dropna().values for m in metrics]
        positions = range(1, len(metrics) + 1)
        if any(len(arr) for arr in data):
            ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
            for pos, arr in zip(positions, data):
                if len(arr):
                    jitter = (np.random.default_rng(pos).random(len(arr)) - 0.5) * 0.25
                    ax.scatter(np.full(len(arr), pos) + jitter, arr, s=12, alpha=0.5, color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("percent difference (local vs website)")
        ax.set_title(res["title"], fontsize=11)
        ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle(f"{designer}: percent-difference distribution by metric", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, f"{designer}_pct_diff_distribution.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def _plot_cumulative_timeseries(designer, results, outdir):
    """Cumulative designer totals over time, local vs website, for units/H100e/power."""
    res = next((r for r in results if r["table"] == "cumulative_by_designer"), None)
    if res is None or res["diffs"].empty:
        return []
    d = res["diffs"].copy()
    d["end_dt"] = pd.to_datetime(d["end"])

    panels = [
        ("Units", "Units median", "Units p5", "Units p95", "cumulative units"),
        ("H100e", "H100e median", "H100e p5", "H100e p95", "cumulative H100e"),
        ("Power", "Power MW median", "Power MW p5", "Power MW p95", "cumulative power (MW)"),
    ]
    panels = [p for p in panels if p[1] in set(d["metric"])]
    if not panels:
        return []

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]
    for ax, (_, med, p5, p95, ylab) in zip(axes, panels):
        # Draw website as a wide line underneath and local as a thin line on top, so that
        # when they match the website color shows as a halo (rather than being hidden).
        for side, color, lw, ms, z in (("website", "tab:orange", 5, 0, 1), ("local", "tab:blue", 1.6, 3, 2)):
            sub_med = d[d["metric"] == med].sort_values("end_dt")
            ax.plot(sub_med["end_dt"], sub_med[side], marker="o", ms=ms, color=color,
                    lw=lw, alpha=0.5 if side == "website" else 1.0, label=side, zorder=z)
            sub5 = d[d["metric"] == p5].sort_values("end_dt")
            sub95 = d[d["metric"] == p95].sort_values("end_dt")
            if side == "local" and not sub5.empty and not sub95.empty:
                ax.fill_between(sub5["end_dt"], sub5[side], sub95[side], color=color, alpha=0.12)
        ax.set_title(ylab, fontsize=11)
        ax.set_xlabel("cumulative through")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"{designer}: cumulative totals over time (band = p5-p95)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, f"{designer}_cumulative_timeseries.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def _plot_timelines_heatmap(designer, results, outdir):
    """Heatmap of per-quarter median-units percent difference (chip x quarter)."""
    res = next((r for r in results if r["table"] == "timelines_by_chip"), None)
    if res is None or res["diffs"].empty:
        return []
    d = res["diffs"]
    d = d[d["metric"] == "Units median"].copy()
    if d.empty:
        return []
    d["pct_diff"] = d["pct_diff"].replace([np.inf, -np.inf], np.nan)
    d["end_dt"] = pd.to_datetime(d["start"])
    quarters = [q for _, q in sorted({(pd.to_datetime(s), qr) for s, qr in zip(d["start"], d["quarter"])})]
    chips = sorted(d["chip"].unique())
    grid = d.pivot_table(index="chip", columns="quarter", values="pct_diff", aggfunc="first")
    grid = grid.reindex(index=chips, columns=quarters)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(quarters)), max(3, 0.6 * len(chips))))
    vmax = np.nanmax(np.abs(grid.values)) if np.isfinite(grid.values).any() else 1
    vmax = max(vmax, 1)
    im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels(quarters, rotation=90, fontsize=8)
    ax.set_yticks(range(len(chips)))
    ax.set_yticklabels(chips, fontsize=9)
    # Annotate each populated cell with its percent difference.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(val) < vmax * 0.6 else "white")
    fig.colorbar(im, ax=ax, label="percent difference (local vs website)")
    ax.set_title(f"{designer}: per-quarter median units, local vs published", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, f"{designer}_timelines_units_heatmap.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def validate_designer(designer, data_dir, outdir, tol_pct, warn_pct, make_plots=True):
    """Run all three comparisons for one designer; return (results, ok_flag)."""
    config = DESIGNER_CONFIGS[designer]
    print("\n" + "#" * 78)
    print(f"#  Validating: {designer}")
    print("#" * 78)

    results = []
    for table_key, spec in TABLE_SPECS.items():
        local_path = config["local_files"][table_key]
        web_path = os.path.join(data_dir, spec["web_file"])
        if not os.path.exists(local_path):
            print(f"\n  skipping {table_key}: local file not found ({local_path})")
            continue
        res = compare_table(local_path, web_path, spec, config["web_name"],
                            config.get("chip_aliases", {}), tol_pct, warn_pct)
        results.append(res)
        print_table_report(res, tol_pct, warn_pct)

    # Decide pass/fail: any major cell or any unmatched row is worth a human look.
    total_major = sum(int((r["diffs"]["status"] == "major").sum()) for r in results if not r["diffs"].empty)
    total_unmatched = sum(len(r["local_only"]) + len(r["web_only"]) for r in results)
    ok = total_major == 0 and total_unmatched == 0

    figures = []
    if make_plots and results:
        figures = make_visualizations(designer, results, outdir)

    diff_csvs = write_diff_csvs(designer, results, outdir)
    report = write_markdown_report(designer, results, outdir, tol_pct, warn_pct, figures)

    print("\n" + "-" * 78)
    print(f"  {designer} summary: {total_major} major cell diff(s), {total_unmatched} unmatched row(s)")
    print(f"  wrote {len(diff_csvs)} diff CSV(s), {len(figures)} figure(s), and {os.path.basename(report)}")
    print(f"  verdict: {'PASS - matches published data within tolerance' if ok else 'REVIEW - drift detected (see above)'}")

    return results, ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--designer", default="nvidia",
                        help="designer to validate, or 'all' (default: nvidia). "
                             f"Configured: {', '.join(DESIGNER_CONFIGS)}")
    parser.add_argument("--data-dir", default=None,
                        help="directory of already-extracted website CSVs (skips download)")
    parser.add_argument("--cache-dir", default="validation_output/website_data",
                        help="where to download/extract the published zip")
    parser.add_argument("--output-dir", default="validation_output",
                        help="where to write diff CSVs, figures, and the report")
    parser.add_argument("--zip-url", default=WEBSITE_ZIP_URL, help="published dataset zip URL")
    parser.add_argument("--no-download", action="store_true",
                        help="reuse cached website data instead of downloading")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL_PCT,
                        help=f"percent tolerance for an exact match (default {DEFAULT_TOL_PCT})")
    parser.add_argument("--warn", type=float, default=DEFAULT_WARN_PCT,
                        help=f"percent threshold for major drift (default {DEFAULT_WARN_PCT})")
    parser.add_argument("--no-viz", action="store_true", help="skip generating figures")
    args = parser.parse_args(argv)

    if args.designer == "all":
        designers = list(DESIGNER_CONFIGS)
    elif args.designer in DESIGNER_CONFIGS:
        designers = [args.designer]
    else:
        parser.error(f"unknown designer '{args.designer}'. Configured: {', '.join(DESIGNER_CONFIGS)} (or 'all')")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_dir:
        data_dir = args.data_dir
        missing = [s["web_file"] for s in TABLE_SPECS.values()
                   if not os.path.exists(os.path.join(data_dir, s["web_file"]))]
        if missing:
            parser.error(f"--data-dir {data_dir} is missing {missing}")
    else:
        data_dir = fetch_website_data(args.zip_url, args.cache_dir, download=not args.no_download)

    all_ok = True
    for designer in designers:
        _, ok = validate_designer(designer, data_dir, args.output_dir, args.tol, args.warn,
                                  make_plots=not args.no_viz)
        all_ok = all_ok and ok

    print("\n" + "=" * 78)
    print(f"  Overall: {'PASS' if all_ok else 'REVIEW - drift detected in at least one table'}")
    print(f"  Artifacts in: {args.output_dir}/")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
