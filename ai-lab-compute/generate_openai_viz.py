"""Generate an interactive HTML visualization of the OpenAI power model results.

Replicates the calculations from openai_power_model.ipynb and emits a single
self-contained HTML file with Plotly.js charts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. Load source data
# ---------------------------------------------------------------------------
chips_df = pd.read_csv(HERE / "nvidia_owners_cumulative_by_chip.csv")
chips_df["End date"] = pd.to_datetime(chips_df["End date"])

power_df = pd.read_csv(HERE / "IT power by chip.csv")

lab_df = pd.read_csv(HERE / "lab IT power.csv")
lab_df["Date"] = pd.to_datetime(lab_df["Date"])

# ---------------------------------------------------------------------------
# 2. Microsoft fleet & watts-per-GPU map (mirrors the notebook)
# ---------------------------------------------------------------------------
TARGET_CHIP_TYPES = ["A100", "H100/H200", "B200", "B300"]

CSV_TO_FLEET_CHIP_NAME = {
    "A100": "A100",
    "H100": "H100/H200",
    "GB200": "B200",
    "GB300": "B300",
}

watts_per_gpu = {}
for _, row in power_df.iterrows():
    csv_name = row["Chip type"]
    if csv_name in CSV_TO_FLEET_CHIP_NAME:
        watts_per_gpu[CSV_TO_FLEET_CHIP_NAME[csv_name]] = row["IT power per GPU (W)"]

msft_fleet = chips_df[
    (chips_df["Owner"] == "Microsoft")
    & (chips_df["Chip type"].isin(TARGET_CHIP_TYPES))
].copy()
msft_fleet = msft_fleet.sort_values("End date")
msft_fleet["watts_per_gpu"] = msft_fleet["Chip type"].map(watts_per_gpu)
msft_fleet["IT_power_mw"] = msft_fleet["Number of Units"] * msft_fleet["watts_per_gpu"] / 1e6
msft_fleet["h100e"] = msft_fleet["Compute estimate in H100e (median)"]
msft_fleet["h100e_per_gpu"] = msft_fleet["h100e"] / msft_fleet["Number of Units"]

power_by_chip_over_time = (
    msft_fleet.pivot_table(
        index="End date", columns="Chip type", values="IT_power_mw", aggfunc="first"
    )
    .fillna(0)
    .sort_index()
)
for chip in TARGET_CHIP_TYPES:
    if chip not in power_by_chip_over_time.columns:
        power_by_chip_over_time[chip] = 0.0
power_by_chip_over_time = power_by_chip_over_time[TARGET_CHIP_TYPES]
power_by_chip_over_time["Total"] = power_by_chip_over_time.sum(axis=1)


# ---------------------------------------------------------------------------
# 3. OpenAI chip-count estimator (incremental Microsoft mix)
# ---------------------------------------------------------------------------
def estimate_lab_chip_counts(delay_months: int = 0) -> pd.DataFrame:
    lab_dates = sorted(lab_df["Date"].unique())
    cumulative_chips = {c: 0.0 for c in TARGET_CHIP_TYPES}
    cumulative_h100e = {c: 0.0 for c in TARGET_CHIP_TYPES}
    prev_lab_mw = 0.0
    prev_msft_date = None
    rows = []

    for lab_date in lab_dates:
        cur_mw = float(lab_df.loc[lab_df["Date"] == lab_date, "Total IT power (MW)"].iloc[0])
        new_mw = cur_mw - prev_lab_mw

        lookup = lab_date - relativedelta(months=delay_months)
        avail = power_by_chip_over_time.index[power_by_chip_over_time.index <= lookup]
        if len(avail) == 0:
            continue
        msft_date = avail[-1]

        if prev_msft_date is None:
            snap = power_by_chip_over_time.loc[msft_date]
            total = snap["Total"]
            mix = {c: snap[c] / total for c in TARGET_CHIP_TYPES} if total else {c: 0 for c in TARGET_CHIP_TYPES}
        else:
            cur_snap = power_by_chip_over_time.loc[msft_date]
            prev_snap = power_by_chip_over_time.loc[prev_msft_date]
            new_by_chip = {c: max(0, cur_snap[c] - prev_snap[c]) for c in TARGET_CHIP_TYPES}
            total_new = sum(new_by_chip.values())
            if total_new == 0:
                t = cur_snap["Total"]
                mix = {c: cur_snap[c] / t for c in TARGET_CHIP_TYPES}
            else:
                mix = {c: new_by_chip[c] / total_new for c in TARGET_CHIP_TYPES}

        for chip in TARGET_CHIP_TYPES:
            chip_mw = new_mw * mix[chip]
            wpg_mw = watts_per_gpu[chip] / 1e6
            new_gpus = chip_mw / wpg_mw

            match = msft_fleet[(msft_fleet["End date"] == msft_date) & (msft_fleet["Chip type"] == chip)]
            h100e_per = match.iloc[0]["h100e_per_gpu"] if len(match) else 0

            cumulative_chips[chip] += new_gpus
            cumulative_h100e[chip] += new_gpus * h100e_per

            rows.append({
                "Lab Date": lab_date,
                "Chip Type": chip,
                "Cumulative Chip Count": int(round(cumulative_chips[chip])),
                "Cumulative H100e": int(round(cumulative_h100e[chip])),
            })

        prev_lab_mw = cur_mw
        prev_msft_date = msft_date

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Build the structured payload that the HTML will consume
# ---------------------------------------------------------------------------
DELAY_SCENARIOS = {
    "no_delay": {"label": "No delay (baseline)", "months": 0},
    "1q_delay": {"label": "1-quarter delay (3 months)", "months": 3},
    "2q_delay": {"label": "2-quarter delay (6 months)", "months": 6},
}

openai_data = {}
for key, info in DELAY_SCENARIOS.items():
    df = estimate_lab_chip_counts(delay_months=info["months"])
    by_date = {}
    for d in sorted(df["Lab Date"].unique()):
        slice_ = df[df["Lab Date"] == d]
        by_date[pd.Timestamp(d).strftime("%Y-%m-%d")] = {
            "chips": {
                c: int(slice_.loc[slice_["Chip Type"] == c, "Cumulative Chip Count"].iloc[0])
                for c in TARGET_CHIP_TYPES
            },
            "h100e": {
                c: int(slice_.loc[slice_["Chip Type"] == c, "Cumulative H100e"].iloc[0])
                for c in TARGET_CHIP_TYPES
            },
            "total_h100e": int(slice_["Cumulative H100e"].sum()),
        }
    openai_data[key] = {"label": info["label"], "by_date": by_date}

# Microsoft & Meta quarterly Nvidia totals
ms_meta = (
    chips_df[chips_df["Owner"].isin(["Microsoft", "Meta"])]
    .groupby(["Owner", "End date"])["Compute estimate in H100e (median)"]
    .sum()
    .unstack("Owner")
    .sort_index()
)

competitor_series = {}
for owner in ["Microsoft", "Meta"]:
    # Drop the final (incomplete) quarter, matching the notebook
    s = ms_meta[owner].dropna().iloc[:-1]
    competitor_series[owner] = [
        {"date": d.strftime("%Y-%m-%d"), "h100e": int(v)} for d, v in s.items()
    ]

payload = {
    "openai": openai_data,
    "competitors": competitor_series,
    "chip_types": TARGET_CHIP_TYPES,
    "openai_dates": sorted(openai_data["no_delay"]["by_date"].keys()),
}

# ---------------------------------------------------------------------------
# 5. Emit the HTML
# ---------------------------------------------------------------------------
html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OpenAI Power Model — Interactive Visualization</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #fafafa;
    --fg: #1a1a1a;
    --muted: #555;
    --border: #d4d4d4;
    --card: #fff;
    --accent: #10a37f;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.45;
  }
  h1 { margin: 0 0 4px; font-size: 22px; }
  h2 { margin: 0 0 6px; font-size: 17px; }
  .sub { color: var(--muted); margin-bottom: 24px; font-size: 13px; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
    margin-bottom: 24px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 14px 22px;
    align-items: center;
    margin: 8px 0 14px;
    font-size: 13px;
  }
  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .control-group label.title {
    font-weight: 600;
    color: var(--muted);
  }
  .control-group .opt {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
  }
  select {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 13px;
    background: white;
  }
  .plot { width: 100%; height: 520px; }
  .footer {
    color: var(--muted);
    font-size: 12px;
    margin-top: 12px;
  }
  details {
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
  }
  details summary { cursor: pointer; }
</style>
</head>
<body>

<h1>OpenAI Power Model — Interactive Visualization</h1>
<div class="sub">
  Estimated OpenAI compute (H100-equivalents) derived from total IT power and
  Microsoft's incremental fleet mix. Median estimates only.
</div>

<!-- ====================== CHART 1: TRENDLINE ====================== -->
<div class="card">
  <h2>OpenAI vs Microsoft &amp; Meta — Cumulative compute (H100e)</h2>
  <div class="sub">
    Microsoft and Meta show Nvidia-only quarterly totals. OpenAI is interpolated
    exponentially between the three year-end anchor points.
  </div>

  <div class="controls">
    <div class="control-group">
      <label class="title">Y-axis:</label>
      <label class="opt"><input type="radio" name="trend-scale" value="linear" checked> Linear</label>
      <label class="opt"><input type="radio" name="trend-scale" value="log"> Log</label>
    </div>

    <div class="control-group">
      <label class="title">OpenAI delay scenario:</label>
      <select id="trend-scenario">
        <option value="no_delay">No delay (baseline)</option>
        <option value="1q_delay" selected>1-quarter delay</option>
        <option value="2q_delay">2-quarter delay</option>
      </select>
    </div>

    <div class="control-group">
      <label class="title">Shift Microsoft/Meta sales by:</label>
      <select id="trend-shift">
        <option value="0" selected>0 (sales date)</option>
        <option value="3">+1 quarter (3 months)</option>
        <option value="6">+2 quarters (6 months)</option>
      </select>
    </div>
  </div>

  <div id="trend-plot" class="plot"></div>

  <details>
    <summary>About the deployment-delay shift</summary>
    Nvidia revenue recognition reflects when chips are sold, not when they're
    operational. Shifting Microsoft/Meta's curves to the right approximates when
    those chips actually came online in data centers — a more apples-to-apples
    comparison with OpenAI's IT-power-based estimate.
  </details>
</div>

<!-- ====================== CHART 2: BAR — TOTAL H100e ====================== -->
<div class="card">
  <h2>OpenAI total H100e by year</h2>
  <div class="sub">Bars stacked by chip type (or chip family). Hover for details.</div>

  <div class="controls">
    <div class="control-group">
      <label class="title">Y-axis:</label>
      <label class="opt"><input type="radio" name="bar-scale" value="linear" checked> Linear</label>
      <label class="opt"><input type="radio" name="bar-scale" value="log"> Log</label>
    </div>

    <div class="control-group">
      <label class="title">Group by:</label>
      <label class="opt"><input type="radio" name="bar-group" value="chip" checked> Chip type</label>
      <label class="opt"><input type="radio" name="bar-group" value="family"> Chip family</label>
    </div>

    <div class="control-group">
      <label class="title">Show scenarios:</label>
      <label class="opt"><input type="checkbox" name="bar-scen" value="no_delay" checked> No delay</label>
      <label class="opt"><input type="checkbox" name="bar-scen" value="1q_delay" checked> 1Q delay</label>
      <label class="opt"><input type="checkbox" name="bar-scen" value="2q_delay" checked> 2Q delay</label>
    </div>
  </div>

  <div id="bar-plot" class="plot"></div>
</div>

<!-- ====================== CHART 3: BAR — CHIP COUNTS ====================== -->
<div class="card">
  <h2>OpenAI chip counts by year</h2>
  <div class="sub">Estimated number of physical GPUs OpenAI is operating, by chip type.</div>

  <div class="controls">
    <div class="control-group">
      <label class="title">Y-axis:</label>
      <label class="opt"><input type="radio" name="cnt-scale" value="linear" checked> Linear</label>
      <label class="opt"><input type="radio" name="cnt-scale" value="log"> Log</label>
    </div>

    <div class="control-group">
      <label class="title">Delay scenario:</label>
      <select id="cnt-scenario">
        <option value="no_delay">No delay (baseline)</option>
        <option value="1q_delay" selected>1-quarter delay</option>
        <option value="2q_delay">2-quarter delay</option>
      </select>
    </div>
  </div>

  <div id="cnt-plot" class="plot"></div>
</div>

<div class="footer">
  Generated from <code>openai_power_model.ipynb</code>. Microsoft/Meta totals
  cover Nvidia chips only (no internal silicon). No chip retirements assumed.
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const CHIP_COLORS = {
  "A100": "#8b5cf6",
  "H100/H200": "#76b900",
  "B200": "#1a73e8",
  "B300": "#e8710a"
};
const CHIP_DISPLAY = {
  "A100": "A100",
  "H100/H200": "H100/H200",
  "B200": "B200 (GB200)",
  "B300": "B300 (GB300)"
};
const FAMILY_OF = {
  "A100": "Ampere",
  "H100/H200": "Hopper",
  "B200": "Blackwell",
  "B300": "Blackwell"
};
const FAMILY_COLORS = {
  "Ampere": "#8b5cf6",
  "Hopper": "#76b900",
  "Blackwell": "#1a73e8"
};
const FAMILY_ORDER = ["Ampere", "Hopper", "Blackwell"];

const COMPETITOR_COLORS = {
  "Microsoft": "#f25022",
  "Meta": "#1877f2",
  "OpenAI": "#10a37f"
};

const fmt = (n) => n.toLocaleString(undefined, {maximumFractionDigits: 0});
const fmtK = (n) => (n / 1000).toLocaleString(undefined, {maximumFractionDigits: 0}) + "k";

function shiftDateMonths(dateStr, months) {
  // dateStr is YYYY-MM-DD
  const [y, m, d] = dateStr.split("-").map(Number);
  // Use UTC to avoid timezone ambiguity
  const dt = new Date(Date.UTC(y, m - 1, d));
  dt.setUTCMonth(dt.getUTCMonth() + months);
  return dt.toISOString().slice(0, 10);
}

function dateRange(startStr, endStr, n) {
  const start = Date.UTC(...startStr.split("-").map((v, i) => i === 1 ? Number(v) - 1 : Number(v)));
  const end = Date.UTC(...endStr.split("-").map((v, i) => i === 1 ? Number(v) - 1 : Number(v)));
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = start + (end - start) * i / (n - 1);
    out.push(new Date(t).toISOString().slice(0, 10));
  }
  return out;
}

function expSegment(d1, y1, d2, y2, n = 60) {
  const t1 = Date.parse(d1);
  const t2 = Date.parse(d2);
  const k = Math.log(y2 / y1) / (t2 - t1);
  const dates = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const t = t1 + (t2 - t1) * i / (n - 1);
    dates.push(new Date(t).toISOString().slice(0, 10));
    ys.push(y1 * Math.exp(k * (t - t1)));
  }
  return {dates, ys};
}

// ---------------------------------------------------------------
// CHART 1: Trendline
// ---------------------------------------------------------------
function renderTrend() {
  const scale = document.querySelector('input[name="trend-scale"]:checked').value;
  const scenario = document.getElementById("trend-scenario").value;
  const shift = parseInt(document.getElementById("trend-shift").value, 10);

  const traces = [];

  // Microsoft & Meta (shifted)
  for (const owner of ["Microsoft", "Meta"]) {
    const series = DATA.competitors[owner];
    const x = series.map(p => shiftDateMonths(p.date, shift));
    const y = series.map(p => p.h100e);
    traces.push({
      x, y,
      mode: "lines+markers",
      name: owner + " (Nvidia)" + (shift ? ` +${shift / 3}Q` : ""),
      line: {color: COMPETITOR_COLORS[owner], width: 2},
      marker: {size: 6},
      hovertemplate: `<b>${owner}</b><br>%{x}<br>H100e: %{y:,.0f}<extra></extra>`
    });
  }

  // OpenAI: anchors + exponential interpolation between them
  const oaiByDate = DATA.openai[scenario].by_date;
  const anchorDates = DATA.openai_dates;
  const anchorY = anchorDates.map(d => oaiByDate[d].total_h100e);

  const interpX = [];
  const interpY = [];
  for (let i = 0; i < anchorDates.length - 1; i++) {
    const seg = expSegment(anchorDates[i], anchorY[i], anchorDates[i + 1], anchorY[i + 1]);
    if (i > 0) { seg.dates.shift(); seg.ys.shift(); }
    interpX.push(...seg.dates);
    interpY.push(...seg.ys);
  }

  traces.push({
    x: interpX, y: interpY,
    mode: "lines",
    name: `OpenAI (${DATA.openai[scenario].label}, exp. interp.)`,
    line: {color: COMPETITOR_COLORS.OpenAI, width: 3},
    hovertemplate: `<b>OpenAI</b> (interpolated)<br>%{x}<br>H100e: %{y:,.0f}<extra></extra>`
  });

  traces.push({
    x: anchorDates, y: anchorY,
    mode: "markers+text",
    name: "OpenAI anchors",
    marker: {color: COMPETITOR_COLORS.OpenAI, size: 11, line: {color: "white", width: 2}},
    text: anchorY.map(v => fmtK(v)),
    textposition: "bottom center",
    textfont: {color: COMPETITOR_COLORS.OpenAI, size: 11},
    showlegend: false,
    hovertemplate: "<b>OpenAI anchor</b><br>%{x}<br>H100e: %{y:,.0f}<extra></extra>"
  });

  const layout = {
    margin: {l: 70, r: 20, t: 10, b: 50},
    hovermode: "closest",
    xaxis: {title: "Date", gridcolor: "#eee"},
    yaxis: {
      title: "Cumulative H100-equivalents",
      type: scale,
      gridcolor: "#eee",
      tickformat: ",.0f",
      ...(scale === "log" ? {
        dtick: 1,
        minor: {dtick: "D1", showgrid: true, gridcolor: "#f3f3f3", ticks: ""}
      } : {})
    },
    legend: {x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)"},
    plot_bgcolor: "white",
    paper_bgcolor: "white"
  };

  Plotly.react("trend-plot", traces, layout, {responsive: true, displaylogo: false});
}

// ---------------------------------------------------------------
// CHART 2: Total H100e bar chart by year × scenario
// ---------------------------------------------------------------
function renderBar() {
  const scale = document.querySelector('input[name="bar-scale"]:checked').value;
  const grouping = document.querySelector('input[name="bar-group"]:checked').value;
  const activeScenarios = Array.from(document.querySelectorAll('input[name="bar-scen"]:checked'))
    .map(el => el.value);

  if (activeScenarios.length === 0) {
    Plotly.react("bar-plot", [], {annotations: [{text: "Select at least one scenario", showarrow: false}]});
    return;
  }

  const dates = DATA.openai_dates;
  // x-axis groups by year; bars within a group are scenarios
  const xLabels = [];
  const scenarioPerBar = [];
  for (const d of dates) {
    for (const s of activeScenarios) {
      xLabels.push(d.slice(0, 4)); // year
      scenarioPerBar.push(s);
    }
  }

  // For grouped-on-x via shared x label, use offsetgroup per scenario
  const traces = [];

  if (grouping === "chip") {
    for (const chip of DATA.chip_types) {
      for (const s of activeScenarios) {
        const xs = dates.map(d => d.slice(0, 4));
        const ys = dates.map(d => DATA.openai[s].by_date[d].h100e[chip]);
        traces.push({
          type: "bar",
          x: xs,
          y: ys,
          name: `${CHIP_DISPLAY[chip]}`,
          legendgroup: chip,
          showlegend: s === activeScenarios[0],
          offsetgroup: s,
          marker: {color: CHIP_COLORS[chip]},
          customdata: xs.map(() => DATA.openai[s].label),
          hovertemplate: `<b>${CHIP_DISPLAY[chip]}</b><br>` +
                         `%{customdata}<br>` +
                         `Year: %{x}<br>` +
                         `H100e: %{y:,.0f}<extra></extra>`
        });
      }
    }
  } else {
    // family
    const familyChips = {Ampere: ["A100"], Hopper: ["H100/H200"], Blackwell: ["B200", "B300"]};
    for (const fam of FAMILY_ORDER) {
      for (const s of activeScenarios) {
        const xs = dates.map(d => d.slice(0, 4));
        const ys = dates.map(d => familyChips[fam].reduce((a, c) => a + DATA.openai[s].by_date[d].h100e[c], 0));
        traces.push({
          type: "bar",
          x: xs,
          y: ys,
          name: fam,
          legendgroup: fam,
          showlegend: s === activeScenarios[0],
          offsetgroup: s,
          marker: {color: FAMILY_COLORS[fam]},
          customdata: xs.map(() => DATA.openai[s].label),
          hovertemplate: `<b>${fam}</b><br>` +
                         `%{customdata}<br>` +
                         `Year: %{x}<br>` +
                         `H100e: %{y:,.0f}<extra></extra>`
        });
      }
    }
  }

  // Add scenario labels under each x group
  const annotations = [];
  if (activeScenarios.length > 1) {
    const scenLabel = {no_delay: "No delay", "1q_delay": "1Q delay", "2q_delay": "2Q delay"};
    // Plotly stack-grouping with offsetgroup needs xaxis "group" mode for offsets
    // We'll show scenario in hover and as a separate xaxis tick
  }

  const layout = {
    barmode: "stack",
    margin: {l: 70, r: 20, t: 10, b: 60},
    xaxis: {title: "Year", gridcolor: "#eee"},
    yaxis: {
      title: "Cumulative H100e",
      type: scale,
      gridcolor: "#eee",
      tickformat: ",.0f",
      ...(scale === "log" ? {
        dtick: 1,
        minor: {dtick: "D1", showgrid: true, gridcolor: "#f3f3f3", ticks: ""}
      } : {})
    },
    legend: {x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)"},
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    bargap: 0.15,
    bargroupgap: 0.05
  };

  Plotly.react("bar-plot", traces, layout, {responsive: true, displaylogo: false});
}

// ---------------------------------------------------------------
// CHART 3: Chip counts by year
// ---------------------------------------------------------------
function renderCounts() {
  const scale = document.querySelector('input[name="cnt-scale"]:checked').value;
  const scenario = document.getElementById("cnt-scenario").value;

  const dates = DATA.openai_dates;
  const traces = DATA.chip_types.map(chip => ({
    type: "bar",
    x: dates.map(d => d.slice(0, 4)),
    y: dates.map(d => DATA.openai[scenario].by_date[d].chips[chip]),
    name: CHIP_DISPLAY[chip],
    marker: {color: CHIP_COLORS[chip]},
    hovertemplate: `<b>${CHIP_DISPLAY[chip]}</b><br>Year: %{x}<br>Chips: %{y:,.0f}<extra></extra>`
  }));

  const layout = {
    barmode: "stack",
    margin: {l: 70, r: 20, t: 10, b: 50},
    xaxis: {title: "Year", gridcolor: "#eee"},
    yaxis: {
      title: "Number of GPUs",
      type: scale,
      gridcolor: "#eee",
      tickformat: ",.0f",
      ...(scale === "log" ? {
        dtick: 1,
        minor: {dtick: "D1", showgrid: true, gridcolor: "#f3f3f3", ticks: ""}
      } : {})
    },
    legend: {x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.85)"},
    plot_bgcolor: "white",
    paper_bgcolor: "white"
  };

  Plotly.react("cnt-plot", traces, layout, {responsive: true, displaylogo: false});
}

// ---------------------------------------------------------------
// Wire up controls
// ---------------------------------------------------------------
function wireControls() {
  document.querySelectorAll('input[name="trend-scale"]').forEach(el => el.addEventListener("change", renderTrend));
  document.getElementById("trend-scenario").addEventListener("change", renderTrend);
  document.getElementById("trend-shift").addEventListener("change", renderTrend);

  document.querySelectorAll('input[name="bar-scale"]').forEach(el => el.addEventListener("change", renderBar));
  document.querySelectorAll('input[name="bar-group"]').forEach(el => el.addEventListener("change", renderBar));
  document.querySelectorAll('input[name="bar-scen"]').forEach(el => el.addEventListener("change", renderBar));

  document.querySelectorAll('input[name="cnt-scale"]').forEach(el => el.addEventListener("change", renderCounts));
  document.getElementById("cnt-scenario").addEventListener("change", renderCounts);
}

wireControls();
renderTrend();
renderBar();
renderCounts();
</script>
</body>
</html>
"""

html = html.replace("__DATA_PLACEHOLDER__", json.dumps(payload))

out_path = HERE / "openai_power_viz.html"
out_path.write_text(html)
print(f"Wrote {out_path}  ({len(html):,} bytes)")
