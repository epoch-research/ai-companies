"""build_compute_page.py — Consolidated chart + model-walkthrough page (draft)

Renders the frontier-lab compute estimates as one self-contained page,
index.html: the grouped bar chart (median bars, 90%-CI whiskers, one bar per
lab per year-end) on top, and one model-walkthrough section per (lab ×
year-end) snapshot below — every intermediate quantity in model order, drawn
as a median dot with a capped 90%-CI whisker on a full-width track.
Each bar's tooltip (and the bar itself) links to its walkthrough section;
OpenAI's 2023-24 bars, which have no separate trace, link to the OpenAI
end-2025 section since they are per-year outputs of the same power model.

The page is a pure view over the two tables from generate_tables — the same
data as the exported CSVs — so it picks up new steps or snapshots added to
MODEL_STEPS in the frontier script (plus their LAB_YEAR_KEYS entry) with no
changes to the viz code. Styled on the Epoch website palette, with a dark
theme keyed off prefers-color-scheme (a data-theme attribute on <html>
overrides it). Lab colors are CVD-validated for both surfaces; Messina Sans
(Epoch's chart font) is commercially licensed and therefore not embedded —
the page falls back to system sans-serifs.

Usage
-----
    python build_compute_page.py

Re-run it whenever the models or priors change.
"""

import datetime
import json
from pathlib import Path

from generate_tables import get_all_tables, LAB_YEAR_KEYS

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "index.html"


def _round(v):
    """Trim floats for the embedded JSON without visible precision loss."""
    return float(f"{float(v):.6g}")


def build_payload():
    """Run the models once and shape both tables for the page: year-end rows
    for the chart, per-snapshot step lists for the walkthrough sections."""
    tables = get_all_tables()

    year_end = [
        dict(lab=r.Lab, year=int(r.Year),
             p5=round(r.h100e_p5), med=round(r.h100e_med), p95=round(r.h100e_p95))
        for r in tables["year_end_by_lab"].itertuples()
    ]

    inter = tables["intermediates_by_lab"]
    labs = []
    for (lab, year) in LAB_YEAR_KEYS:
        rows = inter[(inter["Lab"] == lab) & (inter["Year"] == year)].sort_values("Step")
        steps = [
            dict(i=int(r.Step), name=r.Variable, label=r.Label, kind=r.Kind,
                 units=r.Units,
                 expression=r.Expression if isinstance(r.Expression, str) else "",
                 p5=_round(r.value_p5), med=_round(r.value_med), p95=_round(r.value_p95))
            for r in rows.itertuples()
        ]
        labs.append(dict(lab=lab, year=year, steps=steps))

    return dict(generated=f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                yearEnd=year_end, labs=labs)


def main():
    payload = build_payload()
    html = (PAGE_TEMPLATE
            .replace("__DATA_JSON__", json.dumps(payload))
            .replace("__GENERATED__", payload["generated"]))
    OUT_PATH.write_text(html)
    n_steps = sum(len(lab["steps"]) for lab in payload["labs"])
    print(f"Wrote {OUT_PATH.name}: {len(payload['yearEnd'])} bars, "
          f"{len(payload['labs'])} walkthrough sections, {n_steps} step rows, "
          f"{len(html) / 1024:.0f} KB")


# ── Page template ─────────────────────────────────────────────────────────────
# Self-contained, no external requests. Theme tokens on :root; the dark block
# is duplicated for the media query and the data-theme override so an explicit
# toggle beats the OS preference in both directions.

PAGE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frontier lab compute — chart &amp; model walkthrough</title>
<style>
  :root {
    --bg: #F2FAF9;
    --card: #FFFFFF;
    --border: #E2EEEE;
    --hairline: #EBF5F4;
    --text: #07171A;
    --secondary: #3E555E;
    --muted: #5C737B;
    --ink: #162B32;          /* chart whiskers and CI caps */
    --grid: #E2EEEE;
    --axis: #CCD8D9;
    --track: #EBF5F4;
    --tip-shadow: rgba(7, 23, 26, 0.14);
    --chip-hover: #F2FAF9;
    /* Lab colors — Epoch organizationColors, validated for the light surface */
    --c-openai: #E03D90;
    --c-google-deepmind: #00A5A6;
    --c-anthropic: #6A3ECB;
    --c-meta-superintelligence-labs: #FC6538;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0A171B;
      --card: #0F2127;
      --border: #1E3840;
      --hairline: #162B32;
      --text: #E7F2F1;
      --secondary: #A9C0C4;
      --muted: #7E979D;
      --ink: #D5E6E5;
      --grid: #17303792;
      --axis: #2B4A53;
      --track: #162B32;
      --tip-shadow: rgba(0, 0, 0, 0.45);
      --chip-hover: #132931;
      /* Dark variants validated against the dark card surface: purple raised,
         orange deepened; pink and teal are on-brand unchanged. */
      --c-anthropic: #7B4FD8;
      --c-meta-superintelligence-labs: #F25322;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0A171B;
    --card: #0F2127;
    --border: #1E3840;
    --hairline: #162B32;
    --text: #E7F2F1;
    --secondary: #A9C0C4;
    --muted: #7E979D;
    --ink: #D5E6E5;
    --grid: #17303792;
    --axis: #2B4A53;
    --track: #162B32;
    --tip-shadow: rgba(0, 0, 0, 0.45);
    --chip-hover: #132931;
    --c-anthropic: #7B4FD8;
    --c-meta-superintelligence-labs: #F25322;
  }
  :root[data-theme="light"] {
    --bg: #F2FAF9;
    --card: #FFFFFF;
    --border: #E2EEEE;
    --hairline: #EBF5F4;
    --text: #07171A;
    --secondary: #3E555E;
    --muted: #5C737B;
    --ink: #162B32;
    --grid: #E2EEEE;
    --axis: #CCD8D9;
    --track: #EBF5F4;
    --tip-shadow: rgba(7, 23, 26, 0.14);
    --chip-hover: #F2FAF9;
    --c-anthropic: #6A3ECB;
    --c-meta-superintelligence-labs: #FC6538;
  }

  html { scroll-behavior: smooth; }
  @media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
  }

  * { box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Messina Sans', -apple-system, 'Helvetica Neue', Arial, sans-serif;
    margin: 0;
    padding: 40px 20px 56px;
  }
  .page { max-width: 980px; margin: 0 auto; }

  a { color: var(--secondary); }
  :focus-visible { outline: 2px solid var(--secondary); outline-offset: 2px; }

  /* ── Masthead ─────────────────────────────────────────────────────────── */
  .kicker {
    font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 10px;
  }
  .masthead h1 {
    font-size: 24px; font-weight: 600; letter-spacing: -0.01em;
    margin: 0 0 8px; text-wrap: balance;
  }
  .masthead .lede {
    font-size: 13.5px; line-height: 1.55; color: var(--secondary);
    margin: 0 0 24px; max-width: 74ch;
  }

  /* ── Cards ────────────────────────────────────────────────────────────── */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px 28px 18px;
    margin-bottom: 22px;
  }

  /* ── Chart card ───────────────────────────────────────────────────────── */
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 2px 0 16px; }
  .legend-item {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 13px; color: var(--secondary);
  }
  .swatch { width: 10px; height: 10px; border-radius: 2px; flex: none; }

  .chart-scroll { overflow-x: auto; }
  .chart-scroll svg {
    display: block; width: 100%; min-width: 640px; height: auto;
    font-family: inherit;
  }
  svg .gridline { stroke: var(--grid); stroke-width: 1; }
  svg .gridline.zero { stroke: var(--axis); }
  svg .tick-label { fill: var(--muted); font-size: 12.5px; }
  svg .axis-title { fill: var(--secondary); font-size: 12.5px; }
  svg .year-label { fill: var(--secondary); font-size: 13px; }
  svg .bar-value { fill: var(--secondary); font-size: 11.5px; }
  svg .whisk { stroke: var(--ink); stroke-width: 1.5; }
  svg .bar { cursor: pointer; }
  svg .bar:hover { filter: brightness(0.92); }
  svg .bar:focus { outline: none; }
  svg .bar:focus-visible, svg .bar.focused { stroke: var(--ink); stroke-width: 1.5px; }
  .lab-openai { fill: var(--c-openai); }
  .lab-google-deepmind { fill: var(--c-google-deepmind); }
  .lab-anthropic { fill: var(--c-anthropic); }
  .lab-meta-superintelligence-labs { fill: var(--c-meta-superintelligence-labs); }
  .bg-openai { background: var(--c-openai); }
  .bg-google-deepmind { background: var(--c-google-deepmind); }
  .bg-anthropic { background: var(--c-anthropic); }
  .bg-meta-superintelligence-labs { background: var(--c-meta-superintelligence-labs); }

  .chart-footnote {
    font-size: 12px; line-height: 1.5; color: var(--muted);
    border-top: 1px solid var(--hairline); margin: 10px 0 0; padding-top: 10px;
  }

  /* ── Data table ───────────────────────────────────────────────────────── */
  details.datatable { margin-top: 12px; }
  details.datatable summary {
    font-size: 12.5px; color: var(--secondary); cursor: pointer;
    width: fit-content; border-radius: 4px; padding: 2px 4px;
  }
  details.datatable summary:hover { color: var(--text); }
  .table-scroll { overflow-x: auto; margin-top: 8px; }
  table.values {
    border-collapse: collapse; font-size: 12.5px; min-width: 480px;
    font-variant-numeric: tabular-nums;
  }
  table.values caption {
    text-align: left; font-size: 11px; color: var(--muted); padding-bottom: 6px;
  }
  table.values th, table.values td {
    padding: 5px 14px 5px 0; border-bottom: 1px solid var(--hairline);
    text-align: right; white-space: nowrap;
  }
  table.values th { color: var(--muted); font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; }
  table.values th:first-child, table.values td:first-child { text-align: left; }
  table.values td:first-child { color: var(--text); }
  table.values td { color: var(--secondary); }

  /* ── Walkthrough band ─────────────────────────────────────────────────── */
  .walkthrough-head { margin: 34px 0 18px; }
  .walkthrough-head h2 {
    font-size: 18px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px;
  }
  .walkthrough-head p {
    font-size: 13.5px; line-height: 1.55; color: var(--secondary);
    margin: 0 0 14px; max-width: 78ch;
  }
  .walkthrough-head code {
    font-size: 12px; background: var(--hairline); border-radius: 4px;
    padding: 1px 4px;
  }
  nav.labs { display: flex; flex-wrap: wrap; gap: 8px; }
  nav.labs a {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 12.5px; color: var(--secondary); text-decoration: none;
    border: 1px solid var(--border); border-radius: 999px;
    padding: 5px 12px; background: var(--card);
    font-variant-numeric: tabular-nums;
  }
  nav.labs a:hover { background: var(--chip-hover); color: var(--text); }

  /* ── Walkthrough sections ─────────────────────────────────────────────── */
  section.lab { scroll-margin-top: 16px; }
  section.lab > header {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 14px;
    margin-bottom: 4px;
  }
  section.lab > header .swatch { align-self: center; }
  section.lab h3 { font-size: 16.5px; font-weight: 600; margin: 0; }
  section.lab .headline { font-size: 13px; color: var(--secondary); font-variant-numeric: tabular-nums; }
  section.lab .headline b { color: var(--text); font-weight: 600; }
  section.lab .badge {
    font-size: 10px; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px; padding: 2px 8px;
  }
  section.lab .backlink {
    margin-left: auto; font-size: 11.5px; color: var(--muted);
    text-decoration: none;
  }
  section.lab .backlink:hover { color: var(--text); text-decoration: underline; }
  .sec-openai { --lab: var(--c-openai); }
  .sec-google-deepmind { --lab: var(--c-google-deepmind); }
  .sec-anthropic { --lab: var(--c-anthropic); }
  .sec-meta-superintelligence-labs { --lab: var(--c-meta-superintelligence-labs); }

  .group { border-top: 1px solid var(--hairline); padding: 10px 0 12px; margin-top: 12px; }
  .group:first-of-type { border-top: none; }
  .row, .ghead, .gfoot {
    display: grid; grid-template-columns: 240px 1fr 205px;
    gap: 4px 16px; align-items: center;
  }
  .ghead .gkicker {
    font-size: 10.5px; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--muted);
  }
  .row { padding: 5px 0; }
  .lbl .meta {
    font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted);
  }
  .lbl .name { font-size: 12.5px; color: var(--secondary); line-height: 1.3; }
  .row.final .lbl .name { font-weight: 600; color: var(--text); }
  .lbl .expr { font-size: 11px; color: var(--muted); line-height: 1.35; margin-top: 1px; }

  .track { position: relative; height: 26px; cursor: default; }
  .track .rail {
    position: absolute; left: 0; right: 0; top: 50%; height: 5px;
    transform: translateY(-50%); background: var(--track); border-radius: 2.5px;
  }
  .track .ci, .track .cap {
    position: absolute; top: 50%; transform: translateY(-50%);
    background: var(--lab); border-radius: 1px;
  }
  .track .ci { height: 2px; }
  .track .cap { width: 2px; height: 12px; }
  .track .dot {
    position: absolute; top: 50%; width: 9px; height: 9px; border-radius: 50%;
    transform: translate(-50%, -50%); background: var(--lab);
    box-shadow: 0 0 0 2px var(--card);
  }
  .row.final .track .dot { width: 11px; height: 11px; }
  .track .tick { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--hairline); }

  .val {
    font-size: 12px; color: var(--muted); text-align: right;
    font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .val b { font-size: 12.5px; font-weight: 600; color: var(--text); }

  .gfoot { margin-top: 2px; }
  .axisrow { position: relative; height: 18px; }
  .axisrow .stub { position: absolute; top: 0; width: 1px; height: 4px; background: var(--axis); }
  .axisrow .tlabel {
    position: absolute; top: 5px; font-size: 10px; color: var(--muted);
    transform: translateX(-50%); white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  /* ── Tooltips ─────────────────────────────────────────────────────────── */
  .tooltip {
    position: fixed; display: none; z-index: 10;
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    box-shadow: 0 2px 10px var(--tip-shadow);
    padding: 8px 11px; font-size: 12.5px; line-height: 1.55; color: var(--text);
    font-variant-numeric: tabular-nums; max-width: 260px;
  }
  .tooltip b { font-weight: 600; }
  .tooltip .muted { color: var(--muted); }
  #tip-chart { pointer-events: auto; }
  #tip-chart .tip-link {
    display: inline-block; margin-top: 5px; font-size: 12px; font-weight: 600;
    color: var(--secondary);
  }
  #tip-chart .tip-link:hover { color: var(--text); }
  #tip-step { pointer-events: none; max-width: 280px; font-size: 12px; }

  /* ── Footer ───────────────────────────────────────────────────────────── */
  .page > footer {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 12px; flex-wrap: wrap;
    font-size: 12px; color: var(--muted); margin-top: 26px;
  }
  .page > footer a { color: var(--secondary); }
  .wordmark {
    font-weight: 600; font-size: 11px; letter-spacing: 0.15em;
    color: var(--muted); white-space: nowrap;
  }

  @media (max-width: 720px) {
    body { padding: 24px 14px 40px; }
    .card { padding: 18px 16px 14px; }
    .row, .ghead, .gfoot { grid-template-columns: 1fr; gap: 2px; }
    .val { text-align: left; }
    .ghead .gkicker { margin-bottom: 2px; }
    section.lab .backlink { margin-left: 0; }
  }
</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <p class="kicker">Epoch AI &middot; Frontier-lab compute models</p>
    <h1>How much compute do frontier AI labs have?</h1>
    <p class="lede">
      Estimated total compute rented or used by each lab at the end of the year, in
      H100&#8209;equivalents, from Epoch&rsquo;s frontier-lab compute Monte Carlo models.
      Bars show the median estimate; whiskers span the 90% credible interval.
      <b>Hover or click a bar</b> to see how that estimate is built, step by step.
      2024 values for Google DeepMind and Meta are backcasts covering the predecessor
      frontier-AI orgs (Meta Superintelligence Labs did not exist in 2024).
    </p>
  </header>

  <section class="card" id="chart-card" aria-label="Frontier lab compute chart">
    <div class="legend" id="legend"></div>
    <div class="chart-scroll">
      <svg id="chart" viewBox="0 0 860 430" role="img"
           aria-label="Grouped bar chart of frontier lab compute at end of 2023, 2024 and 2025, in H100 equivalents, with 90 percent credible intervals. Each bar links to a walkthrough of its model."></svg>
    </div>
    <details class="datatable">
      <summary>Data table</summary>
      <div class="table-scroll">
        <table class="values" id="values-table">
          <caption>H100-equivalents; 90% credible interval bounds. Snapshots, not flows &mdash; years must not be summed.</caption>
          <thead>
            <tr><th scope="col">Lab</th><th scope="col">Year-end</th><th scope="col">5th pct</th><th scope="col">Median</th><th scope="col">95th pct</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </details>
    <p class="chart-footnote">
      Estimates are operational-stock snapshots of compute rented or used (not owned) &mdash;
      consecutive years must not be summed. Missing bars mean &ldquo;no estimate&rdquo;, not zero:
      Anthropic&rsquo;s end-2024 backcast is not exported, and 2023 covers OpenAI only.
    </p>
  </section>

  <section class="walkthrough-head" id="walkthrough">
    <h2>How each estimate is built</h2>
    <p>
      Each estimate &mdash; every lab at end-2025, plus end-2024 backcasts for Google
      DeepMind and Meta (whose 2024 scope is the predecessor org, Meta&nbsp;AI&thinsp;/&thinsp;GenAI)
      &mdash; is a Monte Carlo combination of a few sampled inputs. Rows show every quantity
      in model order: a dot at the median with a whisker spanning the 90% credible interval,
      grouped by unit onto a shared scale (5,000 samples each). Sampled inputs are priors from
      <code>lab_model_params.csv</code>; the model structure lives in
      <code>frontier_lab_compute_model.py</code>. OpenAI&rsquo;s 2023&ndash;24 bars are per-year
      outputs of the same power model walked through below.
    </p>
    <nav class="labs" id="labnav" aria-label="Jump to a model walkthrough"></nav>
  </section>

  <div id="sections"></div>

  <footer>
    <span>
      Source: Epoch AI, frontier-lab compute models &middot; consolidated layout mockup,
      not a published page &middot; tables:
      <a href="data/lab_compute_year_end_by_lab.csv">year-end CSV</a> &middot;
      <a href="data/lab_compute_intermediates_by_lab.csv">intermediates CSV</a> &middot;
      generated __GENERATED__
    </span>
    <span class="wordmark">EPOCH&thinsp;AI</span>
  </footer>
</div>

<div class="tooltip" id="tip-chart" role="tooltip"></div>
<div class="tooltip" id="tip-step" role="tooltip"></div>

<script>
// ── Data (injected by build_compute_page.py from generate_tables) ───────────
const DATA = __DATA_JSON__;
const YEAR_END = DATA.yearEnd;

// Fixed lab order: sets legend order and bar order within a year group.
// Color follows the lab everywhere, via the CSS custom properties above.
const LAB_ORDER = ['OpenAI', 'Google DeepMind', 'Anthropic', 'Meta Superintelligence Labs'];
const slug = lab => lab.toLowerCase().replace(/[^a-z]+/g, '-');

// Walkthrough section ids that actually exist, e.g. "openai-2025".
const SECTION_IDS = new Set(DATA.labs.map(e => `${slug(e.lab)}-${e.year}`));
// A bar's walkthrough target: its own (lab, year) section if present, else the
// lab's end-2025 section (OpenAI 2023-24 come from the same per-year model).
function walkthroughTarget(lab, year) {
  const own = `${slug(lab)}-${year}`;
  return SECTION_IDS.has(own) ? { id: own, sameYear: true }
                              : { id: `${slug(lab)}-2025`, sameYear: false };
}
const BACKCAST_NOTE = {
  'google-deepmind-2024': 'End-2024 backcast',
  'meta-superintelligence-labs-2024': 'End-2024 backcast · Meta AI / GenAI scope',
};

// ── Formatting ──────────────────────────────────────────────────────────────
const fmtH100e = v => v >= 1e6 ? (v / 1e6).toFixed(2).replace(/0$/, '') + 'M'
                               : Math.round(v / 1e3) + 'k';
function fmtBig(v) {
  if (Math.abs(v) >= 1e6) { const m = v / 1e6; return (m >= 10 ? m.toFixed(1) : m.toFixed(2)) + 'M'; }
  if (Math.abs(v) >= 1e3) return Math.round(v / 1e3) + 'k';
  return String(Math.round(v));
}
function fmtVal(units, v) {
  switch (units) {
    case 'H100e':
    case 'chips':        return fmtBig(v);
    case 'MW':           return Math.round(v).toLocaleString('en-US') + ' MW';
    case 'share':        return (v * 100).toFixed(v * 100 < 10 ? 1 : 0) + '%';
    case 'ratio':        return v.toFixed(2) + '×';
    case 'quarters':     return v.toFixed(1) + 'q';
    case 'USD B/yr':     return '$' + v.toFixed(2) + 'B/yr';
    case 'USD/H100e-hr': return '$' + v.toFixed(2) + '/hr';
    case 'H100e/MW':     return Math.round(v).toLocaleString('en-US');
    default:             return fmtBig(v);
  }
}

// ── Legend + data table ─────────────────────────────────────────────────────
const legend = document.getElementById('legend');
for (const lab of LAB_ORDER) {
  const item = document.createElement('span');
  item.className = 'legend-item';
  item.innerHTML = `<span class="swatch bg-${slug(lab)}"></span>${lab}`;
  legend.appendChild(item);
}
const tbody = document.querySelector('#values-table tbody');
for (const lab of LAB_ORDER) {
  for (const d of YEAR_END.filter(r => r.lab === lab).sort((a, b) => a.year - b.year)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${d.lab}</td><td>${d.year}</td>` +
      [d.p5, d.med, d.p95].map(v => `<td>${Math.round(v).toLocaleString('en-US')}</td>`).join('');
    tbody.appendChild(tr);
  }
}

// ── Chart ───────────────────────────────────────────────────────────────────
const YEARS = [...new Set(YEAR_END.map(d => d.year))].sort();
const Y_MAX = 3e6;
const TICKS = [0, 0.5e6, 1e6, 1.5e6, 2e6, 2.5e6, 3e6];
const W = 860, H = 430;
const M = { l: 64, r: 14, t: 18, b: 46 };
const plotW = W - M.l - M.r, plotH = H - M.t - M.b;
const y = v => M.t + plotH * (1 - v / Y_MAX);

const svg = document.getElementById('chart');
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, text) => {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  svg.appendChild(node);
  return node;
};

for (const t of TICKS) {
  el('line', { class: 'gridline' + (t === 0 ? ' zero' : ''),
               x1: M.l, x2: W - M.r, y1: y(t), y2: y(t) });
  el('text', { class: 'tick-label', x: M.l - 9, y: y(t) + 4, 'text-anchor': 'end' },
     t === 0 ? '0' : (t / 1e6) + 'M');
}
const yMid = M.t + plotH / 2;
el('text', { class: 'axis-title', x: 16, y: yMid, 'text-anchor': 'middle',
             transform: `rotate(-90 16 ${yMid})` }, 'H100 equivalents');

// ── Chart tooltip: anchored to the bar and hoverable, so its link is usable ─
const tipChart = document.getElementById('tip-chart');
let tipHideTimer = null;
function showChartTip(d, bar) {
  clearTimeout(tipHideTimer);
  const target = walkthroughTarget(d.lab, d.year);
  const note = BACKCAST_NOTE[`${slug(d.lab)}-${d.year}`];
  tipChart.innerHTML =
    `<span class="swatch bg-${slug(d.lab)}" style="display:inline-block;margin-right:6px"></span>` +
    `<b>${d.lab}</b><br>` +
    `<span class="muted">End of ${d.year}${note ? ' · ' + note.replace('End-2024 backcast', 'backcast') : ''}</span><br>` +
    `Median: ${fmtH100e(d.med)} H100e<br>` +
    `<span class="muted">90% CI ${fmtH100e(d.p5)} – ${fmtH100e(d.p95)}</span><br>` +
    `<a class="tip-link" href="#${target.id}">` +
    (target.sameYear ? 'How this estimate is built ↓'
                     : 'OpenAI model walkthrough ↓') + '</a>';
  tipChart.style.display = 'block';
  const r = bar.getBoundingClientRect(), t = tipChart.getBoundingClientRect();
  let left = r.left + r.width / 2 - t.width / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - t.width - 8));
  let top = r.top - t.height - 10;
  if (top < 8) top = r.bottom + 10;
  tipChart.style.left = left + 'px';
  tipChart.style.top = top + 'px';
}
function scheduleChartTipHide() {
  clearTimeout(tipHideTimer);
  tipHideTimer = setTimeout(() => { tipChart.style.display = 'none'; }, 250);
}
tipChart.addEventListener('mouseenter', () => clearTimeout(tipHideTimer));
tipChart.addEventListener('mouseleave', scheduleChartTipHide);
tipChart.addEventListener('click', e => {
  if (e.target.closest('a')) tipChart.style.display = 'none';
});

const groupW = plotW / YEARS.length;
const BAR_W = 42, GAP = 10;
YEARS.forEach((year, gi) => {
  const rows = LAB_ORDER
    .map(lab => YEAR_END.find(d => d.lab === lab && d.year === year))
    .filter(Boolean);
  const totalW = rows.length * BAR_W + (rows.length - 1) * GAP;
  const x0 = M.l + gi * groupW + (groupW - totalW) / 2;

  rows.forEach((d, i) => {
    const x = x0 + i * (BAR_W + GAP);
    const cx = x + BAR_W / 2;
    const target = walkthroughTarget(d.lab, d.year);

    const bar = el('rect', {
      class: `bar lab-${slug(d.lab)}`, tabindex: 0, role: 'link',
      'aria-label': `${d.lab}, end of ${d.year}: median ${fmtH100e(d.med)} H100 equivalents, ` +
        `90% CI ${fmtH100e(d.p5)} to ${fmtH100e(d.p95)}. View model walkthrough.`,
      x, y: y(d.med), width: BAR_W, height: y(0) - y(d.med),
    });
    const go = () => {
      tipChart.style.display = 'none';
      const section = document.getElementById(target.id);
      history.pushState(null, '', '#' + target.id);
      section.scrollIntoView();
      section.focus({ preventScroll: true });
    };
    bar.addEventListener('mouseenter', () => showChartTip(d, bar));
    bar.addEventListener('mouseleave', scheduleChartTipHide);
    bar.addEventListener('click', go);
    bar.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); }
    });
    bar.addEventListener('focus', () => { bar.classList.add('focused'); showChartTip(d, bar); });
    bar.addEventListener('blur', () => { bar.classList.remove('focused'); scheduleChartTipHide(); });

    el('line', { class: 'whisk', x1: cx, x2: cx, y1: y(d.p5), y2: y(d.p95) });
    for (const v of [d.p5, d.p95]) {
      el('line', { class: 'whisk', x1: cx - 6, x2: cx + 6, y1: y(v), y2: y(v) });
    }
    el('text', { class: 'bar-value', x: cx, y: y(d.p95) - 8, 'text-anchor': 'middle' },
       fmtH100e(d.med));
  });

  el('text', { class: 'year-label', x: M.l + gi * groupW + groupW / 2, y: H - 20,
               'text-anchor': 'middle' }, year);
});

// ── Walkthrough sections ────────────────────────────────────────────────────
const FACTOR_UNITS = new Set(['share', 'ratio', 'quarters']);
const family = u => FACTOR_UNITS.has(u) ? 'factor' : u;
const FAMILY_TITLES = {
  'H100e':        'H100 equivalents',
  'factor':       'Adjustment factors (shares, ratios)',
  'MW':           'IT power',
  'chips':        'Chip counts',
  'USD B/yr':     'Cloud rental spend',
  'USD/H100e-hr': 'Rental price',
  'H100e/MW':     'Fleet efficiency (H100e per MW)',
};
const TICK_SUFFIX = { 'H100e': ' H100e', 'chips': ' chips', 'MW': ' MW',
                      'USD B/yr': 'B/yr', 'USD/H100e-hr': '/hr',
                      'H100e/MW': ' H100e/MW' };
function fmtTick(axis, fam, v, isLast) {
  if (v === 0) return '0';
  if (axis.percent) return Math.round(v * 100) + '%';
  let num;
  if (v >= 1e6) num = parseFloat((v / 1e6).toFixed(2)) + 'M';
  else if (v >= 1e4) num = parseFloat((v / 1e3).toFixed(1)) + 'k';
  else num = (+v.toFixed(2)).toLocaleString('en-US');
  if (fam === 'USD B/yr' || fam === 'USD/H100e-hr') num = '$' + num;
  return isLast ? num + (TICK_SUFFIX[fam] ?? '') : num;
}

const tipStep = document.getElementById('tip-step');
function showStepTip(html, x, yPos) {
  tipStep.innerHTML = html;
  tipStep.style.display = 'block';
  const r = tipStep.getBoundingClientRect();
  tipStep.style.left = Math.min(x + 12, window.innerWidth - r.width - 8) + 'px';
  tipStep.style.top = Math.max(yPos - r.height - 10, 8) + 'px';
}
const hideStepTip = () => { tipStep.style.display = 'none'; };

function groupSteps(steps) {
  const groups = [];
  for (const step of steps) {
    const fam = family(step.units);
    const last = groups[groups.length - 1];
    if (last && last.fam === fam) last.steps.push(step);
    else groups.push({ fam, steps: [step] });
  }
  return groups;
}
function niceCeil(v) {
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]) {
    if (m * e >= v - 1e-9) return m * e;
  }
  return 10 * e;
}
const SEGMENTS = { 1: 4, 1.2: 3, 1.5: 3, 2: 4, 2.5: 5, 3: 3, 4: 4, 5: 5, 6: 3, 8: 4 };
function axisFor(rawMax, fam) {
  const percent = fam === 'factor' && rawMax <= 1.0;
  const max = percent ? 1.0 : niceCeil(rawMax * 1.02);
  const mantissa = +(max / Math.pow(10, Math.floor(Math.log10(max)))).toFixed(2);
  const n = SEGMENTS[mantissa] ?? 4;
  const ticks = Array.from({ length: n + 1 }, (_, i) => max * i / n);
  return { max, ticks, percent };
}
function familyAxes(steps) {
  const rawMax = {};
  for (const s of steps) {
    const fam = family(s.units);
    rawMax[fam] = Math.max(rawMax[fam] ?? 0, s.p95);
  }
  return Object.fromEntries(Object.entries(rawMax).map(([fam, m]) => [fam, axisFor(m, fam)]));
}

function trackCell(step, axis) {
  const track = document.createElement('div');
  track.className = 'track';
  const pct = v => Math.min(v / axis.max * 100, 100).toFixed(2) + '%';

  for (const t of axis.ticks) {
    if (t === 0 || t === axis.max) continue;
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = pct(t);
    track.appendChild(tick);
  }
  const rail = document.createElement('div');
  rail.className = 'rail';
  track.appendChild(rail);

  const isConst = step.kind === 'constant';
  if (!isConst) {
    const ci = document.createElement('div');
    ci.className = 'ci';
    ci.style.left = pct(step.p5);
    ci.style.width = Math.max((step.p95 - step.p5) / axis.max * 100, 0).toFixed(2) + '%';
    track.appendChild(ci);
    for (const v of [step.p5, step.p95]) {
      const cap = document.createElement('div');
      cap.className = 'cap';
      cap.style.left = pct(v);
      track.appendChild(cap);
    }
  }
  const dot = document.createElement('div');
  dot.className = 'dot';
  dot.style.left = pct(step.med);
  track.appendChild(dot);

  track.addEventListener('mousemove', e => {
    const body = isConst
      ? `<b>${fmtVal(step.units, step.med)}</b> <span class="muted">(fixed value)</span>`
      : `median <b>${fmtVal(step.units, step.med)}</b><br>` +
        `<span class="muted">90% CI ${fmtVal(step.units, step.p5)} &ndash; ` +
        `${fmtVal(step.units, step.p95)}</span>`;
    const expr = step.expression ? `<br><span class="muted">= ${step.expression}</span>` : '';
    showStepTip(`<b>${step.label}</b><br>${body}${expr}`, e.clientX, e.clientY);
  });
  track.addEventListener('mouseleave', hideStepTip);
  return track;
}

function stepRow(step, axis) {
  const row = document.createElement('div');
  row.className = 'row' + (step.kind === 'final' ? ' final' : '');

  const lbl = document.createElement('div');
  lbl.className = 'lbl';
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${step.i} · ${step.kind}`;
  lbl.appendChild(meta);
  const name = document.createElement('div');
  name.className = 'name';
  name.textContent = step.label;
  lbl.appendChild(name);
  if (step.expression) {
    const expr = document.createElement('div');
    expr.className = 'expr';
    expr.textContent = '= ' + step.expression;
    lbl.appendChild(expr);
  }
  row.appendChild(lbl);
  row.appendChild(trackCell(step, axis));

  const val = document.createElement('div');
  val.className = 'val';
  val.innerHTML = step.kind === 'constant'
    ? `<b>${fmtVal(step.units, step.med)}</b> (fixed)`
    : `${fmtVal(step.units, step.p5)}&ndash;${fmtVal(step.units, step.p95)} · ` +
      `<b>${fmtVal(step.units, step.med)}</b>`;
  row.appendChild(val);
  return row;
}

const nav = document.getElementById('labnav');
const sections = document.getElementById('sections');
for (const entry of DATA.labs) {
  const id = `${slug(entry.lab)}-${entry.year}`;
  const labSlug = slug(entry.lab);

  const link = document.createElement('a');
  link.href = '#' + id;
  link.innerHTML = `<span class="swatch bg-${labSlug}"></span>${entry.lab} &middot; end-${entry.year}`;
  nav.appendChild(link);

  const section = document.createElement('section');
  section.className = `card lab sec-${labSlug}`;
  section.id = id;
  section.tabIndex = -1;

  const final = entry.steps.find(s => s.kind === 'final');
  const note = BACKCAST_NOTE[id];
  const header = document.createElement('header');
  header.innerHTML =
    `<span class="swatch bg-${labSlug}"></span>` +
    `<h3>${entry.lab}</h3>` +
    (final ? `<span class="headline">end-${entry.year}: <b>${fmtVal('H100e', final.med)}</b>` +
             ` H100e (90% CI ${fmtVal('H100e', final.p5)}&ndash;${fmtVal('H100e', final.p95)})</span>`
           : '') +
    (note ? `<span class="badge">${note}</span>` : '') +
    `<a class="backlink" href="#chart-card">↑ back to chart</a>`;
  section.appendChild(header);

  const axes = familyAxes(entry.steps);
  for (const group of groupSteps(entry.steps)) {
    const axis = axes[group.fam];
    const div = document.createElement('div');
    div.className = 'group';

    const ghead = document.createElement('div');
    ghead.className = 'ghead';
    ghead.innerHTML = `<span class="gkicker">${FAMILY_TITLES[group.fam] ?? group.fam}</span>`;
    div.appendChild(ghead);

    for (const step of group.steps) div.appendChild(stepRow(step, axis));

    const gfoot = document.createElement('div');
    gfoot.className = 'gfoot';
    const axisrow = document.createElement('div');
    axisrow.className = 'axisrow';
    axis.ticks.forEach((t, ti) => {
      const isLast = ti === axis.ticks.length - 1;
      const left = (t / axis.max * 100).toFixed(2) + '%';
      const stub = document.createElement('div');
      stub.className = 'stub';
      stub.style.left = left;
      axisrow.appendChild(stub);
      const label = document.createElement('span');
      label.className = 'tlabel';
      label.style.left = left;
      if (ti === 0) label.style.transform = 'translateX(0)';
      if (isLast) label.style.transform = 'translateX(-100%)';
      label.textContent = fmtTick(axis, group.fam, t, isLast);
      axisrow.appendChild(label);
    });
    gfoot.appendChild(document.createElement('span'));
    gfoot.appendChild(axisrow);
    gfoot.appendChild(document.createElement('span'));
    div.appendChild(gfoot);

    section.appendChild(div);
  }
  sections.appendChild(section);
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
