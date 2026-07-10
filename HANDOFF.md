# HANDOFF — current state for agent sessions

Read this before starting work. Before ending a session that changed models,
params, or data: add a dated one-liner under "Recently landed", update or
delete anything that no longer describes reality, and keep this file under
~40 lines. Its git history is the journal — prune freely.

## Source-of-truth map
- Judgment priors + shared chip IT-power specs (`chip_specs` rows):
  `ai-lab-compute/lab_model_params.csv`, read by the lab notebooks and
  `frontier_lab_compute_model.py` (loader in `lab_compute_utils`).
- Model structure: the jupytext-paired notebook `.py` files + the frontier script.
- `ai-lab-compute/docs/` holds prose artifacts (`monte-carlo-compute-models.md`,
  `frontier-lab-compute-modeling-summary.md`), not sources of truth — trust the
  sheet and code over them.

## In flight
- openai_compute_monte_carlo iterated heavily 7/8 (power mixture, gross sweep,
  varying IT overhead) — may still be open in Jupyter; reload tabs from disk
  before editing, and edit the `.py`, not the `.ipynb`.

## Recently landed (prune entries older than ~2 weeks)
- 7/10: lab_compute_tables mockups consolidated into one page,
  build_compute_page.py → index.html: chart on top (bar tooltips + clicks link
  to each estimate's walkthrough section) with the per-snapshot walkthroughs
  below; light/dark themes, CVD-validated lab colors. Replaces
  chart_mockup.html, intermediates.html and build_intermediates_page.py; same
  data pipeline (a pure view over get_all_tables()).
- 7/10: GDM + Meta end-2024 backcasts promoted to canonical (Anthropic's
  deliberately stays notebook-only): five *_2024 sheet rows,
  model_deepmind_2024 / model_msl_2024 in the frontier script (end-2024 fleets
  read from the dashboard CSVs, data-driven lag ratio), and lab_compute_tables
  exports the (lab, 2024) rows — GDM 423k (215k–779k), Meta AI 248k
  (124k–480k) at seed 42 — in both CSVs and the consolidated page.
- 7/10: anthropic_2024_backcast notebook — consolidates the Anthropic end-2024
  backcast (was cloud-spend §9 + lab_2024_backcasts §5): plain-language
  walkthrough, printed/charted intermediates, SA cross-checks → 249k
  (121k–428k). cloud-spend §9 is now a pointer stub (that notebook is
  2025-only again); lab_2024_backcasts imports the samples via runpy.
- 7/10: lab_2024_backcasts notebook (jupytext-paired) — end-2024 walkthrough
  and cross-lab comparison for all four labs: OpenAI 386k (frontier per-date),
  Anthropic imported from anthropic_2024_backcast; the GDM/Meta sections call
  the canonical 2024 models (no restated priors).
- 7/10: lab models record intermediate step traces (MODEL_STEPS in the
  frontier script; pure bookkeeping, outputs unchanged, call sites untouched).
  lab_compute_tables exports them (intermediates_by_lab CSV) and renders them
  as the walkthrough sections of the consolidated page.
- 7/8: openai watts/GPU reworked to server power × a sampled IT overhead factor
  (new `chip_specs,nvidia_it_overhead` row; Colossus sheet median 1.14, 5th–95th
  1.0–1.35, floor 1.0). H100 overridden to server×overhead (1389→1453).
  `data/IT power by chip.csv` gains server-power + 5th/95th cols; median col
  unchanged (back-compat). Then added `openai,figure_accuracy` (to(0.9,1.2),
  median ~1.04): is the internal power figure itself right, aside from rounding
  and IT-vs-gross. OpenAI 2025 → 1.76M (1.18M–2.23M). Frontier re-run.
  (docs writeup NOT updated for these two changes per Josh — OpenAI section and
  Meta row both stale now.)
- 7/8: MSL rented-cloud term added (msl_compute_model §4): zero-inflated spend
  prior (2 new msl sheet rows) × GB200/GB300 rental conversion (InferenceX
  Aug-2025) → MSL 988k (602k–1.63M), was 892k. Frontier + tables re-run.
- 7/8: ai-lab-compute reorg — input CSVs → `data/` (README has provenance),
  prose → `docs/`, superseded one-offs → `archive/`; deleted the stale vendored
  chip_estimates_utils.py (root copy is the only one). Readers updated + re-run.
- 7/7: alphabet_activities Nvidia cost re-anchored on SemiAnalysis InferenceX
  owning TCO → single nvidia_tco sheet prior (0.9–1.4 $/H100e-hr) replaces the
  market-price × (1−margin) rows; Q1 2026 median 1.36M → 1.46M (+7%). Notebook
  gained an old-vs-new comparison section. Frontier script unaffected (doesn't
  read alphabet rows).
- 7/6: new ai-lab-compute/lab_compute_tables/ — ai-chip-components-style
  generator (get_year_end_by_lab: lab × year-end H100e table + data/ CSV);
  model_openai now also returns total_h100e_by_date (end-2025 unchanged).
- 7/3: anthropic trainium_share re-anchored on Epoch site power → (0.35, 0.7);
  Anthropic now 946k / 1.22M / 1.57M, implied chips median ~845k.

## Open decisions
- alphabet_activities compute_share CI (0.30–0.70, median 0.46) sits below the
  model's own sizing arithmetic (~0.6): shift up or add justification (one-cell
  sheet edit).
- Correlate the two DeepMind sub-shares (ρ≈0.5) in the headline model, or keep
  independent draws + the sensitivity check?

## Gotchas
- After editing the sheet, re-execute the affected notebook(s) AND the frontier
  script so outputs match; Excel rewrites its last_updated dates (harmless).
- `ai-lab-compute/docs/alphabet-level activities.docx` still says $5.3B for
  Q1 2026; 10-Q: $5,391M.
