# HANDOFF — current state for agent sessions

Read this before starting work. Before ending a session that changed models,
params, or data: add a dated one-liner under "Recently landed", update or
delete anything that no longer describes reality, and keep this file under
~40 lines. Its git history is the journal — prune freely.

## Source-of-truth map
- Judgment priors: `ai-lab-compute/lab_model_params.csv`, read by the lab
  notebooks and `frontier_lab_compute_model.py` (loader in `lab_compute_utils`).
- Model structure: the jupytext-paired notebook `.py` files + the frontier script.
- `monte-carlo-compute-models.md` and `frontier-lab-compute-modeling-summary.md`
  are prose artifacts, not sources of truth — trust the sheet and code over them.

## In flight
- openai_compute_monte_carlo: power-definition mixture (IT vs gross) + §8 sweep
  added 7/2, possibly still being iterated in Jupyter — reload tabs from disk
  before editing, and edit the `.py`, not the `.ipynb`.

## Recently landed (prune entries older than ~2 weeks)
- 7/2: msl_share widened in the sheet to (0.33, 0.8) (was 0.4–0.6); MSL notebook
  + frontier re-executed → MSL 892k median (531k–1.52M).
- 7/2: canonical priors moved into lab_model_params.csv; all five lab notebooks
  + frontier script read from it (convention documented in CLAUDE.md).
- 7/2: alphabet model Q1-2026 spend corrected to $5.391B per the 10-Q; share
  priors clipped at 1 → Q1 median ~1.35M avg concurrent H100e.
- 7/1: deepmind dm_noncloud_share raised to (0.4, 0.8) → DeepMind median 1.60M
  (90% CI 1.07M–2.45M).

## Open decisions
- alphabet_activities compute_share CI (0.30–0.70, median 0.46) sits below the
  model's own sizing arithmetic (~0.6): shift up or add justification (one-cell
  sheet edit).
- Correlate the two DeepMind sub-shares (ρ≈0.5) in the headline model, or keep
  independent draws + the sensitivity check?

## Gotchas
- After editing the sheet, re-execute the affected notebook(s) AND the frontier
  script so outputs match; Excel rewrites its last_updated dates (harmless).
- "alphabet-level activities.docx" still says $5.3B for Q1 2026; 10-Q: $5,391M.
