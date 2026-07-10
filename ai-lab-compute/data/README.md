# Data inputs for the lab compute models

Hand-maintained inputs:

- `IT power by chip.csv` — per-GPU IT power for Nvidia chips (canonical location
  per CLAUDE.md; TPU/Trainium power specs live in `../lab_model_params.csv` as
  `chip_specs` rows). Read by `frontier_lab_compute_model.py` and
  `openai_compute_monte_carlo`.
- `lab IT power.csv` — OpenAI's disclosed IT power by date. Read by the same two.

Pinned snapshots of root-notebook exports (deliberately frozen so the lab models
don't shift when the root notebooks re-run; refresh by copying from
`../../csv_export/` / `../../owners_csv_export/` when you want new fleet data):

- `nvidia_owners_cumulative_by_chip.csv` — from `owners_csv_export/`, produced by
  the root `nvidia_owners.ipynb`. Read by `frontier_lab_compute_model.py` and
  `openai_compute_monte_carlo`.
- `nvidia_calendar_quarter_chip_timelines.csv` — from `csv_export/`, produced by
  the root `nvidia_estimates.ipynb`. Only read by
  `archive/coreweave_it_power_point_estimate.py`.
- `nvidia_cumulative_by_chip.csv` — from `csv_export/`. **No current readers**;
  kept for now, candidate for deletion.
