# Archived analyses

Superseded or one-off work, frozen as-is. Paths and imports inside these files
predate the `data/` reorg (they expect the CSVs next to them and assume running
from `ai-lab-compute/`), so they won't run from here without minor path fixes.

- `openai_power_model.ipynb` — original point-estimate OpenAI power→fleet model.
  Superseded by `../openai_compute_monte_carlo.ipynb`, which reproduces it
  exactly at `new_chip_share = 0`, zero lag, and nominal power.
- `gpu_hour_model.ipynb` — one-off: converts a lab's compute spending into
  GPU-hours and average H100e under four Hopper/Blackwell mix scenarios.
- `coreweave_it_power_point_estimate.py` — one-off point estimate of CoreWeave's
  fleet from IT power and Nvidia unit mix. The maintained CoreWeave model is
  `coreweave_estimate.ipynb` at the repo root.
