# Monte Carlo compute models: Google DeepMind, Meta SL, OpenAI, Anthropic

Describes the four lab-level Monte Carlo models estimating **end-2025 compute in H100-equivalents (H100e)**. **This is a prose artifact, not a source of truth**: the canonical judgment priors live in `lab_model_params.csv` (read by the per-lab notebooks and `frontier_lab_compute_model.py` alike), and the notebooks (`deepmind_compute_model`, `msl_compute_model`, `openai_compute_monte_carlo`, `anthropic_compute_monte_carlo`) hold the detailed walkthroughs and sensitivity sweeps. Where this document disagrees with the sheet or the code, trust the sheet and the code. Values below are as of **2026-07-02**.

## Shared conventions

| Convention | Value | Notes |
|---|---|---|
| H100e definition | chip's dense 8-bit peak FLOP/s ÷ 1.979e15 | H100 dense FP8 spec is the denominator |
| Input distributions | lognormal unless noted | `sq.to(low, high)` bounds = **90% credible interval** (5th–95th pct); median ≈ geometric mean of bounds |
| Samples / seed | 5,000 / seed 42 (squigglepy) | |
| Canonical model choice | **power-based** models for OpenAI and Anthropic | cloud-spend analyses (`anthropic_cloud_spend_monte_carlo`, gpu-hour model) deliberately excluded from the canonical script |
| Ownership data basis | "sold" stock from AI Chip Owners/Sales dashboards | converted to operational stock via a deployment-lag ratio |

The four models fall into two families. DeepMind and MSL are **top-down allocations**: owned fleet × deployment lag × lab share. OpenAI and Anthropic are **power-based**: disclosed/leaked IT power × a chip mix converting MW to H100e.

## Headline results (canonical script, 5th / median / 95th)

| Lab | 5th | Median | 95th |
|---|---|---|---|
| Google DeepMind | 1.07M | 1.60M | 2.45M |
| Meta SL | 531k | 892k | 1.52M |
| OpenAI | 1.44M | 1.77M | 1.94M |
| Anthropic | 941k | 1.22M | 1.58M |

---

## 1. Google DeepMind

**Structure:** `DeepMind H100e = (Nvidia + TPU owned) × deployment_lag × [cloud_share · dm_cloud_share + (1 − cloud_share) · dm_noncloud_share]`

DeepMind's fraction blends two sub-shares because Google's fleet splits into a cloud half (external rentals + enterprise Gemini) and an internal half (consumer Gemini, DM R&D, recommenders, Waymo).

| Parameter | Distribution (90% CI) | Reasoning |
|---|---|---|
| Nvidia owned | lognormal 955k – 1.59M H100e (median ~1.24M) | AI Chip Owners dashboard CI. Drawn independently of TPUs — a simplification; arguments exist for correlation (shared error about Google totals) and anti-correlation (GPU vs TPU purchase trade-offs). Author judges dashboard CIs arguably too narrow. |
| Google TPU owned | lognormal 3.08M – 4.54M H100e (median ~3.74M) | Same source and caveats. |
| Deployment lag ratio | lognormal 0.55 – 0.87 (median ~0.69) | Operational/owned ratio implied by a 0.5–2 quarter install lag mapped through Google's quarterly stock growth (each extra quarter of lag shaves ~1.1M H100e). 0.5q → 0.87, 1q → 0.76, 2q → 0.55; the median puts the 1-quarter scenario near the center. |
| Cloud share of Google ML compute | lognormal 0.45 – 0.55 (median ~0.50) | CFO: "around half" of ML compute serves cloud in 2025, "just over half" guided for 2026. Tight band around the statement. |
| DM share of the cloud half | lognormal 0.2 – 0.6 (median ~0.35) | The cloud half (~1.6–2M H100e) contains external GPU/TPU rentals (e.g. to Anthropic — not DeepMind compute) plus enterprise Gemini inference. Enterprise Gemini likely ran below OpenAI/Anthropic inference, i.e. under ~1M H100e, so the share is probably under 0.5. |
| DM share of the internal half | lognormal 0.4 – 0.8 (median ~0.57; raised from 0.33–0.75 on 7/1) | Explicitly "vibe-sy." Upward pull: DeepMind is a top compute priority and its R&D is plausibly provisioned like peer frontier labs (~1.5–2M H100e). Downward pull: recommenders (~1M H100e ballpark, supporting a ~$200B ads business) plus ML across Search, Gmail, Maps, Waymo, etc. |

**Result:** 1.07M / 1.60M / 2.45M H100e. DeepMind's implied share of operational Google compute: 0.35–0.64 (median 0.47).

**Sensitivity checks (notebook):** doubling the TPU CI's log-width (→2.54M–5.51M) widens the DeepMind 90% CI only +15% with the median unchanged — the product-of-factors structure dilutes any one input. Correlating the two DM sub-shares at ρ=0.5 (plausible, since both hinge on the same "what counts as DeepMind" question) widens the CI +13%, median unchanged; the independent baseline slightly understates tail uncertainty.

---

## 2. Meta Superintelligence Labs (MSL)

**Structure:** `MSL H100e = (Nvidia + AMD owned) × deployment_lag × MSL_share`

Simpler than DeepMind: Meta's fleet is almost entirely internal at end-2025 (no meaningful external rentals; its own Google/Oracle/CoreWeave rental deals, $10–20B signed late 2025–2026, hadn't ramped), so owned ≈ used and there is no cloud split. MTIA excluded (low volumes through 2025).

| Parameter | Distribution (90% CI) | Reasoning |
|---|---|---|
| Nvidia owned | lognormal 1.43M – 2.38M H100e (median ~1.84M) | AI Chip Owners dashboard, end-2025 sold basis. Independent of AMD draw (same correlation caveats as DeepMind). |
| AMD Instinct owned | lognormal 345k – 612k H100e (median ~460k) | Same source; AMD ≈ 20% of Meta's total. |
| Deployment lag ratio | lognormal 0.62 – 0.90 (median ~0.75) | Meta's stock grew ~0.4–0.5M H100e/quarter through 2025 (~700k end-2024 → ~1.42M Q2 → ~1.83M Q3 → ~2.30M Q4). Mapping a 0.5–2 quarter lag onto that trajectory: 0.5q → 0.90, 1q → 0.79, 2q → 0.62. Ratios sit higher than DeepMind's because the faster ramp means recent purchases are a bigger fleet share. |
| MSL share of operational compute | lognormal **0.33 – 0.80**, clipped to [0.1, 0.9] (median ~0.51) | Frontier-AI work (MSL training/R&D + Meta AI inference) vs core-business ML (ad/feed recommenders). Centered ~50:50: mid-2025 analyst estimates put 50–60% of GPUs on recommenders, and Q1-2025 capex guidance sent the majority to core business — but the frontier share climbed fast (the late-2024 100k-H100 training cluster was only ~15% of the then-fleet; MSL's mid-2025 formation and late-2025 frontier pivot tilted allocation further). Hard-capped at 90:10 either way. Flagged as highly uncertain and sensitive to what counts as MSL. |

**Note:** the notebook and the script share this prior via `lab_model_params.csv`. The sheet widened it on 7/2 from an earlier 0.40–0.60 baseline (kept in the notebook as the "tight" sensitivity row); the widening mostly stretches the upper tail (95th ≈ 1.52M vs ~1.21M under the tight band), with medians nearly unmoved (~850–890k).

**Result:** 531k / 892k / 1.52M H100e.

**Sensitivity checks (notebook):** widening the share CI all the way to uniform(0.1, 0.9) leaves the median pinned near ~850k and tops out ~1.6M at the 95th — still below OpenAI's central ~1.7–1.8M. Fixing discrete lags of 0–3 quarters moves the median across 544k–1.13M, making lag the bigger lever on the central value.

---

## 3. OpenAI (power-based)

**Structure:** convert OpenAI's disclosed data-center IT power into chips via Microsoft's chip deployment mix, then into H100e. Disclosed power: **200 MW (end-2023), 600 MW (end-2024), 1,900 MW (end-2025)**, interpreted as IT power. Each year's power *increment* is assigned the mix of chips Microsoft was deploying at that time (100% Nvidia assumed; no retirements; the first step uses cumulative mix so legacy A100s persist).

Two fleet constructions are built from the mix and blended: **default** (vintage-layered — each year's additions keep their vintage mix and carry forward) and **newest** (whole fleet on the latest year's mix). Fixed conversion constants, from `IT power by chip.csv` and Microsoft's fleet snapshot: IT watts/GPU A100 926, H100/H200 1,389, GB200 2,083, GB300 2,222; H100e/GPU A100 0.315, H100/H200 1.0, B200/B300 2.527.

| Parameter | Distribution | Reasoning |
|---|---|---|
| `new_chip_share` (mix blend) | Beta(2, 6), mean 0.25 | Weight on the "newest" fleet construction. Most of OpenAI's capacity sits on longer-term contracts that don't refresh to the newest chips, so the prior leans toward the vintage-layered default. Notebook sweeps Beta(2,18) → Beta(6,2): 2025 medians move only 1.75M → 1.90M, so the headline is robust to this prior; it mainly reshapes the A100/Blackwell composition. |
| Deployment lag | lognormal 0.5 – 2.0 quarters (median ~1) | Nvidia books revenue at shipment; chips go live quarters later. The lag sets which (interpolated) Microsoft snapshot each year's mix is read from. Anchors: CoreWeave installs took ~3 months in early 2025, "within weeks" by late 2025. |
| Power definition factor | mixture: 80% IT `sq.to(0.95, 1.05)` + 20% gross `1 / sq.to(1.10, 1.30)`; one draw shared across all years | Is each disclosed figure IT power or gross (facility) power? 80% it's already IT (a tight residual band around 1.0); 20% it's gross, so true IT is lower by a 1.1–1.3 datacenter PUE. One-sided — it can only pull the total down (median factor ~0.99, mean ~0.97). Shared because a misreading would bias every year the same way. |
| Rounding jitter | triangular ±50 MW (peaked at 0), independent per year | Disclosures are rounded to 0.1 GW, so truth sits within ±50 MW; triangular rather than uniform because the exact edges are less likely (OpenAI may have rounded differently there). Same absolute band ⇒ 2023's 200 MW is the least precise in relative terms. |

**Result:** 1.44M / 1.77M / 1.94M H100e end-2025 (main-model deterministic reference: 1.87M). Implied end-2025 count mix (notebook): A100 ~8%, Blackwell ~42% of chips.

**Uncertainty decomposition (notebook):** the power definition factor still dominates the total's spread and is now one-sided — its 90% width (~440k) is roughly 2× the lag's and 4× `new_chip_share`'s, and it reaches ~330k below the median but only ~110k above (the 20% gross tail). `new_chip_share` and lag mostly move composition, not the total. A dedicated sweep of the gross probability (§8) slides the 2025 median from 1.81M at 0% gross to 1.54M at 80% gross while the 95th stays near ~1.9M — so the facility-vs-IT interpretation risk, previously flagged as uncovered, is now modeled explicitly and lands almost entirely on the downside.

---

## 4. Anthropic (power-based)

**Structure:** `Anthropic H100e = total IT power × blended H100e-per-MW`, where the blend is set by one lever — the **Trainium2 share of power**. Justification for the two-bucket collapse: at fixed power, the Nvidia mix (991 H100e/MW) and Google's real v5+ TPU fleet mix (1,018 H100e/MW, native 8-bit scoring, IT overhead 1.742× TDP) buy nearly the same compute per watt (within ~3%), while Trainium2 buys 754 H100e/MW (~0.76×). So non-Trainium compute is treated as one bucket at the midpoint (~1,005 H100e/MW), and the Nvidia:TPU split becomes irrelevant to the total. No deployment-lag machinery: the anchor power figure is described as already online.

Chip constants are **borrowed from the OpenAI model** for consistency: H100/GB200 watts and H100e ratios, plus OpenAI's end-2025 Hopper:Blackwell count ratio (~1.24:1, A100 dropped, GB300 folded into Blackwell) to weight the Nvidia mix. Trainium2: 0.656 H100e/chip (1,299 vs 1,979 TFLOP/s dense 8-bit); watts/chip ≈ 871, pinned by a mid-2025 Project Rainier (New Carlisle) snapshot — a 300k-H100-eq Trainium2 fleet draws 398 MW of IT power ([Epoch AI data-center directory, Anthropic–Amazon New Carlisle](https://epoch.ai/data/ai-data-centers/directory/anthropic-amazon-new-carlisle)) — making a Trainium2 watt marginally better than an H100's, though still ~0.75× the Blackwell-heavy Nvidia/TPU mixes.

| Parameter | Distribution (90% CI) | Reasoning |
|---|---|---|
| Total IT power | lognormal 1.4²/1.8 ≈ 1.09 – 1.8 GW (median 1.4 GW) | Leaked OpenAI internal memo put Anthropic at ~1.4 GW online end-2025. Upper bound ~1.8 GW sits just under OpenAI's own 1.9 GW, since the memo was confident OpenAI was ahead; the lower bound then follows mechanically from the lognormal median = geometric mean of bounds. Residual risk: if 1.4 GW were facility power, true IT power (~1.0 GW) would fall below this floor. |
| Trainium2 share of IT power | ~normal 0.35 – 0.70 (median ~0.52), clipped to [0.1, 0.9] | Anchored on site power in Epoch's data-center directory: New Carlisle (Project Rainier) stepped to ~626 MW of IT power in late December 2025 (possibly still ~400 MW at year-end if that step slipped), and the Amazon Madison campus (MS) held ~284 MW from mid-2025 — Trainium2 by its compute-to-power ratio, though perhaps not all Anthropic's. Together up to ~900 MW; residual Trainium2 beyond the two sites brings the high case to ~1.0 GW ≈ 70% of the central 1.4 GW (also near a prior ceiling, given Anthropic's known Nvidia and TPU fleets). Low case (~400 MW New Carlisle + partial Madison) ≈ 500 MW ≈ 35%. Amazon's total ~1.4M deployed Trainium2 ("fully subscribed") remains a hard chip ceiling. Sampled independently of power (scale and mix are largely separate questions). |

**Cross-check (notebook):** implied Trainium2 chips 532k / 844k / 1.24M; median = 60% of Amazon's 1.4M and sits between the New Carlisle-only (~720k) and +Madison (~1.05M) anchors; 1.3% of samples breach the ceiling.

**Result:** 946k / 1.22M / 1.57M H100e — ~70% of OpenAI's median, centered in the research summary's "≥1M, likely <1.5M" (the upper tail pokes just past 1.5M). Power contributes roughly 5× the spread of the share prior, so tightening the power figure is the highest-leverage refinement.

---

## Cross-model notes

The models are deliberately interlocked: Anthropic borrows OpenAI's chip specs and Hopper:Blackwell ratio (the canonical script passes `model_openai()`'s output into `model_anthropic()`), and DeepMind/MSL share the same dashboard-CI + deployment-lag machinery. Common caveats: dashboard CIs may be too narrow; independent draws of correlated quantities (Nvidia/TPU, Nvidia/AMD, the two DM shares, power/share for Anthropic) somewhat understate tail risk; the lab-share parameters for DeepMind and MSL are the least evidenced inputs and are definitionally sensitive (what counts as "the lab"); and the IT-vs-facility-power interpretation risk lands on the downside of both power-based models — now modeled explicitly for OpenAI (an 80/20 IT-vs-gross mixture on the power figure) but still an unmodeled residual for Anthropic.

Excluded by design from the canonical script: the OpenAI and Anthropic cloud-spend models (corroboration only — e.g. OpenAI 2025 spend implies ~1.0–1.3M H100e annual average, consistent with the power model's trajectory).

*Reproduction note: results above regenerated 2026-07-02 from `frontier_lab_compute_model.py` (seed 42, 5,000 samples), with priors from `lab_model_params.csv` and the TPU mix computed from `csv_export/tpu_cumulative_by_chip.csv`.*
