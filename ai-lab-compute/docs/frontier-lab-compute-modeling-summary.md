# Frontier Lab Compute Modeling — Summary of Methods and Parameters

Summary of the modeling in Epoch AI's research appendix, "How much AI compute do frontier developers have?" (Josh You, last major update May 2026). Estimates concern compute **rented or used** (not owned) by the five most compute-rich frontier developers as of **end of 2025**, expressed in **H100-equivalents (H100e)**.

---

## Global conventions and shared parameters

These apply across all per-lab models.

| Parameter | Value | Notes |
|---|---|---|
| H100e definition | chip's dense FP8 (or INT8) peak FLOP/s ÷ 1979 TFLOP/s | 1979 TFLOP/s is the H100's dense FP8 spec |
| Cumulative **sold** AI compute, end-2025 | ~20M H100e | From Epoch's AI Chip Sales (vendor recognized revenue: Nvidia, Broadcom, AMD, etc.) |
| Sold stock, end Q3 2025 | ~16M H100e | |
| Sold stock, end Q2 2025 | ~12M H100e | |
| Deployment lag (sold → operational) | 0–2 quarters; **default 1 quarter** | Reference: CoreWeave disclosed ~3-month install time in early 2025, "within weeks" by late 2025 |
| Implied **deployed** world stock, end-2025 | 20M (0q) / 16M (1q) / 12M (2q) H100e | 1-quarter lag ⇒ deployed stock ~20% smaller than sold stock |
| Ownership uncertainty | ±20–30% per owner depending on chip type | |

**End-2025 ownership baseline (sold basis, from AI Chip Owners):**

| Owner | H100e |
|---|---|
| Google (Alphabet) | ~5.1M (of which ~3.8M TPU; TPU CI 3.1–4.5M) |
| Microsoft | ~3.5M |
| Amazon | ~2.5M |
| Meta | ~2.3–2.5M (excl. MTIA; ~20% AMD) |
| Oracle | ~1M |
| CoreWeave | ~850k |
| xAI/SpaceX | ~600k |
| **Sum** | ~16M (~80% of world total) |

---

## OpenAI

Two independent models; the **data center power model is primary** (more confident parameters, gives an end-of-year snapshot rather than an annual average).

### Model 1: Data center power model (primary)

**Inputs:**

| Parameter | Value |
|---|---|
| Disclosed power capacity, end-2023 | 0.2 GW |
| Disclosed power capacity, end-2024 | 0.6 GW |
| Disclosed power capacity, end-2025 | 1.9 GW |
| Power interpretation | **IT power** (servers + networking peak draw), not facility power |
| Facility-power alternative | Facility power can be up to ~40% above IT power; assuming facility power would cut compute estimates by up to ~30% |
| Chip vendor assumption | 100% Nvidia |
| Per-GPU IT power references | H100 ≈ 1450 W (server-level, per GPU); GB200 ≈ 2100 W per GPU |
| GB200 performance | ~2.5× H100 in rated FLOP/s |
| Deployment lag | 0–2 quarters sensitivity; default 1 quarter |
| Lag sensitivity (2025) | each extra quarter of lag lowers the H100e estimate by ~6–7% (older chip mix, fewer Blackwells) |
| Retirements | none assumed |
| Pre-history | end-2023 capacity assumed procured entirely in 2022–2023 (so legacy A100s persist through 2025) |

**Algorithm:**
1. Compute the year-over-year **increase** in disclosed GW.
2. Convert each year's power increment to chip counts using the **mix of flagship Nvidia chips sold in that period** (from Epoch AI Chip Sales), lagged by the deployment-lag parameter and weighted by per-chip IT power.
3. Convert chip counts to H100e via FLOP/s spec ratios; accumulate across years (no retirement).

**Outputs (1-quarter lag, central):**

| Year-end | H100e |
|---|---|
| 2023 | ~100k (102k) |
| 2024 | ~380k (377k) |
| 2025 | **~1.7M (1.72M)** |

Sanity bounds for end-2025: all-H100 fleet ⇒ ~1.3M H100s; all-GB200 fleet ⇒ ~900k GB200s ≈ 2.25M H100e. True fleet (Hopper/Blackwell mix + trace A100s) lies between.

### Model 2: Cloud spending model (corroboration)

**Inputs:**

| Parameter | Value |
|---|---|
| 2024 cloud compute spend | $5.8B reported (post-amortization of long-term research compute; $1B amortized item ⇒ ~$6.8B actual proxy) |
| 2025 cloud compute spend | $16.3B |
| Fleet simplification | H100 + GB200 only |
| Price scenario A (low) | Hopper $1.50/GPU-hr; Blackwell $3.00/GPU-hr (SemiAnalysis Aug-2025 3-yr-contract survey: H100 $1.30, H200 $1.60, B200 $2.90, GB200 $3.30) |
| Price scenario B (high) | Hopper $2.00/GPU-hr; Blackwell $4.00/GPU-hr (Silicon Data spot index, early 2026) |
| GB300 treatment | treat as GB200 (ignore FP4 +50%); use NVL72 "GB" specs and pricing throughout |
| Hours per year | 8760 (continuous rental implied) |

**Algorithm:** spend ÷ (price per GPU-hr × hours), allocated across the assumed Hopper/Blackwell mix; convert to H100e by spec ratio.

**Outputs (2025 annual average):** ~1.3M H100e (low price) or ~1.0M H100e (high price). For 2024: $6.8B all-Hopper at $2/hr ⇒ ~388k H100e average.

**Reconciliation:** power model endpoints (377k → 1.72M) imply a 2025 average of ~900k (logarithmic mean, exponential interpolation) or ~1.05M (linear). Cost model runs ~0–40% higher; 2024 discrepancy is larger (388k vs ~210k log-mean), suggesting reported cloud costs include secondary costs beyond raw GPU-hours. The power model is treated as more credible.

**World share:** ~1.7M / deployed world stock ⇒ 8.5% (0q lag), ~10–11% (1q), 14% (2q), ~18% (3q).

---

## Anthropic

No direct disclosure; triangulated from third-party estimates, spending ratios vs OpenAI, and data center evidence.

| Parameter / evidence | Value |
|---|---|
| OpenAI internal memo estimate | 1.4 GW end-2025 (~75% of OpenAI's 1.9 GW); 7–8 GW projected by end-2027 |
| Dylan Patel estimate (Mar 2026) | Anthropic and OpenAI both ~2–2.5 GW (likely includes early-2026 growth and third-party API compute) |
| Cloud compute spend | $2.5B (2024) = 43% of OpenAI's; $6.8B (2025) = 42% of OpenAI's |
| Naive spend-ratio estimate | 0.42 × 1.7M ≈ **730k H100e** |
| Naive power-ratio estimate | 0.75 × 1.7M ≈ **1.3M H100e** |
| Project Rainier (Indiana), end-2025 | ~470k H100e ≈ ~700k Trainium2 chips (vs target of 1M Trainium2, missed by months) |
| Amazon Mississippi campus | ~200k H100e, likely Trainium/Anthropic but less certain |
| Total Trainium estimate | 500–700k H100e across the two campuses |
| Chip mix | Trainium2-heavy; remainder Nvidia (Hopper + Blackwell) and Google TPU |
| Efficiency adjustments | Trainium2 ≈ Hopper-gen energy efficiency ⇒ Anthropic's fleet less power-efficient per GW than OpenAI's; non-Nvidia chips somewhat more cost-efficient per dollar (well under 2× advantage) |

**Bottom line:** ≥**1M H100e** end-2025 (conservative floor), likely **<1.5M**; ratio to OpenAI well over 50%, ~60% central. Caveats: 2025 spend ratio may understate a backloaded late-2025 ramp; all spend figures come from media interpretation of leaked documents (The Information).

---

## xAI (SpaceX)

Mostly owned compute; largely a bottom-up data center sum.

| Parameter | Value |
|---|---|
| Colossus 1 + Colossus 2 (Memphis), end-2025 | ~550k H100e combined (Colossus 1 ≈ 300k) |
| Georgia data center | ~10k GPUs |
| Portland data center | small, size unspecified |
| Oracle cloud (legacy) | 20k H100s used to train Grok 2 (2024); plausibly still rented but limited |
| **Total end-2025** | **~600–700k H100e** |
| Revenue context | ~$100M in Q3 2025 (~$400M annualized) ⇒ compute allocation skews heavily to R&D/training rather than inference |
| 2026 changes | Colossus 1 rented entirely to Anthropic (plus part of Colossus 2, up to ~$15B/yr); Colossus 2 expansion targeted at ~1.4M H100e by mid-2026; Cursor compute-sharing partnership |

---

## Google DeepMind (GDM)

Top-down allocation of Alphabet's owned compute.

| Parameter | Value |
|---|---|
| Google owned compute, end-2025 (sold basis) | ~5.1M H100e (~3.8M TPU, CI 3.1–4.5M; ~¼ of total is Nvidia) |
| Installed compute (1–2 quarter lag) | ~3.9M (1q) to ~2.8M (2q) H100e |
| Cloud share of ML compute | ~50% in 2025 (CFO statement); ~"just over half" guided for 2026 |
| Cloud bucket contents | external GPU/TPU rentals + Vertex API (incl. Gemini and third-party models) + Gemini Enterprise + Workspace subscriptions |
| Non-Cloud bucket contents | consumer Gemini inference (Google Services), DeepMind R&D/training, non-frontier internal ML (recommenders, Waymo, etc.) |
| Decision variable | DeepMind share > 50% iff Cloud-side enterprise Gemini inference > non-DeepMind internal use; author's guess: it is **not**, so DeepMind < 50% |
| Cloud-side enterprise Gemini inference | < 1M H100e (Google trails Anthropic and OpenAI in enterprise AI per Menlo Ventures; Anthropic's total enterprise inference itself < 1M H100e) |
| Non-DeepMind internal (recommenders) | ~1M H100e ballpark (parity argument with Meta via similar ad revenue) |
| Hard upper bound on DeepMind share | very unlikely > 70% of Google total (Nvidia quarter mostly serves external cloud; significant TPU also rented out, e.g. to Anthropic) |
| "Alphabet-level activities" expense (mostly shared AI R&D + corporate overhead) | $9.19B (2023), $10.5B (2024), $16.76B (2025), $5.3B (Q1 2026) — rough **upper bound** on DeepMind R&D compute spend; excludes product inference; billed at cost, not market GPU-hour rates |

**Bottom line:** DeepMind compute ≈ slightly under half of Google's installed total ⇒ roughly **1.5–2M H100e**. Not clear whether DeepMind exceeded OpenAI at end-2025.

---

## Meta Superintelligence Labs (MSL)

Top-down allocation of Meta's owned compute.

| Parameter | Value |
|---|---|
| Meta owned compute, end-2025 (sold basis) | ~2.3M H100e total (excludes low-volume MTIA) |
| — Nvidia | 1.84M H100e (90% CI: 1.43M–2.38M) |
| — AMD Instinct | 460k H100e (90% CI: 345k–612k), ~20% of total |
| Cloud rentals end-2025 | minimal (Google / Oracle / CoreWeave deals worth $10–20B signed late 2025–2026, not yet ramped) |
| Frontier vs recommender split | ~50:50 point estimate for 2025 (analysts: 50–60% of GPUs to recommenders mid-2025); bounded well inside 90:10 either direction; likely tilting toward MSL after the late-2025 frontier pivot |
| Historical anchors | end-2024 Meta owned ~700k H100e (~600k Nvidia); 100k-H100 frontier training cluster (late 2024) ≈ 15% of Meta's compute; Llama 3 405B: 3.8e25 FLOP on 16k H100s; Llama 4 Behemoth: 32k H100s |
| Capex signal | Q1-2025 guidance: majority of 2025 capex to "core business" (incl. recommender AI), not generative AI |

**Bottom line:** MSL ≈ ~50% of ~2.3M owned (less when adjusted for deployment lag) ⇒ roughly **~1M H100e**, probably **less than OpenAI** at end-2025.

---

## Cross-cutting adjustment: third-party (hyperscaler-resold) inference

Compute used by Microsoft/Amazon/Google to serve OpenAI/Anthropic models through their own APIs (Azure OpenAI, Bedrock, Vertex) is assumed **excluded** from the lab compute figures above. Optional add-on:

| Parameter | OpenAI | Anthropic |
|---|---|---|
| Revenue accounting | books only its 20% share of Microsoft-resold revenue (gross = 5× booked share) | books gross hyperscaler-resold revenue; hyperscaler cut booked as expense |
| Resold gross revenue vs first-party | likely < OpenAI's first-party revenue; ~60% if Microsoft-resold ≈ half of API revenue (API ≈ 25% of revenue mid-2025, per FT) | 25–50% of total revenue (The Information: first-party is majority; OpenAI CRO memo: ~$8B of $30B run rate is revshare gross-up) |
| Inference share of lab compute (2025) | ~50% | ~40% full-year; assume ~50% at end-2025 |
| Resulting boost to total compute | **~+25% plausible; +50% upper bound** | **~+25% plausible; +50% upper bound** (implies Google+Amazon hosted a few hundred thousand H100e for Claude APIs) |

---

## Headline aggregates (end-2025, 1-quarter deployment lag ⇒ ~16M H100e world)

| Entity | H100e (central) | World share |
|---|---|---|
| OpenAI | ~1.7M | ~10–11% |
| Anthropic | ~1–1.3M | ~6–8% |
| xAI | ~0.6–0.7M | ~4% |
| OpenAI + Anthropic + xAI | < 4M | ~20–30% (across lag assumptions) |
| + hyperscaler-resold inference | — | up to ~+5pp |
| GDM + MSL | ~½ of parents' compute each | ~15% combined |
| **All five labs** | — | **probably < 50% of world total** |

## Implementation notes

- The OpenAI power model is the anchor; Anthropic, GDM, and MSL estimates are ratios/allocations applied against it or against the ownership dataset, so propagate the OpenAI estimate's uncertainty downstream.
- Key sensitivity dimensions to expose as code parameters: deployment lag (0–3 quarters, both for labs and for the world stock), IT-vs-facility power interpretation (×1.0 vs ×~0.7), GPU-hour price scenario (low/high), Anthropic ratio source (spend 0.42 vs memo 0.75), GDM Cloud share (~0.5) and DeepMind-vs-internal split, Meta frontier share (0.4–0.6, bounded 0.1–0.9), third-party inference boost (0–0.5, central 0.25).
- Reference code: `epoch-research/ai-chip-counts` repo, branch `lab-compute` — `openai_power_model.ipynb` and `gpu_hour_model.ipynb`.
