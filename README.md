# ai-chip-counts
Code for estimating quantities of AI chips

For more information, see:
https://epoch.ai/data/ai-chip-sales
https://epoch.ai/data/ai-chip-owners

nvidia_estimates, tpu_estimates, amd_estimates, etc generate estimates for chip sales by designer

nvidia_owners allocates nvidia chips the hyperscaler and official Chinese owners using a revenue-based model. Other nvidia owners are modeled separately in nvidia_owners_other, with the exception of smuggled Chinese chips which are handled separately in v_diversion_and_resale
