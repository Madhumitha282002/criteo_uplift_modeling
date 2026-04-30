# Data Overview: Criteo Uplift v2.1

## Source

Criteo AI Lab, released alongside Diemert et al. (2018), *A Large Scale
Benchmark for Uplift Modeling* (AdKDD/KDD 2018). Mirror used:
[criteo/criteo-uplift on HuggingFace](https://huggingface.co/datasets/criteo/criteo-uplift).

## Shape

- **Rows:** 13,979,592
- **Columns:** 16 (12 features `f0`–`f11`, plus `treatment`, `conversion`, `visit`, `exposure`)
- **All features are dense floats.** Names anonymized; values randomly projected.

## Treatment / control split

| treatment | n | pct |
|---|---|---|
| 0 (control)   | 2096937 | 0.15 |
| 1 (treated)   | 11882655| 0.85 |

The split is approximately 85/15, matching the spec on the dataset card.

## Outcome rates by treatment group

| treatment | visit_rate | conversion_rate | exposure_rate | n |
|---|---|---|---|---|
| 0 (control) | 0.038201 | 0.001938 | 0.000000 | 2096937 |
| 1 (treated) | 0.048543 | 0.003089 | 0.036037 | 11882655 |

Raw differences (no CIs yet — see Day 4 for formal ATE):

- Visit lift:      `0.010342`
- Conversion lift: `0.001152`

## Notes

- **Conversion is extremely rare** (~0.3% overall). This drives several
  downstream choices: stratified sampling on `(treatment, conversion)`
  to preserve positives, `class_weight="balanced"` for logistic
  regression baselines, and `is_unbalance=True` for LightGBM.
- **Treatment vs. exposure.** `treatment` is the random assignment
  (intent to treat). `exposure` indicates whether the ad actually
  reached the user. We use `treatment` as the causal variable
  throughout — the assignment is what was randomized, so it preserves
  unbiasedness.
- **Treatment imbalance (85/15)** means standard errors on the control
  side are the binding constraint for ATE precision. The dataset is
  still well-powered because of its size, but the imbalance shows up
  again at modeling time (the control-arm T-learner sees ~6x less data
  than the treated-arm one).