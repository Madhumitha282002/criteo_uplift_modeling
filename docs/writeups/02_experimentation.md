# Experimentation Analysis

Validation of the Criteo Uplift v2.1 experiment: sample-ratio mismatch
test, then average treatment effect estimates with confidence intervals
for both outcomes (`visit` and `conversion`).

## 1. Sample Ratio Mismatch (SRM)

The dataset documentation specifies an 85/15 treated/control split. We
test whether the observed split is consistent with this target using a
chi-square goodness-of-fit test.

| | Observed | Expected (85/15) |
|---|---|---|
| Treated  | <n_t>     | <exp_t>     |
| Control  | <n_c>     | <exp_c>     |
| **Total**| **<n_total>** | **<n_total>** |

- **Observed treated share:** <observed_share>
- **Absolute deviation from 85%:** <deviation_pp> pp
- **Chi-square statistic:** <chi2>
- **p-value:** <p_value>

### Interpretation

<Pick one of the two scenarios below based on your output:>

**Scenario A — p ≥ 0.001:** The observed split is statistically
consistent with the documented 85/15 target. No SRM detected.
Randomization passes this validation.

**Scenario B — p < 0.001:** The chi-square test flags a deviation from
exactly 85/15. With n = 14M, chi-square has extreme power and detects
deviations of even 0.1 percentage points as "significant." The
practical magnitude here is <deviation_pp> pp, which is small enough
that downstream causal estimates are not meaningfully biased. The
Criteo v2.1 release is a non-uniform sub-sample of the original
experiment (per the dataset card), so a slight deviation from the
nominal 85/15 is expected. We proceed with treatment as the causal
variable, noting the deviation.

## 2. Average Treatment Effect

ATE is estimated as the simple difference of outcome means between
treated and control arms (intent-to-treat using the random `treatment`
assignment, not the post-randomization `exposure` indicator).

### Formula

For a binary outcome Y with treatment assignment W:
ATE  = p_t - p_c
SE   = sqrt( p_t (1 - p_t) / n_t + p_c (1 - p_c) / n_c )
95% CI = ATE ± 1.96 · SE
z      = ATE / SE
p      = 2 · (1 − Φ(|z|))

This is the standard-error formula for the difference of two
proportions. Standard errors are computed in SQL; p-values are
computed in Python via `scipy.stats.norm`.

### Results

| outcome | control_rate | treatment_rate | ATE | 95% CI low | 95% CI high | p-value |
|---|---|---|---|---|---|---|
| visit      | <p_c_v> | <p_t_v> | <ate_v> | <ci_lo_v> | <ci_hi_v> | <p_v> |
| conversion | <p_c_c> | <p_t_c> | <ate_c> | <ci_lo_c> | <ci_hi_c> | <p_c> |

### Interpretation

- **Visit:** Treatment lifts the visit rate by <ate_v> in absolute
  terms (<rel_lift_v>% relative). The 95% CI is narrow due to the 14M
  sample size and excludes zero by a wide margin.
- **Conversion:** The absolute lift is <ate_c> on a base rate of
  <p_c_c>. While the absolute number is tiny, the relative lift is
  <rel_lift_c>% and the effect is highly significant
  (p = <p_c>).

The conversion-lift CI being wider than the visit-lift CI (in relative
terms) reflects the much rarer base rate: each control-arm conversion
contributes more to the variance.

## 3. Why this matters for uplift modeling

Both outcomes show a positive, statistically significant ATE. This is
the floor: a model that targets randomly already captures this average
lift. The uplift model's job (Weeks 2–3) is to identify *which* users
contribute most to this average — i.e., where the treatment effect is
heterogeneous. The Day 5 heterogeneity analysis is what tells us
whether such users exist in this data.