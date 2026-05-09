# Methodology

## 1. The T-learner approach

### What we're estimating

In standard binary classification we estimate `P(Y=1 | X)`. In uplift
modeling we want something different: the **conditional average
treatment effect** (CATE):
τ(x) = E[Y | X=x, W=1] − E[Y | X=x, W=0]

This is the per-individual lift in outcome probability caused by
treatment, conditional on features. It tells us *who responds to
treatment*, not just *who converts*. The whole reason uplift modeling
exists is that targeting users by `P(Y=1)` and targeting users by
`τ(x)` give different answers, and the latter is what an advertiser
actually wants to optimize.

### How a T-learner estimates τ(x)

The T-learner ("Two-model" learner) is the most direct approach:
fit two separate outcome models, one per treatment arm, and
difference their predictions.

μ̂_t(x) = model trained on (X, Y) where W=1, predicts E[Y | X, W=1]
μ̂_c(x) = model trained on (X, Y) where W=0, predicts E[Y | X, W=0]
τ̂(x)  = μ̂_t(x) − μ̂_c(x)

Each base model is a standard binary classifier — logistic regression
or gradient boosting in our case. The T-learner is just the wrapper
that splits the data, fits both, and differences predictions.

### Why this works

Under the standard causal-inference assumptions (unconfoundedness,
overlap, SUTVA), randomized treatment makes the two conditional
expectations identifiable from observational data:

- `E[Y | X, W=1]` is estimated unbiasedly from the treated arm.
- `E[Y | X, W=0]` is estimated unbiasedly from the control arm.

Their difference is then an unbiased estimate of τ(x). In Criteo,
the assumptions hold by construction: treatment was randomly
assigned (verified Day 4 SRM), so unconfoundedness is automatic.

### Strengths and weaknesses

**Strengths:**
- Conceptually simple — two binary classifiers, one subtraction.
- Each base model is solving a problem it's well-suited to (predicting
  conditional outcome rates), with mature tooling.
- Trivial to swap base models (LR → LightGBM in Day 14) without
  touching the wrapper.

**Weaknesses:**
- Each model is trained on a subset of the data. With 85/15 treatment
  split, the control model sees ~6× less data than the treated model.
  This asymmetry inflates variance on the control side, which
  propagates to τ̂.
- Each model optimizes for outcome prediction, not for the difference
  of outcomes. Small errors in each model can amplify when differenced,
  especially when τ(x) is small relative to μ_t(x) and μ_c(x).
  This is exactly Criteo's regime: ATE ≈ 0.001 vs base rate ≈ 0.002.

### Alternatives we are not pursuing (and why)

- **S-learner (Single model):** include treatment W as a feature in
  one combined model, then predict twice with W=1 and W=0 to get the
  difference. Simpler than T-learner but tends to under-detect
  heterogeneity when the treatment effect is small relative to the
  outcome's marginal predictability — exactly Criteo's setup. We skip.

- **X-learner:** uses the T-learner predictions to construct
  pseudo-outcomes, then fits a second-stage model on those. Lower
  variance than T-learner when treatment groups are very unbalanced
  (which Criteo's 85/15 is). A natural follow-up if the T-learner
  underperforms; out of scope for this 4-week build.

- **R-learner / DR-learner:** orthogonalize via a propensity model
  and an outcome model, then fit τ directly. Most efficient under
  good nuisance models but adds complexity for a marginal gain on
  randomized data.

- **Causal forest:** non-parametric direct estimation of τ via
  honest random forests (e.g. EconML, GRF). Strong performer on
  heterogeneous data but heavier to interpret and slower to fit.
  Listed in Day 19 as a follow-up extension.

We pick T-learner because it's the cleanest baseline that exposes the
core uplift-modeling concepts — two conditional outcome models and a
ranking-based evaluation. Days 10 and 14 implement it with logistic
regression and LightGBM bases respectively.

## 4. Qini curve and Qini coefficient

### What we're trying to evaluate

The T-learner produces τ̂(x), a per-individual estimate of treatment
effect. To use it, we rank users by τ̂(x) descending and treat only
the top fraction. The question Qini answers is: **how many incremental
conversions does this ranking produce, compared to ranking randomly?**

A model that ranks well concentrates the actual responders at the top
of the list. A bad model spreads them uniformly — which is what random
targeting does, by definition.

### The incremental-conversions formula

For a test set with n_t treated and n_c control rows, after sorting all
rows by τ̂ descending and taking the top K fraction:

incremental(K)  =  Σ_{top K, treated} y − Σ_{top K, control} y × (n_treated_in_top_K / n_control_in_top_K)

The reweighting term is the only non-obvious piece. It corrects for
the fact that the top-K slice typically has a different treated/control
split than the overall test set — e.g., if the top 25% has 3 treated
to 2 control rows, the treated arm has 1.5× more chances to observe a
conversion. To make the two sums comparable on the same "what would
this many people have converted under treatment vs control" basis, we
scale up the control sum by that ratio before subtracting.

When evaluated at K = 100%, the reweighting collapses to (n_t / n_c)
and the result equals the unconditional ATE × n_test, which is the
total incremental effect of treating everybody.

### The Qini curve

Plot incremental(K) on the y-axis against K (fraction targeted) on
the x-axis. Two reference lines:

- **Random baseline:** the straight line from (0, 0) to (1, total_incremental).
  This is what random targeting achieves in expectation.
- **Perfect ranker:** sort by *true* treatment effect (which we don't
  have in practice). Sets the upper envelope.

The model's curve sits between these two lines if the model has signal.

### The Qini coefficient

The signed area between the Qini curve and the random baseline:

Q  =  ∫₀¹ [ incremental_curve(K) − random_baseline(K) ] dK

Positive Q = beating random. Larger = more concentrated signal. A
random predictor gets Q ≈ 0; a perfect predictor gets the maximum
possible Q for the dataset (which is dataset-dependent and typically
much smaller than the 0.5 figures seen in textbook examples).

### Worked example (20 rows)

To verify intuition, we trace the formula on a 20-row toy example
(10 treated, 10 control). Total conversions: 4 treated, 2 control.

| K     | top rows | Σy_t | n_t | Σy_c | n_c | reweight | incremental |
|-------|----------|------|-----|------|-----|----------|-------------|
| 25%   | 5        | 3    | 3   | 1    | 2   | 1.500    | **1.500**   |
| 50%   | 10       | 3    | 6   | 2    | 4   | 1.500    | **0.000**   |
| 75%   | 15       | 3    | 8   | 2    | 7   | 1.143    | **0.714**   |
| 100%  | 20       | 4    | 10  | 2    | 10  | 1.000    | **2.000**   |

The random baseline at K=25% is 2.0 × 0.25 = 0.5; the model produces
1.5. So the model beats random by 1.0 incremental conversions at the
top quartile, despite being worse than random at K=50%. This is the
typical shape: gains concentrate at the top of the ranking and noise
dominates further down.

### What good looks like for Criteo

Published Criteo Qini values cluster in the **0.005 to 0.015 range**
for sensible models. This is small in absolute terms because conversion
itself is rare (~0.3% control rate), but is reliably positive: any Q
> 0 means the model beats random targeting, and Q > 0.005 is a real
ranking improvement on this data. We do not expect the Q ≈ 0.5 values
seen in synthetic-benchmark papers — those use easier datasets with
much stronger heterogeneity than Criteo's anonymized features admit.

### Implementation note

We implement the Qini curve and coefficient from scratch on Day 12
rather than using a library (e.g. `causalml`, `econml`). The reasons:
the math is simple enough that "from scratch" is a 20-line function,
and rolling our own forces us to internalize the reweighting term —
which is the part that distinguishes Qini from a vanilla cumulative-
gains chart and the part most practitioners get wrong when explaining
it.