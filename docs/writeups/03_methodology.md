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