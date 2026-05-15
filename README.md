# Criteo Uplift Modeling Report

## Background

Criteo, a global digital advertising platform, faces the challenge every ad-targeting business shares: most users who see an ad would have converted (or not) regardless of whether they saw it. Spending budget on these users delivers no incremental value. The real targeting problem is identifying the subset of users whose behavior is actually *changed* by the ad, the responders, and concentrating spend on them. This analysis uses the Criteo Uplift v2.1 dataset (13.9M users, ~85% randomly treated) to estimate per-individual treatment effects and quantify the business value of uplift-based targeting versus untargeted spend.

## Executive Summary

Our analysis demonstrates that Criteo's advertising treatment produces a small but highly significant average lift (0.115 percentage points in conversion rate, a 59% relative lift over control), and that this average effect is strongly concentrated in an identifiable subset of users. The top 10% of users ranked by predicted uplift respond at 7.85x the sample-average rate, while the bottom 10% show zero measurable response. Targeting only the top 20% of users captures 84.8% of the total incremental conversions that treat-everyone would deliver, at one-fifth of the cost. The following sections detail the key findings and strategic recommendations for Criteo's targeting strategy.

## Insights

![Decile Lift Analysis](docs/figures/decile_lift_chart.png)

1. **Criteo's Advertising Effect Is Highly Concentrated in a Small User Segment:**

   - The top 10% of users, roughly 1.4M of the 14M-user base, capture nearly 80% of the entire treatment effect.
   - Within this segment, users respond at 7.85x the sample-average treatment lift (observed lift 0.00904 vs sample mean 0.00115).
   - This concentration means a small targeting budget, applied to the right users, recovers most of the value of universal advertising.

2. **A Bottom Tier of Users Is Completely Unresponsive to Treatment:**

   - The bottom 10% of users shows an observed lift of 0.000041, statistically indistinguishable from zero.
   - Treatment produces no measurable effect on this segment regardless of advertising creative or frequency.
   - This is a definitive finding: ad spend on these users is wasted in expectation, not just inefficient.

3. **The Average Treatment Effect Is Small in Absolute Terms but Large in Relative Terms:**

   - Treatment lifts the conversion rate by 0.115 percentage points (from 0.194% control to 0.309% treated).
   - In relative terms this is a 59.4% lift over the control conversion rate, a substantial campaign-level effect.
   - The treatment also moves users into the funnel earlier, with a 27.1% relative lift in visit rate (3.82% to 4.85%).

4. **Three Behavioral Features Drive Almost All of the Targeting Signal:**

   - Features `f4`, `f2`, and `f9` show the largest variation in treatment lift across feature quartiles and account for the majority of the model's discriminative power.
   - Feature `f3` reveals a counter-responsive segment: a specific quartile of users converts *less* when treated, possibly due to ad fatigue or audience mismatch.
   - The remaining nine features explain who is likely to convert in general but not who specifically responds to advertising, a critical distinction for ad targeting.

5. **The Underlying Experiment Is Statistically Pristine:**

   - The randomized 85/15 treatment split is exact to four decimal places, with no sample ratio mismatch detected (chi-square p = 0.9989).
   - The conversion treatment effect is significant at z = 33.5 (p < 1e-200), with a tight 95% CI of [0.00108, 0.00122].
   - The high data quality means the findings are reliable for production decision-making rather than artifacts of randomization or sampling bias.

## Recommendations

![Qini Curve Comparison](docs/figures/qini_curves_comparison.png)

1. **Adopt a Top-20% Targeting Budget as the Default Spend Strategy:**

   - At 20% of the cost of treat-everyone, Criteo captures 84.8% of total available uplift, the diminishing-returns inflection point in the Qini curve.
   - Expanding to top 30% adds only 8% more incremental conversions for 50% more cost; contracting below top 10% loses meaningful coverage.
   - Communicate this to media planners as "spend 20% to capture 85%", the cleanest framing of the targeting advantage for non-technical stakeholders.

2. **Eliminate Spend on the Bottom-Decile Audience Entirely:**

   - The bottom decile produces zero observed lift; treating these 1.4M users yields no measurable incremental conversions.
   - Removing them from targeted campaigns recovers roughly 10% of the advertising budget with no loss in conversion performance, a pure efficiency gain.
   - This is the single highest-confidence recommendation in the analysis given the bottom-decile lift is statistically indistinguishable from zero.

3. **Build Targeting Around the Three Heterogeneous Features and a Counter-Responsive Segment:**

   - Features `f4`, `f2`, and `f9` are the primary predictors of treatment response and should anchor any production targeting model.
   - The negative-lift quartile of `f3` represents users to actively *exclude* from targeted campaigns, not just deprioritize.
   - Audience segmentation built on these features will outperform demographic or contextual segmentation alone.

4. **Use the Top-Decile Segment for Premium-Spend Campaigns:**

   - The 7.85x response concentration in the top decile justifies higher cost-per-impression bids for these users.
   - This segment is large enough (1.4M users) to support major campaign rollouts but small enough for elevated targeting investment.
   - Test creative variations and bidding strategies on this segment first; insights from this group will generalize better than insights from average users.

5. **Operationalize the Uplift Pipeline for Ongoing Campaign Iteration:**

   - The four-stage pipeline (EDA, experimentation validation, modeling, evaluation) runs end-to-end in approximately 20 minutes on a 2M-row subsample, enabling weekly or per-campaign iteration.
   - Treat the experimentation validation (SRM and ATE checks) as a required quality gate before any targeting model is deployed; it catches randomization issues that would otherwise silently bias targeting decisions.
   - Re-run the heterogeneity analysis quarterly to detect when the responsive segment shifts, ensuring the targeting strategy stays aligned with user behavior over time.

By implementing these recommendations, Criteo can concentrate ad spend on its most responsive 20% of users with 4.24x the efficiency of untargeted spend, eliminate budget waste on the unresponsive bottom decile, and build a reusable targeting infrastructure that supports rapid campaign iteration. The full technical methodology, per-decile feature profiles, and modeling details are documented in `docs/writeups/`.

## Technologies

This analysis was built end-to-end using a combination of SQL for full-dataset experimentation work and Python for modeling and evaluation. The tooling was selected to handle 14M rows on a laptop while keeping the pipeline reproducible and easy to extend.

1. **Data Engineering and SQL:**

   - **DuckDB**: in-process OLAP database used as the primary data layer. Handles the full 14M-row dataset on a single laptop with sub-second query latency, replacing the need for a Spark or Postgres setup.
   - **pandas** and **pyarrow**: DataFrame operations and columnar serialization for moving data between DuckDB and Python.
   - **SQL**: used for schema validation, feature distributions, ATE estimation with confidence intervals, per-feature heterogeneity analysis, and decile aggregation. SQL files are version-controlled in `sql/`.

2. **Statistical Analysis:**

   - **scipy**: chi-square test for sample ratio mismatch and normal-distribution p-values for ATE significance.
   - **NumPy**: array operations for the from-scratch Qini curve and coefficient implementation.
   - **scikit-learn**: `StratifiedShuffleSplit` for train/test splitting and `StandardScaler` for feature standardization.

3. **Modeling:**

   - **scikit-learn (LogisticRegression)**: baseline T-learner base model. Fast to train and produces well-calibrated probability estimates on this dataset.
   - **LightGBM**: gradient-boosted tree T-learner base model, tested as a non-linear alternative. Configured with `min_child_samples=200` and L1/L2 regularization to handle the rare-event class.
   - **Custom T-learner module** (`src/models/t_learner.py`): pluggable wrapper that accepts any sklearn-compatible classifier as the base, making model swaps a one-line change.

4. **Evaluation:**

   - **Custom Qini implementation** (`src/evaluation/qini.py`): Radcliffe-Surry Qini curve and coefficient written from scratch in approximately 30 lines, with unit tests verifying expected behavior on random and perfect-ranking inputs.
   - **Custom top-K and decile modules**: business-facing evaluation including three-strategy comparison (treat-everyone vs random vs model) and per-decile observed-lift profiles with 95% confidence intervals.
   - **matplotlib** and **seaborn**: all charts produced at 150 DPI with a consistent style (`seaborn-v0_8-darkgrid`, fontsize 12, titles 14pt).

5. **Engineering and Reproducibility:**

   - **pytest**: 10 unit tests covering the T-learner, Qini, and top-K modules. Includes tests for input validation, output shapes, and statistical properties (random predictor produces near-zero Qini).
   - **ruff** and **black**: linting and formatting applied uniformly across `src/` and `tests/`.
   - **Jupyter notebooks**: four numbered notebooks (`01_eda` through `04_evaluation`) that run top-to-bottom in a fresh kernel, with orchestration logic only; all computation lives in `src/` modules.
   - **Central seed configuration** (`configs/seed.py`): single `SEED = 42` imported everywhere randomness is involved, ensuring reproducibility across runs.

The full project structure, dependencies, and a runnable Quick Start are available in the repository. The pipeline is designed to run on a 16 GB laptop in approximately 20 minutes end-to-end on a 2M-row stratified subsample of the full dataset.