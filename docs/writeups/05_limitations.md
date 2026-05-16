# Limitations and Next Steps

This document records the known limitations of the current pipeline and the analyses that would extend or strengthen the findings. It is the document a future maintainer or reviewer should read before extending the work.

## Limitations

1. **Anonymized features prevent business-level interpretation.**

   The 12 features `f0` through `f11` are random projections of Criteo's original user features. The Day 5 analysis identifies `f4`, `f2`, and `f9` as the strongest drivers of treatment heterogeneity, but the *meaning* of these features (browsing history, demographics, prior engagement, etc.) is unrecoverable. A production deployment would substitute the projection with real features and re-run the heterogeneity analysis to make the segmentation actionable.

2. **A single point estimate is reported for the Qini coefficient.**

   The Qini values (LR 161.4, GBM 85.1) are computed on one train/test split with no confidence intervals. The proper way to express uncertainty is to bootstrap the test set, recompute Qini on each resample, and report a 95% CI. With 400K test rows the CI would be tight, but reporting "Qini = 161 [155, 167]" is meaningfully more credible than "Qini = 161." Time-budgeted out of the 4-week scope.

3. **The experimentation analysis is intent-to-treat, not as-treated.**

   We use the random `treatment` assignment throughout as the causal variable. The dataset also includes an `exposure` indicator (whether the ad actually reached the user). ITT analysis is unbiased and is the right default, but the as-treated analysis would estimate a different quantity (the per-exposure effect, larger in absolute magnitude). Both are useful in different decision contexts; only one is reported here.

4. **No causal validation beyond the SRM and ATE checks.**

   Validation of the experiment relies on the chi-square SRM test (Day 4) and the ATE significance test. A more thorough validation would include covariate balance checks across treatment arms (per-feature mean differences with their CIs) and orthogonality tests showing that `treatment` is uncorrelated with each `f_i`. These would catch subtle randomization issues that SRM alone misses.

5. **The 2M-row stratified subsample is used for modeling, not the full 14M rows.**

   Subsampling was a deliberate choice to enable laptop-scale iteration (Day 7). It preserves all relative proportions (treatment split, conversion rate, visit rate) within tolerance, but does discard 86% of the signal. Training on the full data would tighten variance on the control-arm model especially (currently ~1.6M training rows on a 0.19% positive class). A production pipeline would train on the full dataset and refresh weekly.

6. **Only the T-learner architecture is evaluated.**

   The plan considered S-learner, X-learner, R-learner, and causal forest alternatives (see methodology Section 1), but only the T-learner was implemented. X-learner in particular is theoretically better-suited to Criteo's 85/15 treatment imbalance and is the most likely single-day extension to materially improve results.

7. **No hyperparameter tuning is performed.**

   Both the LR and LightGBM T-learners use sensible-default hyperparameters chosen from domain knowledge. A proper grid search or Bayesian optimization on the validation set would likely improve Qini by some margin. The current hyperparameters were tuned by hand based on diagnostic feedback (Day 10 LR retraining without `class_weight`, Day 14 LightGBM retraining with `min_child_samples=200`).

8. **The bottom-decile "zero effect" finding is sample-specific.**

   The bottom-decile lift of 0.000041 is statistically indistinguishable from zero on the 400K test set, but the 95% CI on that estimate is roughly [-0.0003, +0.0004]. A future deployment must re-verify zero response on each new dataset before excluding the bottom decile from spend; the *finding* generalizes but the *specific decile boundary* will shift.

## Possible Extensions

1. **Implement an X-learner.** The X-learner constructs pseudo-outcomes from the T-learner's predictions and fits a second-stage model on those. Under Criteo's 85/15 treatment imbalance it typically reduces variance on the control-arm estimate. Estimated effort: one day. Expected impact: 10-30% Qini improvement.

2. **Bootstrap CIs for Qini.** Resample the test set 100-500 times with replacement, compute Qini on each, and report the 2.5th and 97.5th percentiles. Costs ~30 minutes of compute, gives every reported number a defensible uncertainty range.

3. **Train on the full 14M rows.** Drop the 2M-row stratified subsample step from the pipeline and train both T-learners on the full data. Memory-feasible with float32 features (~14M x 12 x 4 bytes = 672 MB). Training time goes from 5 minutes to 30-45 minutes for LightGBM. Likely improves the GBM more than the LR.

4. **Add a causal forest baseline.** The EconML or grf-python libraries provide reference implementations. Causal forests are heavier to interpret but are the current gold-standard for non-parametric uplift estimation. Useful as a calibration baseline: if a causal forest only ties the LR, the linear-effects-dominate hypothesis from the results writeup is strengthened.

5. **Build a propensity-weighted check on the ATE.** Even with random assignment, fitting a propensity model `P(W=1|X)` and checking that it is approximately constant across users (no feature predicts treatment) gives an independent check on the randomization. Five minutes of work, catches subtle randomization bugs in non-Criteo datasets.

6. **Replace the four-notebook pipeline with a single CLI entrypoint.** A `python -m criteo_uplift run` command that orchestrates the four stages, with progress bars and a final results JSON, would make the pipeline production-deployable. Notebooks would remain as the exploratory interface.

7. **Add monitoring for distribution drift.** In production the feature distributions and treatment-effect heterogeneity will shift over time. A drift dashboard comparing per-feature p10/p50/p99 across the current refresh vs the previous one, with alerts on >10% changes, is the minimum production-grade addition.

8. **Validate on a second uplift dataset.** Hillstrom, Lenta, or a synthetic dataset with known ground-truth treatment effects. Reusing the pipeline on a second dataset is the cleanest way to demonstrate the code is not Criteo-specific.

These extensions are roughly ordered from "highest impact for lowest effort" (X-learner, bootstrap CIs) to "production hardening" (CLI, monitoring) to "scope expansion" (second dataset). For interview discussion, the X-learner and bootstrap CIs are the two extensions most likely to come up as natural follow-up questions.