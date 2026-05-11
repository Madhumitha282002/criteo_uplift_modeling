# Results

## Section 1 — Top-K and strategy comparison (LR T-learner)

The Qini coefficient measures ranking quality over all budget levels.
For a business audience, the more useful question is: at a fixed
budget, how many incremental conversions does each strategy deliver?

### Top-K incremental conversions

| K | Users targeted | Incremental conversions | Per 1,000 targeted |
|---|---|---|---|
| 10% | 40,000  | 311.6 | 7.791 |
| 20% | 80,000  | 332.9 | 4.161 |
| 30% | 120,000 | 359.9 | 2.999 |
| 50% | 200,000 | 377.6 | 1.888 |

The per-1,000-targeted column tells the real story: the top 10% of the
ranking is **4.1× more efficient per user targeted** than the top 50%
(7.791 vs 1.888). Incremental conversions concentrate sharply at the
top of the model's ranking — exactly the property uplift modeling is
supposed to deliver.

### Three-strategy comparison at K = 20%

| Strategy | Cost (users) | Incremental conversions | Per 1,000 cost |
|---|---|---|---|
| Treat everyone (K=100%) | 400,000 | 392.7 | 0.9817 |
| Random targeting (K=20%) | 80,000 | 78.5 | 0.9817 |
| Model targeting (K=20%) | 80,000 | **332.9** | **4.1608** |

![Strategy comparison](../figures/strategy_comparison_lr.png)

### Headline metrics

- **Model vs random at top 20%:** **4.24×** — the model identifies
  responders 4.24 times more efficiently than random selection within
  a fixed 20% budget.
- **% of total uplift captured at 20% cost:** **84.8%** — by treating
  only the top 20% of users (one fifth of the cost), the model still
  captures 84.8% of the total incremental conversions that
  treating-everyone would deliver.

### Interpretation

"Treat everyone" maximizes raw incremental conversions (392.7) but at
5× the cost of a top-20% strategy. "Random K=20%" is what targeting
without a model achieves — it captures exactly 20% of total uplift
by construction (78.5 ≈ 392.7 × 0.20), which is why its
per-1,000-cost ROI matches treat-everyone's: random targeting just
samples from the same population.

The LR T-learner's value is the gap between 20% (random) and 84.8%
(model) at the same cost. That's a **4.24× efficiency multiplier**
for the same spend. In a real ad-targeting application:

- Same ad budget, ~4× more incremental conversions.
- Or equivalently: same conversion goal, ~80% lower ad spend.

Even more striking is the top 10%: 311.6 incremental conversions on a
10% budget vs 39.3 from random targeting — that's **7.9× lift** at a
half-cost budget, while still capturing 79.3% of treat-everyone's
total uplift. This is a clean concentration result: most of the
treatment effect lives in a small, identifiable subset of users.

These numbers come from a logistic-regression baseline. Day 14 swaps
in LightGBM to see how much additional lift non-linear interactions
unlock; Day 15 does the per-decile breakdown that makes this
concentration pattern visible at a glance.