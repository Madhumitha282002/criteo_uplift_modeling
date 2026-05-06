"""Central random seed for reproducibility.

Import this everywhere randomness is involved:
    from configs.seed import SEED

Used by: stratified sampling, train/test splits, model training,
bootstrap CIs, etc.
"""

SEED = 42