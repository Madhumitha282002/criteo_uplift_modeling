-- Criteo Uplift v2.1 — Schema definition
-- Source: https://huggingface.co/datasets/criteo/criteo-uplift
-- Paper:  Diemert et al. 2018, "A Large Scale Benchmark for Uplift Modeling" (AdKDD/KDD 2018)
-- Rows:   13,979,592
-- Treatment ratio: ~85% treated, ~15% control
--
-- Column definitions:
--   f0..f11      Anonymized dense float features. Names obfuscated and values
--                randomly projected to prevent recovery of original user context
--                while preserving predictive power.
--   treatment    1 = randomly assigned to be targeted by advertising,
--                0 = held out (control).
--   conversion   Binary outcome label. 1 if the user converted.
--   visit        Binary outcome label. 1 if the user visited the advertiser.
--   exposure     1 if the user was effectively exposed to the ad.
--                Treatment is the *intent* to treat (random assignment);
--                exposure is whether treatment actually landed. Use treatment
--                for ITT analysis (Day 4 onward).

CREATE OR REPLACE TABLE criteo AS
SELECT *
FROM read_csv_auto('data/criteo-uplift.csv');

-- Sanity check (should return 13,979,592)
-- SELECT COUNT(*) FROM criteo;