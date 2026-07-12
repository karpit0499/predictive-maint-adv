-- z_shift is an EFFECT SIZE (a standardized mean shift — Cohen's d):
--     |recent_mean - baseline_mean| / baseline_stddev
-- Read it as "how many baseline standard deviations has the mean moved?"
-- Convention: ~0.2 small, ~0.5 medium, ~0.8 large. We fire at 1.0 — a full
-- baseline standard deviation, which is unambiguous and never trips on noise.
WITH bounds AS (
  SELECT MIN(window_end) AS t0, MAX(window_end) AS t1
  FROM `predictive-maint-adv.maintenance.features_windowed`
),
baseline AS (
  -- The EARLIEST 2 hours of data, as a proxy for the training distribution.
  -- (Do NOT use "older than 1 day": for the first 24 hours of the project's
  -- life that window is EMPTY, so the mean is NULL, so z is NULL, so the drift
  -- check silently does nothing — forever, and without ever erroring.)
  SELECT AVG(vibration_mean) AS mu, STDDEV(vibration_mean) AS sd, COUNT(*) AS n
  FROM `predictive-maint-adv.maintenance.features_windowed`, bounds
  WHERE window_end < TIMESTAMP_ADD(bounds.t0, INTERVAL 2 HOUR)
),
recent AS (
  SELECT AVG(vibration_mean) AS mu, COUNT(*) AS n
  FROM `predictive-maint-adv.maintenance.features_windowed`, bounds
  WHERE window_end >= TIMESTAMP_SUB(bounds.t1, INTERVAL 30 MINUTE)
)
SELECT
  SAFE_DIVIDE(ABS(recent.mu - baseline.mu), NULLIF(baseline.sd, 0)) AS z_shift,
  baseline.mu AS baseline_mean,
  baseline.sd AS baseline_sd,
  baseline.n  AS baseline_windows,
  recent.mu   AS recent_mean,
  recent.n    AS recent_windows
FROM baseline, recent;