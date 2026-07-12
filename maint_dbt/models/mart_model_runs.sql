SELECT
  DATE(checked_at) AS day,
  metric,
  MAX(z_shift) AS max_z_shift,
  LOGICAL_OR(retrained) AS retrained_that_day
FROM {{ source('maintenance', 'drift_log') }}
GROUP BY day, metric