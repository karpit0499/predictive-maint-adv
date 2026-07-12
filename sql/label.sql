-- sql/label.sql
-- HORIZON is the ONE number that defines this project's prediction problem.
-- Change it here and nowhere else.
DECLARE horizon_minutes INT64 DEFAULT 10;

CREATE OR REPLACE TABLE `predictive-maint-adv.maintenance.features_labeled` AS
SELECT
  f.*,
  CAST(EXISTS(
    SELECT 1
    FROM `predictive-maint-adv.maintenance.failure_events` e
    WHERE e.machine_id = f.machine_id
      AND e.failed_at >  f.window_end                                            -- strictly in the FUTURE
      AND e.failed_at <= TIMESTAMP_ADD(f.window_end, INTERVAL horizon_minutes MINUTE)
  ) AS INT64) AS will_fail
FROM `predictive-maint-adv.maintenance.features_windowed` f
-- LABEL CENSORING: a window whose horizon has not fully elapsed yet would be
-- labeled 0 purely because the future has not happened. Those rows are not
-- "negatives", they are UNKNOWN — training on them teaches the model that the
-- most recent, most-degraded windows are safe. Drop them.
WHERE f.window_end <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL horizon_minutes MINUTE);