SELECT
  machine_id,
  scored_at,
  failure_prob,
  CASE WHEN failure_prob >= 0.8 THEN 'high'
       WHEN failure_prob >= 0.5 THEN 'medium'
       ELSE 'low' END AS risk_bucket,
  model_version
FROM {{ source('maintenance', 'predictions') }}