WITH latest AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY machine_id ORDER BY scored_at DESC) AS rn
  FROM {{ ref('stg_predictions') }}
)
SELECT machine_id, scored_at, failure_prob, risk_bucket, model_version
FROM latest
WHERE rn = 1