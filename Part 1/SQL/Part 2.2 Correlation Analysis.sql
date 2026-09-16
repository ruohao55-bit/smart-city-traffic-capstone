SELECT
    (COUNT(*) * SUM(temp * traffic_volume) - SUM(temp) * SUM(traffic_volume))
    /
    (
        SQRT(COUNT(*) * SUM(temp * temp)            - SUM(temp) * SUM(temp))
      * SQRT(COUNT(*) * SUM(traffic_volume * traffic_volume) - SUM(traffic_volume) * SUM(traffic_volume))
    ) AS pearson_r
FROM traffic
WHERE temp IS NOT NULL
  AND traffic_volume IS NOT NULL;