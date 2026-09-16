-- P(Clear Weather | Congestion) / P(High Temperature | Congestion), 
SELECT
    AVG(CASE WHEN weather_main = 'Clear' THEN 1.0 ELSE 0 END) AS p_clear_given_congestion,
    AVG(CASE WHEN temp > 292        THEN 1.0 ELSE 0 END) AS p_hightemp_given_congestion
FROM traffic
WHERE traffic_volume > 5500;


