-- Part 1.2
WITH yearly AS (
    SELECT
        substr(date_time,instr(date_time, '/') + instr(substr(date_time, instr(date_time, '/') + 1), '/') + 1, 4) AS year,
        SUM(traffic_volume) AS total_traffic
    FROM traffic
    WHERE CAST(substr(date_time,instr(date_time, '/') +instr(substr(date_time, instr(date_time, '/') + 1), '/') + 1,4) AS INTEGER) BETWEEN 2012 AND 2017
    GROUP BY year
)
SELECT
    year,
    total_traffic,
    total_traffic - LAG(total_traffic) OVER (ORDER BY year) AS change,
    ROUND(100.0 * (total_traffic - LAG(total_traffic) OVER (ORDER BY year)) / LAG(total_traffic) OVER (ORDER BY year),2) AS percentage_change
FROM yearly
ORDER BY year;

