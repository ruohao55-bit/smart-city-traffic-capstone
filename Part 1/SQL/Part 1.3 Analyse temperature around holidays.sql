-- Part 1.3
WITH holiday_temp AS (
    SELECT
        holiday,
        substr(date_time,instr(date_time, '/') + instr(substr(date_time, instr(date_time, '/') + 1), '/') + 1, 4) AS year,
        AVG(temp) AS avg_temp
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND CAST(
          substr(date_time,instr(date_time, '/') + instr(substr(date_time, instr(date_time, '/') + 1), '/') + 1,4) AS INTEGER) BETWEEN 2015 AND 2017
    GROUP BY holiday, year
)
SELECT
    holiday,
    year,
    ROUND(avg_temp, 2) AS average_temperature,
    ROUND(avg_temp - LAG(avg_temp) OVER (PARTITION BY holiday ORDER BY year),2) AS temperature_change
FROM holiday_temp
ORDER BY holiday, year;