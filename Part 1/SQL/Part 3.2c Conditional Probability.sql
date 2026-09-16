-- odds ratio of congestion in clear versus cloudy weather
SELECT
    a, b, c, d,
    (a * 1.0 * d) / (b * c) AS odds_ratio
FROM (
    SELECT
        SUM(CASE WHEN weather_main = 'Clear'  AND traffic_volume > 5500 THEN 1 ELSE 0 END) AS a, -- clear + congested
        SUM(CASE WHEN weather_main = 'Clear'  AND traffic_volume <= 5500 THEN 1 ELSE 0 END) AS b, -- clear + not
        SUM(CASE WHEN weather_main = 'Clouds' AND traffic_volume > 5500 THEN 1 ELSE 0 END) AS c, -- cloudy + congested
        SUM(CASE WHEN weather_main = 'Clouds' AND traffic_volume <= 5500 THEN 1 ELSE 0 END) AS d  -- cloudy + not
    FROM traffic
) counts;