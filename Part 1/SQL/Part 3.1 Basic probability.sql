SELECT
    AVG(CASE WHEN traffic_volume > 5500
             THEN 1.0 ELSE 0 END)                    AS 'P(congestion)',

    AVG(CASE WHEN weather_main = 'Clear'
             THEN 1.0 ELSE 0 END)                    AS 'P(clear_weather)',

    AVG(CASE WHEN traffic_volume > 5500
              AND weather_main = 'Clear'
             THEN 1.0 ELSE 0 END)                    AS 'P(congestion_and_clear)'
FROM traffic;