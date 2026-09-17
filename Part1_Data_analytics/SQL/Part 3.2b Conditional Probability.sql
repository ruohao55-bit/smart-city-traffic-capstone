-- P(A ∩ B) = P(A) × P(B)
SELECT
    AVG(CASE WHEN traffic_volume > 5500 THEN 1.0 ELSE 0 END)              AS p_cong,
    AVG(CASE WHEN weather_main = 'Clear' THEN 1.0 ELSE 0 END)            AS p_clear,
    AVG(CASE WHEN traffic_volume > 5500
              AND weather_main = 'Clear' THEN 1.0 ELSE 0 END)            AS p_joint,
    AVG(CASE WHEN traffic_volume > 5500 THEN 1.0 ELSE 0 END)
      * AVG(CASE WHEN weather_main = 'Clear' THEN 1.0 ELSE 0 END)        AS p_product
FROM traffic;