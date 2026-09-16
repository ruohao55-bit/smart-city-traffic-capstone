--Part 2.1
WITH stats AS (
    SELECT
        COUNT(traffic_volume) AS n,
        AVG(traffic_volume) AS mean,
        AVG(traffic_volume * traffic_volume)
            - AVG(traffic_volume) * AVG(traffic_volume) AS variance,
        MAX(traffic_volume) AS maximum,
        MIN(traffic_volume) AS minimum
    FROM traffic
    WHERE traffic_volume IS NOT NULL
),
ordered AS (
    SELECT
        traffic_volume,
        ROW_NUMBER() OVER (ORDER BY traffic_volume) AS row_num,
        COUNT(*) OVER () AS total_rows
    FROM traffic
    WHERE traffic_volume IS NOT NULL
),
median_value AS (
    SELECT AVG(traffic_volume) AS median
    FROM ordered
    WHERE row_num IN (
        (total_rows + 1) / 2,
        (total_rows + 2) / 2
    )
)
SELECT
    n,
    ROUND(mean, 2) AS mean,
    ROUND(median, 2) AS median,
	ROUND(sqrt(variance), 2) AS standard_deviation,
    ROUND(variance, 2) AS variance,
    maximum - minimum AS range
FROM stats, median_value;