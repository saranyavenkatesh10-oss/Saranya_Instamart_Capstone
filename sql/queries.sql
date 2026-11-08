/* Validation script for Q1 , Q3, Q4*/

WITH Test_Q1 AS (
    SELECT customer_id FROM orders
    WHERE status = 'completed' AND order_amount > 500
    GROUP BY customer_id HAVING SUM(order_amount) > 8000
),
Test_Q3 AS (
    SELECT product_id FROM products
    WHERE price_inr > (SELECT AVG(price_inr) FROM products)
),
Test_Q4 AS (
    SELECT c.customer_id FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
    WHERE o.order_id IS NULL
)
SELECT 
    'Q1 (WHERE/HAVING)' AS test_name, 
    COUNT(*) AS actual_value, 30 AS expected_value,
    CASE WHEN COUNT(*) = 30 THEN 'PASS' ELSE 'FAIL' END AS status FROM Test_Q1
UNION ALL
SELECT 
    'Q3 (Scalar Subquery)', 
    COUNT(*), 11,
    CASE WHEN COUNT(*) = 11 THEN 'PASS' ELSE 'FAIL' END FROM Test_Q3
UNION ALL
SELECT 
    'Q4 (LEFT JOIN)', 
    COUNT(*), 25,
    CASE WHEN COUNT(*) = 25 THEN 'PASS' ELSE 'FAIL' END FROM Test_Q4;


-- Q3

WITH RevenueMatrix AS (
    SELECT 
        c.city, 
        p.category, 
        SUM(o.order_amount) AS revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.status = 'completed'
    GROUP BY c.city, p.category
),
Metrics AS (
    SELECT 
        COUNT(*) AS total_rows,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(MAX(revenue), 2) AS max_cell_revenue,
        MAX(CASE WHEN city = 'Bengaluru' AND category = 'Electronics' THEN ROUND(revenue, 2) END) AS blr_elec_revenue
    FROM RevenueMatrix
)
SELECT 
    metric_name, actual_value, expected_value,
    CASE WHEN actual_value = expected_value THEN 'PASS' ELSE 'FAIL' END AS status
FROM (
    SELECT 'Q2 Row Count' AS metric_name, CAST(total_rows AS DECIMAL(18,2)) AS actual_value, 30.00 AS expected_value FROM Metrics
    UNION ALL
    SELECT 'Q2 Total Revenue', total_revenue, 2998668.80 FROM Metrics
    UNION ALL
    SELECT 'Q2 Max Cell (Bengaluru Electronics)', blr_elec_revenue, 233909.45 FROM Metrics
) Summary;



-- Q5

WITH CustomerSpend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
    WHERE status = 'completed' GROUP BY customer_id
),
AverageSpend AS (
    SELECT ROUND(AVG(total_spend), 2) AS global_avg_spend FROM CustomerSpend
),
FinalFilter AS (
    SELECT cs.customer_id FROM CustomerSpend cs
    CROSS JOIN AverageSpend asg WHERE cs.total_spend > asg.global_avg_spend
)
SELECT 
    'Q5 Intermediate Average' AS test_name, (SELECT global_avg_spend FROM AverageSpend) AS actual_val, 2552.06 AS expected_val,
    CASE WHEN (SELECT global_avg_spend FROM AverageSpend) = 2552.06 THEN 'PASS' ELSE 'FAIL' END AS status
UNION ALL
SELECT 
    'Q5 High Spender Count', COUNT(*), 550,
    CASE WHEN COUNT(*) = 550 THEN 'PASS' ELSE 'FAIL' END FROM FinalFilter;

-- Q6
WITH CustomerCitySpend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed' GROUP BY c.customer_id, c.city
),
Ranks AS (
    SELECT city, customer_id, ROUND(total_spend, 2) AS total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
    FROM CustomerCitySpend
)
SELECT 'Q6 Ranking Discrepancy (Ties)' AS test, COUNT(*) AS actual, 8 AS expected, CASE WHEN COUNT(*) = 8 THEN 'PASS' ELSE 'FAIL' END AS status 
FROM Ranks WHERE rnk != row_num
UNION ALL
SELECT 'Q6 Top Spender: Bengaluru', total_spend, 12856.20, CASE WHEN total_spend = 12856.20 THEN 'PASS' ELSE 'FAIL' END FROM Ranks WHERE city = 'Bengaluru' AND rnk = 1
UNION ALL
SELECT 'Q6 Top Spender: Mumbai', total_spend, 14766.55, CASE WHEN total_spend = 14766.55 THEN 'PASS' ELSE 'FAIL' END FROM Ranks WHERE city = 'Mumbai' AND rnk = 1;

-- Q7

WITH DailyRevenue AS (
    SELECT SUBSTRING(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue FROM orders
    WHERE status = 'completed' GROUP BY SUBSTRING(order_ts, 1, 10)
),
RunningTotals AS (
    SELECT day, SUM(revenue) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
    FROM DailyRevenue
)
SELECT 
    metric_name, actual, expected,
    CASE WHEN actual = expected THEN 'PASS' ELSE 'FAIL' END AS status
FROM (
    SELECT 'Q7 Day Row Count' AS metric_name, (SELECT COUNT(*) FROM RunningTotals) AS actual, 183 AS expected
    UNION ALL
    SELECT 'Q7 Final Cumulative Revenue', ROUND(MAX(cumulative_revenue), 2), 2998668.80 FROM RunningTotals
) Summary;
