SELECT
     SUM(l_extendedprice * l_discount) AS revenue
FROM
     lineitem
WHERE
     l_shipdate >= 8766
     AND l_shipdate < 9131
     AND l_discount >= .05 
     AND l_discount <= 0.070001
     AND l_quantity < 24;