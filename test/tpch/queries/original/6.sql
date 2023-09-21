-- $ID$
-- TPC-H/TPC-R Forecasting Revenue Change Query (Q6)
-- Functional Query Definition
-- Approved February 1998

SELECT
     SUM(l_extendedprice * l_discount) AS revenue
FROM
     lineitem
WHERE
     l_shipdate >= date '1994-01-01'
     AND l_shipdate < (date '1994-01-01' + interval '1 year')
     AND l_discount BETWEEN .06 - 0.01 AND .06 + 0.010001
     AND l_quantity < 24;
