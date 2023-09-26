-- $ID$
-- TPC-H/TPC-R Important Stock Identification Query (Q11)
-- Functional Query Definition
-- Approved February 1998

SELECT
     ps_partkey,
     SUM(ps_supplycost * ps_availqty) AS value
FROM
     partsupp,
     supplier,
     nation
WHERE
     ps_suppkey = s_suppkey
     AND s_nationkey = n_nationkey
     AND n_name = 'GERMANY'
GROUP BY
     ps_partkey 
HAVING SUM(ps_supplycost * ps_availqty) > (
			SELECT
				SUM(ps_supplycost * ps_availqty) * 0.0001000000e-2
			FROM
				partsupp,
				supplier,
				nation
			WHERE
				ps_suppkey = s_suppkey
				AND s_nationkey = n_nationkey
				AND n_name = 'GERMANY'
		)
ORDER BY
     value DESC
LIMIT 10;