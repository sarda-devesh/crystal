-- $ID$
-- TPC-H/TPC-R Customer Distribution Query (Q13)
-- Functional Query Definition
-- Approved February 1998

SELECT
          c_count,
          COUNT(*) AS custdist
  FROM (
          SELECT
                  c_custkey,
                  COUNT(o_orderkey) AS c_count
          FROM
                  (SELECT * FROM customer
                  LEFT OUTER JOIN orders ON
                    c_custkey = o_custkey AND
                    o_comment NOT LIKE '%special%requests%') c_customer
          GROUP BY
                  c_custkey
          ) c_orders
  GROUP BY
          c_count
  ORDER BY
          custdist DESC,
          c_count DESC;
