-- $ID$
-- TPC-H/TPC-R Order Priority Checking Query (Q4)
-- Functional Query Definition
-- Approved February 1998

SELECT o_orderpriority, COUNT(*) AS order_count
  FROM orders
 WHERE o_orderdate >= date '1993-07-01'
   AND o_orderdate < date '1993-07-01' + interval '3 month'
   AND EXISTS (
      SELECT *
        FROM lineitem
       WHERE l_orderkey = o_orderkey
         AND l_commitdate < l_receiptdate
     )
GROUP BY o_orderpriority
ORDER BY o_orderpriority;
