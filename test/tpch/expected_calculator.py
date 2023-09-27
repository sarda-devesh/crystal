import pandas as pd
import os
import json

tables_dir = "/gscratch/ubicomp/devess/crystal/test/tpch/data/s1_converted_all"
def read_table(table_name, schema):
    col_names = schema[table_name]["columns"]
    table_file = os.path.join(tables_dir, table_name + ".tbl")
    return pd.read_csv(table_file, header = None, names = col_names, sep = '|').astype(float)

schema_dir = "/gscratch/ubicomp/devess/crystal/test/tpch/data/schema_metadata"
def read_schema():
    all_schema_file = os.path.join(schema_dir, "schema.json")
    with open(all_schema_file, 'r') as reader:
        schema = json.load(reader)
    return schema

def q6():
    schema = read_schema()
    lineitem_df = read_table("lineitem", schema)
    print(lineitem_df.columns)
    print(len(lineitem_df.index))
    lineitem_df = lineitem_df[lineitem_df["l_shipdate"] >= 8766]
    print(len(lineitem_df.index))
    lineitem_df = lineitem_df[lineitem_df["l_shipdate"] < 9131]
    print(len(lineitem_df.index))
    lineitem_df = lineitem_df[lineitem_df["l_discount"] >= 0.05]
    print(len(lineitem_df.index))
    lineitem_df = lineitem_df[lineitem_df["l_discount"] <= 0.070001]
    print(len(lineitem_df.index))
    lineitem_df = lineitem_df[lineitem_df["l_quantity"] < 24]
    print(len(lineitem_df.index))
    lineitem_df["product"] = lineitem_df["l_extendedprice"] * lineitem_df["l_discount"]
    print(lineitem_df["product"].sum())

if __name__ == "__main__":
    q6()