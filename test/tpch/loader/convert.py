import argparse
import pandas as pd
import os
import json
import datetime
import time
import re

def all_methods(value, mappings):
    if value not in mappings:
        new_id = len(mappings)
        mappings[value] = new_id
    return mappings[value]

def valid_values(value, valid_regex):
    for idx in range(len(valid_regex)):
        curr_regex = valid_regex[idx]
        if re.match(curr_regex, value):
            return idx + 1

    return 0

def convert_strings(table_col, table_schema, col_name, method):
    mappings = {}
    updated_series = []

    if method != "all":
        if "group_by_cols" in table_schema and col_name in table_schema["group_by_cols"]:
            # If this value is in a group by then we must map all of the values
            method = "all"
            print(col_name, "is in group by and thus has type", method)
        elif method == "only_valid":
            valid_key = col_name + "_valid"
            valid_regex = None
            mappings["all_others"] = 0
            if valid_key in table_schema:
                valid_sql_matches = table_schema[valid_key]
                valid_regex = ["^" + val.replace("%", ".*") + "$" for val in valid_sql_matches]
                for idx in range(len(valid_regex)):
                    curr_regex = valid_regex[idx]
                    curr_sql = valid_sql_matches[idx]
                    mappings[curr_sql] = idx + 1
            print("Col", col_name, "has valid values", valid_regex, "with mappings", mappings)

    for value in table_col.to_numpy():
        value = value.strip()
        mapped_int = None
        if method == "all":
            mapped_int = all_methods(value, mappings)
        else:
            mapped_int = 0
            if valid_regex is not None:
                mapped_int = valid_values(value, valid_regex)

        assert mapped_int is not None
        updated_series.append(mapped_int)

    return pd.Series(updated_series, copy = False), mappings

start_date = datetime.datetime(1970, 1, 1)
def convert_dates(table_col):
    updated_series = []
    date_format = "%Y-%m-%d"
    for value in table_col.to_numpy():
        date_time = datetime.datetime.strptime(value, date_format)
        days_diff = (date_time - start_date).days
        updated_series.append(int(days_diff))

    return pd.Series(updated_series, copy = False)

def convert_table(table_file, table_schema, schema_save_dir, table_save_dir, method):
    # Check if it has already been converted
    table_name = os.path.basename(table_file)
    save_path = os.path.join(table_save_dir, table_name)
    print("Processing table", table_name)

    # Read the table
    columns = table_schema["columns"]
    temp_name = columns + ["empty"]
    table_df = pd.read_csv(table_file, delimiter = "|", header = 0, names = temp_name)
    table_df = table_df.drop(columns=["empty"])
    
    converted_schema = {}
    for col_name in columns:
        column_type = table_schema[col_name]
        print("Processing col", col_name)
        if "str" in column_type:
            updated_series, mappings = convert_strings(table_df[col_name], table_schema, col_name, method)
            table_df[col_name] = updated_series
            converted_schema[col_name] = mappings
        elif "date" in column_type:
            updated_series = convert_dates(table_df[col_name])
            table_df[col_name] = updated_series
    
    # Save the results
    table_df.to_csv(save_path, sep = "|", index = False, header = False)
    table_name_without_ext = table_name[ : table_name.rindex(".")]
    schema_save_path = os.path.join(schema_save_dir, table_name_without_ext + "_" + method + ".json")
    with open(schema_save_path, "w+") as writer:
        json.dump(converted_schema, writer, indent = 4)

def runner(data_dir, method):
    # Get the paths
    tables_dir = os.path.join(data_dir, "s1")
    table_save_dir = os.path.join(data_dir, "s1_converted_" + method)
    os.makedirs(table_save_dir, exist_ok = True)
    schema_dir = os.path.join(data_dir, "schema_metadata")
    
    # Read the overall schema
    all_schema_file = os.path.join(schema_dir, "schema.json")
    with open(all_schema_file, 'r') as reader:
        schema = json.load(reader)

    for table_name in os.listdir(tables_dir):
        if ".tbl" not in table_name:
            continue
        
        table_name_without_dir = table_name[ : table_name.rindex(".")]
        table_schema = schema[table_name_without_dir]
        table_path = os.path.join(tables_dir, table_name)
        convert_table(table_path, table_schema, schema_dir, table_save_dir, method)

if __name__ == '__main__':
    # Read the args
    parser = argparse.ArgumentParser(description = 'convert')
    parser.add_argument('data_directory', type=str, help='Data Directory')
    parser.add_argument('--method', type=str, default="all")
    args = parser.parse_args()

    if args.method not in ["all", "only_valid"]:
        raise Exception("Invalid method type of", args.method)

    runner(args.data_directory, args.method)