import os
import argparse
import json
import pandas as pd
import struct
import sys
import subprocess

# Data is written in little endian order
def write_to_file(col_val, col_type, writer):
    # Figure out format and value
    data_format = "<"
    if "decimal" in col_type:
        col_val = float(col_val)
        data_format += "f"
    else:
        col_val = int(col_val)
        data_format += "i"
    
    # Write the value to disk
    value = struct.pack(data_format, col_val)
    writer.write(value)

def make_columnar(table_path, table_schema, column_dir):
    file_name = os.path.basename(table_path)
    database_name = file_name[ : file_name.rindex(".")]

    print("Writing data for table", file_name)
    col_writers = {}
    col_names = table_schema["columns"]
    for col_name in col_names:
        save_name = database_name + "_" + col_name + ".tbl"
        save_dir = os.path.join(column_dir, save_name)

        save_file = open(save_dir, "wb+")
        col_writers[col_name] = save_file
    
    table_df = pd.read_csv(table_path, delimiter = '|', header = None, names = col_names)
    print("Total number of rows is", len(table_df.index))
    for idx, row in table_df.iterrows():
        for col_name in col_names:
            col_type = table_schema[col_name]
            col_val = row[col_name]
            write_to_file(col_val, col_type, col_writers[col_name])

    # Close the files
    for col_name in col_writers:
        col_writers[col_name].close()

processed_files = ["customer.tbl", "part.tbl", "lineitem.tbl", "region.tbl", "nation.tbl"]
def runner(data_dir):
    # Set the file paths
    converted_dir = os.path.join(data_dir, "s1_converted")
    column_dir = os.path.join(data_dir, "s1_columnar")
    os.makedirs(column_dir, exist_ok = True)
    schema_dir = os.path.join(data_dir, "schema_metadata")

    # Read the overall schema
    all_schema_file = os.path.join(schema_dir, "schema.json")
    with open(all_schema_file, 'r') as reader:
        schema = json.load(reader)

    for table_name in os.listdir(converted_dir):
        if ".tbl" not in table_name or table_name in processed_files:
            continue
        
        table_name_without_ext = table_name[ : table_name.rindex(".")]
        table_schema = schema[table_name_without_ext]
        table_path = os.path.join(converted_dir, table_name)
        make_columnar(table_path, table_schema, column_dir)

def verify(data_directory):
    converted_dir = os.path.join(data_directory, "s1_converted")
    column_dir = os.path.join(data_directory, "s1_columnar")
    schema_dir = os.path.join(data_directory, "schema_metadata")

    all_schema_file = os.path.join(schema_dir, "schema.json")
    with open(all_schema_file, 'r') as reader:
        schema = json.load(reader)
    
    for table_name in os.listdir(converted_dir):
        if ".tbl" not in table_name:
            break
        
        table_path = str(os.path.join(converted_dir, table_name))
        output = subprocess.check_output(["wc", "-l", table_path]).decode('ascii')
        num_lines = int(output.strip().split(" ")[0])
        target_size = 4 * num_lines
        
        table_without_txt = table_name[ : table_name.rindex(".")]
        table_schema = schema[table_without_txt]
        columns = table_schema["columns"]

        for col_name in columns:
            col_file = os.path.join(column_dir, table_without_txt + "_" + col_name + ".tbl")
            file_size = os.path.getsize(col_file)
            if file_size != target_size:
               print("File", col_file, "has size", file_size, "but expected size", target_size)

        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'convert')
    parser.add_argument('data_directory', type=str, help='Data Directory')
    args = parser.parse_args()

    verify(args.data_directory)
