import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str,
                    default='/home/merileo/vissl-env/lib/python3.7/site-packages/configs/config/dataset_catalog.json',
                    help='Path of shell script')

args = parser.parse_args()

json_path = args.json_path

# function to add to JSON
def write_json(new_data, filename='data.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["unions_200x200_ugriz_lsb_slurm"] = new_data
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


# python object to be appended
y = {"train": [os.getenv("SLURM_TMPDIR"), "<unused>"],
     "test": [os.getenv("SLURM_TMPDIR"), "<unused>"]}

write_json(y, json_path)
