import json
import argparse
from pipeline_function import run_using_function
from pipeline_data import run_using_data

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help="Configuration file in JSON format")
args = parser.parse_args()

with open(args.input_file) as f:
    config = json.load(f)

if config["model_type"] == "function":
    run_using_function(config)
elif config["model_type"] == "data":
    run_using_data(config)
else:
    raise RuntimeError('Unknown input in JSON configuration file. Valid options for "model_type" are "function"/"data".')
