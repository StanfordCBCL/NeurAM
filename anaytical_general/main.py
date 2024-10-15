import json
import argparse
from run_test import run_using_model

parser = argparse.ArgumentParser()
parser.add_argument('input_file', help="Configuration file in JSON format")
args = parser.parse_args()
with open(args.input_file) as f:
    config = json.load(f)

run_using_model(config)
