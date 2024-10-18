import json
import argparse
from pipeline_function import run_using_function
from pipeline_data import run_using_data, process_resampled_sim_data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Configuration file in JSON format")
    parser.add_argument('--resampled_data', help="Configuration file in JSON format", action='store_true')
    args = parser.parse_args()

    with open(args.input_file) as f:
        config = json.load(f)

    if config["model_type"] == "function":
        run_using_function(config)

    elif config["model_type"] == "data":
        if not args.resampled_data:
            # Run initial pipeline to create autoencoders, surrogate models, resampled inputs
            run_using_data(config)
        else:
            # Process simulations outputs from resampled inputs
            process_resampled_sim_data(config)

    else:
        raise RuntimeError('Unknown input in JSON configuration file. Valid options for "model_type" are "function"/"data".')
