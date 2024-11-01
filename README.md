# NeurAM: Nonlinear dimensionality reduction through neural active manifolds

## Usage

```
neuram <path_to_input.json>
```

## Inputs

This package uses a `.json` input file with the following format to specify the parameters for the construction of the reduced dimension and surrogate models:

```json
{
    "model_type" : "data",
    "number_of_iterations" : 1,
    "epochs" : 10000,
    "random_seed" : 2025,
    "save" : true,
    "hyperparameter_tuning" : "hyperparameters.txt",
    "model" : { ... }
}
```

### Common parameters

| Parameter    | Options | Description | Default value |
| -------- | ------- | ------- | ------- |
| `"model_type"`  | `"function"`/`"data"` | If the low- and high-fidelity models are provided as Python functions, this should be `"function"`. The Python function should be in a named `get_model(name)`, where the argument specifies the name of the QoI, in a script called `model.py`. If the low- and high-fidelity models are provided as data (from simulations/measurements performed separately), this should be `"data"`. See the examples for details. | - |
| `"number_of_iterations"` | Integer >= 1  | Number of independent trials to perform in constructing the shared space and surrogate models. | 1 |
| `"epochs"`   | Integer >= 1 | Number of epochs to train the autencoders and surrogate models | 10000 |
| `"random_seed"` | Integer  | Seed for the random number generators. Providing a seed makes the code more repeatable. | False |
| `"save"` | `true`/`false` | Toggles whether to save data files. | True |
| `"hyperparameter_tuning"` | `true`/`false`/`"file_name"` | Specifies whether to tune hyperparameters using Optuna (when set to `true`), use default hyperparameters (when set to `false`) or read hyperparameters from a file (when provided a `"file_name"`)  | False |

### The `model` block for models of type `"function"`

```json
{   ...,
    "model" : {
        "model_path" : "./",
        "number_of_training_samples" : 1000,
        "number_of_testing_samples" : 1000,
        "HF_QoI_name" : "Q_HF",
        "LF_QoI_name" : "Q_LF",
        "cost_ratio" : 0.01
    }
}
```

### The `model` block for models of type `"data"`

```json
{   ...,
    "model" : {
        "HF_QoI_name" : "max_osi_sten_lad",
        "LF_QoI_name" : "mean_flow:lca1:BC_lca1",
        "cost_ratio" : 4.1275E-6,
        "data_files" : {
            "HF_inputs" : "./simulations/all_param_values.json",
            "HF_outputs" : "./simulations/all_3d_data.json",
            "LF_inputs_pilot" : "./simulations/all_param_values.json",
            "LF_outputs_pilot" : "./simulations/all_0d_data.json",
            "LF_inputs_propagation" : "./simulations/all_param_values_propagation.json",
            "LF_outputs_propagation" : "./simulations/all_0d_data_propagation.json",
            "LF_outputs_pilot_AE" : "./simulations/all_0d_data_AE.json",
            "LF_outputs_propagation_AE" : "./simulations/all_0d_data_AE_propagation.json",
            "LF_inputs_limits" : "./simulations/param_limits.json",
            "HF_inputs_limits" : "./simulations/param_limits.json"
        },
        "num_pilot_samples" : -1,
        "load_NN_models": "./"
    }
}
```
