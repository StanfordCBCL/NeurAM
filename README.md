# NeurAM: Nonlinear dimensionality reduction for uncertainty quantification through neural active manifolds

## Usage

```
neuram <path_to_input.json>
```

## Inputs

```json
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```

| Parameter    | Description | Required or optional (default value) |
| -------- | ------- | ------- |
| `model_type`  | Type of data    | Required |
| `number_of_iterations` | Number of independent trials to perform in constructing the shared space and surrogate models | 1 |
| `epochs`   | Number of epochs to train the autencoders and surrogate models | 10000 |
