# skordinal

| Overview  |                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD** | [![Unit tests](https://github.com/ayrna/skordinal/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/ayrna/skordinal/actions/workflows/unit-tests.yml) [![!python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/) |
| **Code**  | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![License - BSD 3-Clause](https://img.shields.io/pypi/l/pandas.svg)](https://github.com/ayrna/skordinal/blob/main/LICENSE) |


## What is skordinal?

**skordinal** is an experimental framework built on Python that integrates with scikit-learn to automate machine learning experiments through simple JSON configuration files. Initially designed for ordinal classification, it supports regular classification algorithms as long as they are compatible with scikit-learn, making it easy to run reproducible experiments across multiple datasets and classification methods.

## Table of Contents

- [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
    - [Testing Installation](#testing-installation)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
    - [settings](#settings)
    - [models](#models)
- [Running Experiments](#running-experiments)
    - [Basic Usage](#basic-usage)
    - [Example Output](#example-output)
- [License](#license)

## Installation

### Requirements

skordinal requires Python 3.10 or higher and is tested on Python 3.10, 3.11, 3.12, 3.13, and 3.14.

All dependencies are managed through `pyproject.toml` and include:
- numpy (>=1.21)
- pandas (>=1.0.1)
- scikit-learn (>=1.3.0)
- scipy (>=1.7)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ayrna/skordinal
   cd skordinal
   ```

2. **Install the framework**:
   ```bash
   pip install .
   ```

   For development purposes, use editable installation:
   ```bash
   pip install -e .
   ```

   Optional dependencies for development:
   ```bash
   pip install -e .[dev]
   ```

### Testing Installation

Test your installation with the provided example:

```bash
python examples/run_recipe.py examples/recipes/full_demo.py
```

## Quick Start

skordinal includes sample datasets with pre-partitioned train/test splits using a 30-holdout experimental design.

**Basic experiment configuration:**

A recipe is a Python file that defines a top-level `RECIPE` dict whose keys
map directly to `Benchmark` constructor parameters. The required keys are
`models`, `datasets`, `eval_metrics`, and `results_path`; all other keys are
optional and fall back to the `Benchmark` defaults.

```python
from sklearn.svm import SVC

from skordinal.classifiers import OrdinalDecomposition
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale", "era", "esl"],
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": "std",
    "results_path": "results/",
    "eval_metrics": [
        "accuracy_score",
        "mean_absolute_error",
        "average_mean_absolute_error",
        "mean_zero_one_error",
    ],
    "tuning_metric": "neg_mean_absolute_error",
    "models": {
        "SVM": ModelConfig(
            SVC(),
            param_grid={"C": [0.001, 0.1, 1, 10, 100], "gamma": [0.1, 1, 10]},
        ),
        "SVMOP": ModelConfig(
            OrdinalDecomposition(
                dtype="ordered_partitions",
                decision_method="frank_hall",
                base_classifier=SVC(probability=True),
            ),
            param_grid={
                "base_classifier__C": [0.01, 0.1, 1, 10],
                "base_classifier__gamma": [0.01, 0.1, 1, 10],
            },
        ),
    },
}
```

**Run the experiment:**
```bash
python examples/run_recipe.py my_experiment.py
```

Results are saved in `results/` folder with performance metrics for each dataset-classifier combination. The framework automatically performs cross-validation, hyperparameter tuning, and evaluation on test sets.

## Configuration Files

Experiments are defined as Python recipe files — a module that exposes a
top-level `RECIPE` dict. Every key in `RECIPE` corresponds to a `Benchmark`
constructor parameter. Recipes can be run from the command line or loaded
programmatically via `Benchmark.from_recipe("path/to/recipe.py")`.

### settings

These keys control how the benchmark is executed.

**Required:**
- **`datasets`**: list of dataset names. A loader or subfolder with each name
  must be available under `data_home`.
- **`eval_metrics`**: list of metric names computed on train and test for every
  partition.
- **`results_path`**: folder where result files are written.

**Optional:**
- **`tuning_metric`** (default `"neg_mean_absolute_error"`): scoring criterion
  passed to `GridSearchCV` to select the best hyperparameters.
- **`cv`** (default `3`): number of cross-validation folds.
- **`n_jobs`** (default `1`): parallel jobs for `GridSearchCV`.
- **`input_preprocessing`** (default `None`): `"std"` for standardisation,
  `"norm"` for normalisation, `None` for no scaling.
- **`resamples`** (default `30`): number of train/test resamples per dataset.
- **`data_home`** (default `None`): base directory for dataset files; `None`
  uses the bundled datasets.
- **`random_state`** (default `None`): integer seed for reproducibility.
- **`verbose`** (default `True`): print progress during the run.

### models

**Required.** A dict mapping a label (`str`) to a `ModelConfig` instance.
`ModelConfig` binds a scikit-learn-compatible estimator to an optional
`param_grid`.

- **`ModelConfig(estimator, param_grid=None)`**: wraps any estimator that
  implements the scikit-learn estimator interface. `param_grid` is a dict of
  hyperparameter name → list of values for `GridSearchCV`. For pipeline-style
  estimators (e.g. `OrdinalDecomposition`) use the double-underscore syntax
  (`"base_classifier__C"`) to target nested parameters.

## Running Experiments

### Basic Usage

```bash
python examples/run_recipe.py experiment_file.py
```

### Example Output

Results are stored in the specified output folder with detailed performance metrics and hyperparameter information for each dataset and configuration combination.

## License
[BSD 3](LICENSE)

<hr>

[Go to Top](#table-of-contents)
