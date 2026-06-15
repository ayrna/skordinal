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
    - [general-conf](#general-conf)
    - [configurations](#configurations)
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
python config.py examples/recipes/full_demo.py
```

## Quick Start

skordinal includes sample datasets with pre-partitioned train/test splits using a 30-holdout experimental design.

**Basic experiment configuration:**

A recipe is a Python file that defines a top-level `RECIPE` dict with two
sections: `general_conf` (run-wide settings) and `configurations` (classifier
definitions).

```python
RECIPE = {
    "general_conf": {
        "basedir": "skordinal/datasets/data",
        "datasets": ["balance_scale", "era", "esl"],
        "hyperparam_cv_nfolds": 3,
        "output_folder": "results/",
        "metrics": ["accuracy_score", "mean_absolute_error", "average_mean_absolute_error"],
        "cv_metric": "neg_mean_absolute_error",
    },
    "configurations": {
        "SVM": {
            "classifier": "SVC",
            "parameters": {
                "C": [0.001, 0.1, 1, 10, 100],
                "gamma": [0.1, 1, 10],
            },
        },
        "SVMOP": {
            "classifier": "OrdinalDecomposition",
            "parameters": {
                "dtype": "ordered_partitions",
                "decision_method": "frank_hall",
                "base_classifier": "SVC",
                "parameters": {
                    "C": [0.01, 0.1, 1, 10],
                    "gamma": [0.01, 0.1, 1, 10],
                    "probability": [True],
                },
            },
        },
    },
}
```

**Run the experiment:**
```bash
python config.py my_experiment.py
```

Results are saved in `results/` folder with performance metrics for each dataset-classifier combination. The framework automatically performs cross-validation, hyperparameter tuning, and evaluation on test sets.

## Configuration Files

Experiments are defined using Python recipe files (a module exposing a top-level `RECIPE` dict) with two main sections: general_conf for experiment settings and configurations for classifier definitions.

### general-conf

Controls global experiment parameters.

**Required parameters:**
- **`basedir`**: folder containing all dataset subfolders, it doesn't allow more than one folder at a time. It can be indicated using a full path, or a relative one to the framework folder.
- **`datasets`**: name of datasets that will be experimented with. A subfolder with the same name must exist inside `basedir`.

**Optional parameters:**

- **`hyperparam_cv_nfolds`**: number of folds used while cross-validating.
- **`jobs`**: number of jobs used for GridSearchCV during cross-validation.
- **`input_preprocessing`**: data preprocessing (`"std"` for standardization, `"norm"` for normalization, `""` for none)
- **`output_folder`**: name of the folder where all experiment results will be stored.
- **`metrics`**: name of the accuracy metrics to measure the train and test performance of the classifier.
- **`cv_metric`**: error measure used for GridSearchCV to find the best set of hyper-parameters.

### configurations

Defines classifiers and their hyperparameters for GridSearchCV. Each configuration has a name and consists of:

- **`classifier`**: scikit-learn or built-in skordinal classifier
- **`parameters`**: hyperparameters for grid search (nested for ensemble methods)

## Running Experiments

### Basic Usage

```bash
python config.py experiment_file.py
```

### Example Output

Results are stored in the specified output folder with detailed performance metrics and hyperparameter information for each dataset and configuration combination.

## License
[BSD 3](LICENSE)

<hr>

[Go to Top](#table-of-contents)
