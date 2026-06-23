"""NNOP ordinal neural network classifier on balance_scale."""

from skordinal.classifiers import NNOP
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale"],
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": "std",
    "results_path": "results/",
    "eval_metrics": [
        "accuracy_score",
        "mean_absolute_error",
        "mean_zero_one_error",
    ],
    "tuning_metric": "neg_mean_absolute_error",
    "models": {
        "NNOP": ModelConfig(
            NNOP(epsilon_init=0.5),
            param_grid={
                "n_hidden": [5, 10, 20, 30, 40, 50],
                "max_iter": [250, 500],
                "alpha": [0.001, 0.01, 1],
            },
        ),
    },
}
