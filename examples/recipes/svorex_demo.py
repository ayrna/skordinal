"""SVOREX kernel and regularisation grid search on balance_scale."""

from skordinal.classifiers import SVOREX
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
        "SVOREX": ModelConfig(
            SVOREX(kernel="rbf"),
            param_grid={
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
        ),
    },
}
