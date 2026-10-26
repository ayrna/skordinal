"""EBANO soft-voting ensemble on balance_scale."""

from skordinal.classifiers import EBANO, LogisticAT, LogisticIT
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale"],
    "cv": 5,
    "n_jobs": 1,
    "results_path": "results/",
    "eval_metrics": [
        "accuracy_score",
        "mean_absolute_error",
        "average_mean_absolute_error",
    ],
    "tuning_metric": "neg_mean_absolute_error",
    "models": {
        "EBANO": ModelConfig(
            EBANO(
                [("at", LogisticAT()), ("it", LogisticIT())],
                cv=3,
                n_iter=200,
                random_state=0,
            ),
            param_grid={
                "at__alpha": [0.01, 0.1, 1.0],
                "it__alpha": [0.01, 0.1, 1.0],
                "n_iter": [200, 300],
            },
        ),
    },
}
