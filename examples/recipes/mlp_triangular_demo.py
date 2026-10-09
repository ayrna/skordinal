"""MLP Triangular ordinal neural network classifier on balance_scale."""

from skordinal.classifiers import MLPTriangularClassifier
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale"],
    "cv": 3,
    "n_jobs": 1,
    "results_path": "results/",
    "eval_metrics": [
        "accuracy_score",
        "mean_absolute_error",
        "mean_zero_one_error",
    ],
    "tuning_metric": "neg_mean_absolute_error",
    "models": {
        "MLPTriangular": ModelConfig(
            MLPTriangularClassifier(random_state=42),
            param_grid={
                "activation": ["relu", "tanh"],
                "n_hidden_layers": [1, 2],
                "n_hidden_units": [10, 20],
                "t_alpha": [0.01, 0.05, 0.1],
                "eta": [0.5, 1.0],
                "alpha": [0.001, 0.01],
                "max_iter": [500, 1000, 2000],
            },
        ),
    },
}
