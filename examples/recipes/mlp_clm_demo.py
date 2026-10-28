"""MLP-CLM ordinal neural network classifier on balance_scale."""

from skordinal.classifiers import MLPCLMClassifier
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
        "MLP-CLM": ModelConfig(
            MLPCLMClassifier(random_state=42),
            param_grid={
                "n_hidden_layers": [1, 2],
                "n_hidden_units": [10, 20],
                "alpha": [0.001, 0.01],
                "max_iter": [500, 1000, 2000],
            },
        ),
    },
}
