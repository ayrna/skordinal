"""MLP softmax classifier on balance_scale."""

from sklearn.preprocessing import StandardScaler

from skordinal.classifiers import MLPSoftmaxClassifier
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale"],
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": StandardScaler(),
    "results_path": "results/",
    "eval_metrics": [
        "accuracy_score",
        "mean_absolute_error",
        "mean_zero_one_error",
    ],
    "tuning_metric": "neg_mean_absolute_error",
    "models": {
        "MLPSoftmax": ModelConfig(
            MLPSoftmaxClassifier(random_state=42),
            param_grid={
                "n_hidden_units": [10, 20],
                "alpha": [0.001, 0.01],
                "max_iter": [500, 1000],
            },
        ),
    },
}
