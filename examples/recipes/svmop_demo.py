"""SVMOP ordered-partitions SVC decomposition on balance_scale."""

from sklearn.svm import SVC

from skordinal.classifiers import OrdinalDecomposition
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
        "SVMOP": ModelConfig(
            OrdinalDecomposition(
                dtype="ordered_partitions",
                decision_method="frank_hall",
                base_classifier=SVC(probability=True),
            ),
            param_grid={
                "base_classifier__C": [0.1, 1, 10],
                "base_classifier__gamma": [0.1, 1, 10],
            },
        ),
    },
}
