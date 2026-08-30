"""SVMOP ordered-partitions SVC decomposition on balance_scale."""

from sklearn.calibration import CalibratedClassifierCV
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
                estimator=CalibratedClassifierCV(estimator=SVC(), ensemble=False),
                decomposition="ordered_partitions",
                decision_method="frank_hall",
            ),
            param_grid={
                "estimator__estimator__C": [0.1, 1, 10],
                "estimator__estimator__gamma": [0.1, 1, 10],
            },
        ),
    },
}
