"""Several ordinal classifiers and a nominal baseline across three datasets."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skordinal.classifiers import (
    REDSVM,
    SVOR,
    MLPTriangularClassifier,
    OrdinalDecomposition,
)
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale", "era", "swd"],
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": StandardScaler(),
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
            param_grid={
                "C": [0.001, 0.1, 1, 10, 100],
                "gamma": [0.1, 1, 10],
            },
        ),
        "SVMOP": ModelConfig(
            OrdinalDecomposition(
                estimator=CalibratedClassifierCV(estimator=SVC(), ensemble=False),
                decomposition="ordered_partitions",
                decision_method="frank_hall",
            ),
            param_grid={
                "estimator__estimator__C": [0.01, 0.1, 1, 10],
                "estimator__estimator__gamma": [0.01, 0.1, 1, 10],
            },
        ),
        "LR": ModelConfig(
            OrdinalDecomposition(
                estimator=LogisticRegression(),
                decision_method="exponential_loss",
            ),
            param_grid={
                "decomposition": ["ordered_partitions", "one_vs_next"],
                "estimator__C": [0.01, 0.1, 1, 10],
            },
        ),
        "REDSVM": ModelConfig(
            REDSVM(
                kernel="rbf",
                degree=3,
                gamma=0.1,
                coef0=0,
                C=1,
                tol=0.001,
                shrinking=True,
            ),
        ),
        "SVOR": ModelConfig(
            SVOR(kernel="rbf", tol=0.001),
            param_grid={
                "C": [0.1, 1, 10],
                "gamma": [0.1, 1, 10],
                "constraints": ["explicit", "implicit"],
            },
        ),
        "MLPTriangular": ModelConfig(
            MLPTriangularClassifier(random_state=42),
            param_grid={
                "activation": ["relu", "tanh"],
                "n_hidden_layers": [1, 2],
                "n_hidden_units": [10, 20],
                "alpha2": [0.01, 0.05, 0.1],
                "eta": [0.5, 1.0],
                "alpha": [0.001, 0.01],
                "max_iter": [500, 1000, 2000],
            },
        ),
    },
}
