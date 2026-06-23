"""Several ordinal classifiers and a nominal baseline across three datasets."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from skordinal.classifiers import REDSVM, SVOREX, OrdinalDecomposition
from skordinal.experiments import ModelConfig

RECIPE = {
    "datasets": ["balance_scale", "era", "esl"],
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": "std",
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
                dtype="ordered_partitions",
                decision_method="frank_hall",
                base_classifier=SVC(probability=True),
            ),
            param_grid={
                "base_classifier__C": [0.01, 0.1, 1, 10],
                "base_classifier__gamma": [0.01, 0.1, 1, 10],
            },
        ),
        "LR": ModelConfig(
            OrdinalDecomposition(
                decision_method="exponential_loss",
                base_classifier=LogisticRegression(),
            ),
            param_grid={
                "dtype": ["ordered_partitions", "one_vs_next"],
                "base_classifier__C": [0.01, 0.1, 1, 10],
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
        "SVOREX": ModelConfig(
            SVOREX(kernel="rbf", tol=0.001),
            param_grid={
                "C": [0.1, 1, 10],
                "gamma": [0.1, 1, 10],
            },
        ),
    },
}
