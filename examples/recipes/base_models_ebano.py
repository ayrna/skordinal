import numpy as np
from sklearn.preprocessing import StandardScaler

from skordinal.classifiers import LogisticAT, LogisticIT
from skordinal.datasets import download_tocuco
from skordinal.experiments import ModelConfig

cv_hidden_units = [5, 8, 10, 15, 20, 50, 100]
cv_max_iter = [1000, 1500, 3000, 5000]
cv_alpha = [0.001, 0.01, 0.1]
tocuco_root = download_tocuco()

RECIPE = {
    "datasets": ["dr04_machine"],
    "data_home": tocuco_root,
    "cv": 3,
    "n_jobs": 1,
    "input_preprocessing": StandardScaler(),
    "results_path": "results/base_models/",
    "eval_metrics": ["mean_absolute_error", "weighted_kappa"],
    "tuning_metric": "neg_mean_absolute_error",
    "models": {
        "LogisticAT": ModelConfig(
            LogisticAT(),
            param_grid={
                "alpha": np.logspace(-3, 3, 7).tolist(),
                "max_iter": cv_max_iter,
            },
        ),
        "LogisticIT": ModelConfig(
            LogisticIT(),
            param_grid={
                "alpha": np.logspace(-3, 3, 7).tolist(),
                "max_iter": cv_max_iter,
            },
        ),
    },
}
