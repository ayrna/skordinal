"""NNPOM ordinal neural network classifier on balance_scale."""

RECIPE = {
    "general_conf": {
        "basedir": "skordinal/datasets/data",
        "datasets": ["balance_scale"],
        "hyperparam_cv_nfolds": 3,
        "jobs": 1,
        "input_preprocessing": "std",
        "output_folder": "results/",
        "metrics": [
            "accuracy_score",
            "mean_absolute_error",
            "mean_zero_one_error",
        ],
        "cv_metric": "neg_mean_absolute_error",
    },
    "configurations": {
        "NNPOM": {
            "classifier": "NNPOM",
            "parameters": {
                "epsilon_init": 0.5,
                "n_hidden": [5, 10, 20, 30, 40, 50],
                "max_iter": [250, 500],
                "alpha": [0.001, 0.01, 1],
            },
        },
    },
}
