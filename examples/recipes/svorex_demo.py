"""SVOREX kernel and regularisation grid search on balance_scale."""

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
        "SVOREX": {
            "classifier": "SVOREX",
            "parameters": {
                "kernel": "rbf",
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
        },
    },
}
