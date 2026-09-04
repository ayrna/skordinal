"""
Quickstart: an ordinal classifier in a scikit-learn pipeline
============================================================

Every skordinal estimator follows the scikit-learn API, so it drops into
a :class:`~sklearn.pipeline.Pipeline` and a
:class:`~sklearn.model_selection.GridSearchCV` unchanged. This example
fits a support vector ordinal regressor on a bundled dataset, tunes it
with an ordinal scorer and evaluates it with three ordinal metrics.
"""

# %%
# Load a bundled dataset
# ----------------------
# SWD has 1000 samples, 10 features and 4 ordered classes. The split is
# stratified so every class keeps its share in both partitions.

from sklearn.model_selection import train_test_split

from skordinal.datasets import load_swd

X, y = load_swd(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")

# %%
# Tune an ordinal classifier inside a pipeline
# --------------------------------------------
# ``SVOR`` is a regular estimator, so it takes the last step of a pipeline
# after a scaler. The search is driven by an ordinal scorer: MAE charges a
# prediction by how many classes it lands from the truth, so the search
# prefers models whose mistakes stay close.

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skordinal.classifiers import SVOR
from skordinal.metrics import get_ordinal_scorer

pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", SVOR())])

C_grid = np.logspace(-2, 2, 9)
search = GridSearchCV(
    pipeline,
    param_grid={"classifier__C": C_grid},
    scoring=get_ordinal_scorer("neg_mean_absolute_error"),
    cv=3,
)
search.fit(X_train, y_train)

print(f"Best C: {search.best_params_['classifier__C']:.3g}")

# %%
# The validation curve
# --------------------
# The scorer is negated MAE, so the curve is flipped back to MAE for
# reading. A small ``C`` underfits, and past the optimum the extra
# flexibility buys nothing.

import matplotlib.pyplot as plt

cv_mae = -search.cv_results_["mean_test_score"]
cv_std = search.cv_results_["std_test_score"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(C_grid, cv_mae, yerr=cv_std, marker="o", capsize=3)
ax.axvline(
    search.best_params_["classifier__C"], color="grey", linestyle="--", linewidth=0.8
)
ax.set_xscale("log")
ax.set_xlabel("C")
ax.set_ylabel("Cross-validated MAE")
ax.set_title("SVOR regularisation on SWD (3-fold CV)")
fig.tight_layout()
plt.show()

# %%
# Evaluate with ordinal metrics
# -----------------------------
# The fitted search predicts like any estimator. MAE averages the class
# distance of every error, AMAE averages it per class first so a small
# class counts as much as a large one, and weighted kappa is agreement
# corrected for chance with linear ordinal weights.

from skordinal.metrics import (
    average_mean_absolute_error,
    mean_absolute_error,
    weighted_kappa,
)

y_pred = search.predict(X_test)

print(f"MAE:   {mean_absolute_error(y_test, y_pred):.3f}")
print(f"AMAE:  {average_mean_absolute_error(y_test, y_pred):.3f}")
print(f"Kappa: {weighted_kappa(y_test, y_pred):.3f}")
