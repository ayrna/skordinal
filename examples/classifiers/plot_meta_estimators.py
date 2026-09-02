"""
Building an ordinal classifier from a scikit-learn estimator
============================================================

Three meta-estimators turn an ordinary classifier or regressor into an
ordinal one, each by a different route: decompose the order into binary
questions, reweight the errors of a one-vs-rest scheme by their
distance, or regress on the class rank and round. This example runs the
three on the LEV dataset next to the plain estimator they wrap.
"""

# %%
# Load the LEV dataset
# --------------------
# 1000 samples, 4 features and 5 ordered classes.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skordinal.datasets import load_lev

X, y = load_lev(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")

# %%
# The three routes
# ----------------
# ``OrdinalDecomposition`` asks one binary question per boundary, "is the
# class above k?", and combines the answers. ``CostSensitiveWrapper``
# trains one one-vs-rest classifier per class and weights each negative
# sample by its distance to that class. ``RegressorWrapper`` fits a
# regressor to the class rank and rounds its output to the nearest one.
# The first two wrap a ``LogisticRegression``, the last a random forest.
# A plain ``LogisticRegression`` is the nominal baseline.

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from skordinal.classifiers import (
    CostSensitiveWrapper,
    OrdinalDecomposition,
    RegressorWrapper,
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "OrdinalDecomposition": OrdinalDecomposition(LogisticRegression(max_iter=1000)),
    "CostSensitiveWrapper": CostSensitiveWrapper(LogisticRegression(max_iter=1000)),
    "RegressorWrapper": RegressorWrapper(RandomForestRegressor(random_state=0)),
}

# %%
# Fit and score
# -------------
# Which route helps depends on the data. Here the regression route gains
# the most, and the other two stay close to the nominal baseline.

from skordinal.metrics import accuracy_score, mean_absolute_error, weighted_kappa

predictions = {}
print(f"{'':22s} {'accuracy':>8s} {'MAE':>6s} {'kappa':>6s}")
for name, clf in models.items():
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    predictions[name] = y_pred
    print(
        f"{name:22s} {accuracy_score(y_test, y_pred):8.3f} "
        f"{mean_absolute_error(y_test, y_pred):6.3f} "
        f"{weighted_kappa(y_test, y_pred):6.3f}"
    )

# %%
# What the regressor sees
# -----------------------
# The wrapped forest predicts a continuous rank. ``RegressorWrapper``
# rounds it at the midpoints between ranks, drawn here as dashed lines,
# and its ``thresholds_`` are those midpoints. A point lands in the wrong
# class when the forest's output falls past a midpoint, so the mistakes
# cluster near the lines and almost never reach a class two steps away.
# The forest never outputs past the last midpoint, so the rare top class
# is never predicted.

import matplotlib.pyplot as plt
import numpy as np

wrapper = models["RegressorWrapper"]
score = wrapper.estimator_.predict(X_test)
correct = predictions["RegressorWrapper"] == y_test

rng = np.random.default_rng(0)
jitter = rng.uniform(-0.15, 0.15, size=len(y_test))

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(
    score[correct],
    y_test[correct] + jitter[correct],
    s=12,
    color="steelblue",
    label="correct",
)
ax.scatter(
    score[~correct],
    y_test[~correct] + jitter[~correct],
    s=12,
    color="darkorange",
    label="wrong",
)
for t in wrapper.thresholds_:
    ax.axvline(t, color="grey", linestyle="--", linewidth=0.8)
ax.set_yticks(wrapper.classes_)
ax.set_xlabel("Random forest output (class rank)")
ax.set_ylabel("True class")
ax.set_title("RegressorWrapper: round the regression to the nearest rank")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
