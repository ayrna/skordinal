"""
Per-class metrics for imbalanced targets
========================================

MAE averages the error over samples, so a class with few samples barely
moves it. AMAE averages the per-class errors instead, and MMAE keeps the
worst class. On an imbalanced dataset the three tell different stories,
and a model that wins on one can lose on another.
"""

# %%
# An imbalanced dataset
# ---------------------
# SWD has four ordered classes and the lowest one holds 32 of the 1000
# samples.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skordinal.datasets import load_swd

X, y = load_swd(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Class counts: {np.bincount(y).tolist()}")

# %%
# Two fits of the same model
# --------------------------
# ``LogisticAT`` with its default weights, and again with
# ``class_weight="balanced"``, which scales each sample by the inverse
# frequency of its class so the small class pulls on the fit as hard as
# the large ones.

from skordinal.classifiers import LogisticAT

models = {
    "default": LogisticAT(),
    "balanced": LogisticAT(class_weight="balanced"),
}
predictions = {
    name: clf.fit(X_train, y_train).predict(X_test) for name, clf in models.items()
}

# %%
# Three ways to average the error
# -------------------------------
# Per class, the error is the mean class distance of that class's
# samples. MAE weights those by class size, AMAE weights them equally
# and MMAE keeps the largest. The default fit never gets the small class
# right and still wins MAE. The balanced fit spreads the error across
# classes and wins AMAE and MMAE.

from skordinal.metrics import (
    average_mean_absolute_error,
    maximum_mean_absolute_error,
    mean_absolute_error,
)

classes = np.unique(y)
per_class = {}
print(f"{'':10s} {'MAE':>6s} {'AMAE':>6s} {'MMAE':>6s}")
for name, y_pred in predictions.items():
    per_class[name] = [np.abs(y_pred[y_test == k] - k).mean() for k in classes]
    print(
        f"{name:10s} {mean_absolute_error(y_test, y_pred):6.3f} "
        f"{average_mean_absolute_error(y_test, y_pred):6.3f} "
        f"{maximum_mean_absolute_error(y_test, y_pred):6.3f}"
    )

# %%
# Per-class error
# ---------------
# The bars are each class's error and the lines the three summaries. In
# the default fit the tallest bar belongs to class 0, and only AMAE and
# MMAE register it.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.bar(classes, per_class[name], 0.6, color="steelblue")
    for label, metric, style in [
        ("MAE", mean_absolute_error, "-"),
        ("AMAE", average_mean_absolute_error, "--"),
        ("MMAE", maximum_mean_absolute_error, ":"),
    ]:
        ax.axhline(
            metric(y_test, y_pred), color="darkorange", linestyle=style, label=label
        )
    ax.set_xticks(classes)
    ax.set_xlabel("True class")
    ax.set_title(f"LogisticAT ({name})")
axes[0].set_ylabel("Mean absolute error")
axes[0].legend()
fig.tight_layout()
plt.show()
