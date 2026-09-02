"""
Decision regions of ordinal classifiers
=======================================

A threshold model scores each point on a single axis and cuts that axis
with ordered thresholds, so its decision regions follow the class order:
a point can only leave class k into class k-1 or k+1. This example draws
the regions of four ordinal classifiers on a 2-D problem, checks that
property on a fine grid, and contrasts it with a nominal classifier that
is free to interleave classes.
"""

# %%
# A 2-D ordinal dataset
# ---------------------
# 300 samples, 2 informative features and 4 ordered classes. The features
# are standardised so every model sees the same scale.

from sklearn.preprocessing import StandardScaler

from skordinal.datasets import make_ordinal_classification

X, y = make_ordinal_classification(
    n_samples=300, n_features=2, n_informative=2, n_classes=4, random_state=0
)
X = StandardScaler().fit_transform(X)

# %%
# Four ordinal classifiers and a nominal one
# ------------------------------------------
# ``POM`` and ``LogisticAT`` project linearly, ``NNPOM`` through a hidden
# layer and ``KDLOR`` through an RBF kernel, but all four threshold a
# one-dimensional score. ``KNeighborsClassifier`` is the nominal
# contrast: it votes among neighbours and knows nothing about the order.

from sklearn.neighbors import KNeighborsClassifier

from skordinal.classifiers import KDLOR, NNPOM, POM, LogisticAT

classifiers = {
    "POM": POM(),
    "LogisticAT": LogisticAT(),
    "NNPOM": NNPOM(max_iter=5000, random_state=0),
    "KDLOR": KDLOR(),
    "KNN (nominal)": KNeighborsClassifier(n_neighbors=5),
}
for clf in classifiers.values():
    clf.fit(X, y)

# %%
# Label a grid and count the forbidden jumps
# ------------------------------------------
# Each model labels every cell of a 200 x 200 grid. Two neighbouring
# cells whose classes differ by two or more steps are a jump the ordinal
# structure forbids: crossing from class 0 straight into class 2 without
# passing through class 1. The threshold models never do it. The nominal
# model does, wherever two classes that are not neighbours touch.

import numpy as np

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
)
grid = np.column_stack([xx.ravel(), yy.ravel()])

regions = {}
for name, clf in classifiers.items():
    Z = clf.predict(grid).reshape(xx.shape)
    regions[name] = Z
    jumps = (np.abs(np.diff(Z, axis=0)) >= 2).sum()
    jumps += (np.abs(np.diff(Z, axis=1)) >= 2).sum()
    print(f"{name:14s} jumps of two or more classes: {jumps}")

# %%
# Plot the regions
# ----------------
# A sequential colormap maps the class order to brightness, so the
# ordered bands read at a glance. The linear models draw parallel bands,
# the neural network bends them and the kernel model curves them into
# pockets, yet every region still touches only its neighbouring classes.
# The nominal panel has ragged borders and small pockets of one class
# inside another, and the count above says how often two classes that
# are not neighbours meet there.

import matplotlib.pyplot as plt

n_classes = len(np.unique(y))
levels = np.arange(n_classes + 1) - 0.5

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (name, Z) in zip(axes.ravel(), regions.items()):
    ax.contourf(xx, yy, Z, levels=levels, cmap="viridis", alpha=0.6)
    ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=15, linewidths=0.4
    )
    ax.set_title(name)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
axes.ravel()[-1].axis("off")
fig.tight_layout()
plt.show()
