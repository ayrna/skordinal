"""
Visualising cumulative probabilities
====================================

Threshold (cumulative-link) models predict P(Y <= k | x) for each class
boundary, and the class boundaries are ordered thresholds on one latent
score.
"""

# %%
# A one-dimensional ordinal problem
# ---------------------------------
# A small synthetic dataset with a single feature, so the decision
# boundaries can be read directly off the x-axis.

from skordinal.datasets import make_ordinal_classification

X, y = make_ordinal_classification(
    n_samples=400,
    n_features=1,
    n_classes=4,
    n_informative=1,
    noise=0.4,
    random_state=0,
)

print(f"Shape: {X.shape},  classes: {sorted(set(y.tolist()))}")

# %%
# Fit a cumulative-link model
# ---------------------------
# ``LogisticAT`` learns one weight per feature and K-1 ordered thresholds,
# one for each class boundary. With a single feature the latent score is
# ``coef_ * x``. Here the coefficient comes out negative, so the classes
# run from 3 on the left of the feature axis to 0 on the right.

import numpy as np

from skordinal.classifiers import LogisticAT

clf = LogisticAT()
clf.fit(X, y)

print(f"Coefficient: {clf.coef_[0]:.3f}")
print(f"Thresholds:  {np.round(clf.thresholds_, 3)}")

# %%
# Cumulative probability curves
# -----------------------------
# ``predict_cumproba`` returns P(Y <= k | x) for k = 0, 1, 2. The curves
# are monotone and nested: P(Y <= 0) <= P(Y <= 1) <= P(Y <= 2). The k-th
# curve crosses 0.5 where the latent score equals the k-th threshold,
# that is at ``x = thresholds_[k] / coef_``.

import matplotlib.pyplot as plt

grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
cum = clf.predict_cumproba(grid)  # shape (300, K-1)

fig, ax = plt.subplots(figsize=(8, 4))
for k in range(cum.shape[1]):
    ax.plot(grid, cum[:, k], label=f"P(Y ≤ {k})")
ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="0.5 line")
ax.set_xlabel("Feature value")
ax.set_ylabel("Cumulative probability")
ax.set_title("Cumulative probability curves (LogisticAT)")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Per-class probabilities
# -----------------------
# ``predict_proba`` returns P(Y = k | x) for each class. These are the
# successive differences of the cumulative curves: P(Y = k) =
# P(Y <= k) - P(Y <= k-1). The stacked-area chart makes the ordered
# structure immediately visible.

proba = clf.predict_proba(grid)  # shape (300, K)

fig, ax = plt.subplots(figsize=(8, 4))
ax.stackplot(
    grid.ravel(),
    proba.T,
    labels=[f"class {k}" for k in clf.classes_],
    alpha=0.75,
)
ax.set_xlabel("Feature value")
ax.set_ylabel("Class probability")
ax.set_title("Per-class probabilities (successive differences of cum. curves)")
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
