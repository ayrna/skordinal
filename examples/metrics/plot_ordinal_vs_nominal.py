"""
Ordinal-aware versus nominal classification
===========================================

A nominal classifier treats the classes as unrelated labels, so a mistake
two classes away costs it no more than a mistake next door. An ordinal
classifier models the order and keeps its mistakes close. The Balance
Scale dataset makes the difference visible.
"""

# %%
# Load the Balance Scale dataset
# ------------------------------
# Each sample gives the weight and distance on both arms of a balance
# scale, and the class is where it tips: left, balanced or right. The
# classes are ordered, and the middle one is rare (49 of 625).

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skordinal.datasets import load_balance_scale

X, y = load_balance_scale(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape},  Test: {X_test.shape}")

# %%
# Fit a nominal and an ordinal model
# ----------------------------------
# ``LogisticRegression`` fits one score per class and picks the largest.
# ``LogisticAT`` fits one score for all classes and two ordered
# thresholds on it, so the classes are laid out on a line.

from sklearn.linear_model import LogisticRegression

from skordinal.classifiers import LogisticAT

nominal = LogisticRegression(max_iter=1000).fit(X_train, y_train)
ordinal = LogisticAT().fit(X_train, y_train)

y_pred_nom = nominal.predict(X_test)
y_pred_ord = ordinal.predict(X_test)

# %%
# Compare with ordinal metrics
# ----------------------------
# Accuracy counts every error once. MAE counts each by its distance, so
# calling a left-tipping scale right-tipping costs two. The last column
# counts exactly those two-step errors.

import numpy as np

from skordinal.metrics import accuracy_score, mean_absolute_error, weighted_kappa

print(
    f"{'':8s}  {'accuracy':>8s}  {'MAE':>6s}  {'kappa':>6s}  {'two-step errors':>15s}"
)
for name, y_pred in [("Nominal", y_pred_nom), ("Ordinal", y_pred_ord)]:
    far = int((np.abs(y_pred - y_test) >= 2).sum())
    print(
        f"{name:8s}  {accuracy_score(y_test, y_pred):8.3f}  "
        f"{mean_absolute_error(y_test, y_pred):6.3f}  "
        f"{weighted_kappa(y_test, y_pred):6.3f}  {far:15d}"
    )

# %%
# Confusion matrices
# ------------------
# The two-step errors sit in the corners: a left scale predicted right,
# or the reverse. The nominal model fills both corners and never predicts
# the rare middle class. The ordinal model almost empties the corners and
# its errors move next to the diagonal instead.

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name, y_pred in [
    (axes[0], "Nominal (LogisticRegression)", y_pred_nom),
    (axes[1], "Ordinal (LogisticAT)", y_pred_ord),
]:
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["left", "balanced", "right"],
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{name}\nMAE = {mean_absolute_error(y_test, y_pred):.3f}")
fig.tight_layout()
plt.show()
