"""
Ordinal metrics penalise distant errors
=======================================

Accuracy treats every mistake the same. Ordinal metrics weight a mistake
by how far it lands from the true class, so two predictors with identical
accuracy can look very different under MAE, weighted kappa and off-by-one
accuracy.
"""

# %%
# A true ordinal target
# ---------------------
# A balanced 200-sample ground truth with 5 ordered classes: 40
# repetitions of ``[0, 1, 2, 3, 4]``.

import numpy as np

rng = np.random.default_rng(0)

y_true = np.tile(np.arange(5), 40)
print(f"Samples: {len(y_true)},  classes: {np.unique(y_true).tolist()}")

# %%
# Two predictors with the same accuracy
# -------------------------------------
# Both predictors corrupt the same 40 labels (20 % of the data), so their
# accuracy is equal by construction. ``y_near`` moves each corrupted label
# one step, ``y_far`` sends it to the opposite end of the scale. Only the
# distance of the errors differs.

n_corrupt = 40
corrupt_idx = rng.choice(len(y_true), size=n_corrupt, replace=False)

y_near = y_true.copy()
y_far = y_true.copy()

for i in corrupt_idx:
    orig = int(y_true[i])
    # One step up, or down at the top of the scale
    y_near[i] = orig - 1 if orig == 4 else orig + 1
    # The opposite end
    y_far[i] = 0 if orig >= 2 else 4

from skordinal.metrics import accuracy_score

acc_near = accuracy_score(y_true, y_near)
acc_far = accuracy_score(y_true, y_far)
print(f"Accuracy: y_near={acc_near:.3f}  y_far={acc_far:.3f}")

# %%
# Compare ordinal metrics
# -----------------------
# ``mean_absolute_error`` counts the class steps of each error, so a
# ``y_near`` error costs 1 and a ``y_far`` error costs 2 to 4.
# ``weighted_kappa`` penalises errors in proportion to their distance.
# ``accuracy_off1_score`` counts a prediction as correct when it is at
# most one step away, so every ``y_near`` error still passes.

from skordinal.metrics import (
    accuracy_off1_score,
    mean_absolute_error,
    weighted_kappa,
)

header = f"{'Predictor':10s}  {'Accuracy':>8s}  {'MAE':>6s}  "
header += f"{'Kappa':>7s}  {'Off-by-1':>8s}"
print(header)
print("-" * len(header))

for name, yp in [("y_near", y_near), ("y_far", y_far)]:
    acc = accuracy_score(y_true, yp)
    mae = mean_absolute_error(y_true, yp)
    kap = weighted_kappa(y_true, yp)
    off1 = accuracy_off1_score(y_true, yp)
    print(f"{name:10s}  {acc:>8.3f}  {mae:>6.3f}  {kap:>7.3f}  {off1:>8.3f}")

# %%
# Visualise
# ---------
# Same accuracy, and ``y_near`` wins every ordinal metric. Lower is
# better for MAE, higher for the other three.

import matplotlib.pyplot as plt

metric_names = ["Accuracy", "MAE", "Kappa", "Off-by-1 acc"]

values_near = [
    accuracy_score(y_true, y_near),
    mean_absolute_error(y_true, y_near),
    weighted_kappa(y_true, y_near),
    accuracy_off1_score(y_true, y_near),
]
values_far = [
    accuracy_score(y_true, y_far),
    mean_absolute_error(y_true, y_far),
    weighted_kappa(y_true, y_far),
    accuracy_off1_score(y_true, y_far),
]

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars_near = ax.bar(
    x - width / 2,
    values_near,
    width,
    label="y_near (adjacent errors)",
    color="steelblue",
)
bars_far = ax.bar(
    x + width / 2, values_far, width, label="y_far (distant errors)", color="darkorange"
)

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Metric value")
ax.set_title(
    "Same accuracy, different ordinal behaviour\n"
    "(MAE: lower is better, all others: higher is better)"
)
ax.legend()

for bar in list(bars_near) + list(bars_far):
    height = bar.get_height()
    ax.annotate(
        f"{height:.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
    )

fig.tight_layout()
plt.show()
