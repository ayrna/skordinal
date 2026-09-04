"""
Benchmarking classifiers across datasets
========================================

:class:`~skordinal.experiments.Benchmark` fits every combination of
model, dataset and resample, tunes every model that has a grid with an
ordinal scorer and
writes the per-resample results to disk. This example runs a small one
and reads the results back as a table and a chart.
"""

# %%
# Configure the benchmark
# -----------------------
# Two models, two bundled datasets and three stratified resamples.
# ``ModelConfig`` binds an estimator to an optional grid, tuned on every
# resample by cross-validation. The preprocessing is cloned per resample
# and fitted on its training split only. A real run uses ``resamples=30``
# and a results folder worth keeping. Here a temporary one is enough.

import tempfile

from sklearn.preprocessing import StandardScaler

from skordinal.classifiers import POM, SVOR
from skordinal.experiments import Benchmark, ModelConfig

results_path = tempfile.mkdtemp()

benchmark = Benchmark(
    models={
        "POM": ModelConfig(POM()),
        "SVOR": ModelConfig(SVOR(), param_grid={"C": [0.1, 1.0, 10.0]}),
    },
    datasets=["swd", "lev"],
    eval_metrics=["mean_absolute_error", "average_mean_absolute_error"],
    input_preprocessing=StandardScaler(),
    resamples=3,
    results_path=results_path,
    verbose=False,
)
benchmark.run()

# %%
# Read the results back
# ---------------------
# ``tabulate_results`` pivots the stored per-resample scores into a
# classifiers-by-datasets table of mean and standard deviation.

from skordinal.experiments import tabulate_results

print(tabulate_results(results_path, metric="mean_absolute_error"))

# %%
# Plot them
# ---------
# ``summarize`` returns the same aggregates as numbers, indexed by model
# and dataset, which is what a chart needs.

import matplotlib.pyplot as plt

from skordinal.experiments import summarize

mae = summarize(results_path)["mean_absolute_error_test"].unstack("dataset")

fig, ax = plt.subplots(figsize=(7, 4))
mae["mean"].T.plot.bar(yerr=mae["std"].T, ax=ax, capsize=3, rot=0)
ax.set_xlabel("")
ax.legend(title=None)
ax.set_ylabel("Test MAE (mean over 3 resamples)")
fig.tight_layout()
plt.show()
