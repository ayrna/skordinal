skordinal
=========

**skordinal** is a scikit-learn compatible library for **ordinal
classification**, the supervised task whose class labels have a natural order
but no known distance between them, such as a rating from ``poor`` to
``excellent`` or a disease graded from ``I`` to ``IV``.

A nominal classifier ignores that order, so confusing ``poor`` with
``excellent`` costs it exactly as much as confusing ``poor`` with ``fair``.
The estimators here model the order instead, and the metrics score a
prediction against it.

Every estimator follows the scikit-learn API, so it drops unchanged into
:class:`~sklearn.pipeline.Pipeline`,
:class:`~sklearn.model_selection.GridSearchCV` and the rest of the ecosystem.
See :doc:`getting_started` to install it and fit the first model, then
:doc:`auto_examples/index` for worked examples.

At a glance
-----------

**Thirteen ordinal classifiers**: threshold models
(:class:`~skordinal.linear_model.POM`,
:class:`~skordinal.linear_model.LogisticAT`,
:class:`~skordinal.linear_model.LogisticIT`), neural networks
(:class:`~skordinal.neural_network.NNPOM`,
:class:`~skordinal.neural_network.NNOP`,
:class:`~skordinal.neural_network.ELMOP`), kernel discriminants
(:class:`~skordinal.discriminant_analysis.KDLOR`), support vector machines
(:class:`~skordinal.svm.REDSVM`,
:class:`~skordinal.svm.SVOR`), boosting
(:class:`~skordinal.ensemble.ORBoost`) and three meta-estimators
(:class:`~skordinal.reduction.OrdinalDecomposition`,
:class:`~skordinal.reduction.RegressorWrapper`,
:class:`~skordinal.reduction.CostSensitiveWrapper`) that build an ordinal
classifier out of an ordinary one.

**Fourteen metrics** spanning the ways an ordinal prediction can be judged:
distance to the true class (MAE, AMAE, MMAE), exact and off-by-one accuracy
and its complement the error rate (MZE), per-class sensitivity (MS, GMSEC and
two more), agreement corrected for chance (weighted kappa), rank correlation
(Kendall's tau, Spearman's rho) and the full predicted distribution (ranked
probability score). All but the last are available as scikit-learn scorers
through :func:`~skordinal.metrics.get_ordinal_scorer`.

**Datasets and an experiment harness**: five bundled ordinal datasets, a
loader for your own CSV files with reproducible resamples, and a
:class:`~skordinal.experiments.Benchmark` that fits every combination of
model, dataset and resample and writes the aggregated results to disk.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   auto_examples/index
   api/index
