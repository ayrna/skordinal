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
See :doc:`getting_started` to install it and fit the first model.

At a glance
-----------

**Fourteen ordinal classifiers**: threshold models
(:class:`~skordinal.classifiers.POM`,
:class:`~skordinal.classifiers.LogisticAT`,
:class:`~skordinal.classifiers.LogisticIT`), neural networks
(:class:`~skordinal.classifiers.NNPOM`,
:class:`~skordinal.classifiers.NNOP`,
:class:`~skordinal.classifiers.ELMOP`), kernel discriminants
(:class:`~skordinal.classifiers.KDLOR`), support vector machines
(:class:`~skordinal.classifiers.REDSVM`,
:class:`~skordinal.classifiers.SVOREX`,
:class:`~skordinal.classifiers.SVORIM`), boosting
(:class:`~skordinal.classifiers.ORBoost`) and three meta-estimators
(:class:`~skordinal.classifiers.OrdinalDecomposition`,
:class:`~skordinal.classifiers.RegressorWrapper`,
:class:`~skordinal.classifiers.CostSensitiveWrapper`) that build an ordinal
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
   api/index
