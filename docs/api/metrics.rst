Metrics
=======

An ordinal prediction can be wrong by one class or by five, and these metrics
tell those cases apart, which plain accuracy cannot. Thirteen of them score
predicted labels and fall into five groups: distance to the true class (MAE,
AMAE, MMAE), exact and off-by-one accuracy and its complement the error rate
(MZE), per-class sensitivity (MS, GMSEC, the geometric mean of the
sensitivities and the mean extreme sensitivity), agreement corrected for
chance (weighted kappa) and rank correlation (Kendall's tau, Spearman's rho).
The fourteenth, :func:`~skordinal.metrics.ranked_probability_score`, scores a
full predicted distribution instead.

:func:`~skordinal.metrics.get_ordinal_scorer` wraps any of them, except
:func:`~skordinal.metrics.ranked_probability_score`, as a scikit-learn scorer
for :class:`~sklearn.model_selection.GridSearchCV`.

.. currentmodule:: skordinal.metrics

.. autosummary::
   :toctree: generated/

   mean_absolute_error
   average_mean_absolute_error
   maximum_mean_absolute_error
   mean_zero_one_error
   accuracy_score
   accuracy_off1_score
   geometric_mean
   gmsec
   minimum_sensitivity
   mean_extreme_sensitivity
   kendalls_tau
   spearmans_rho
   weighted_kappa
   ranked_probability_score
   get_ordinal_scorer
   get_ordinal_scorer_names
