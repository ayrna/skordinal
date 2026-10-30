Reductions
==========

Meta-estimators that split the ordinal problem into fixed subproblems, fit one
estimator on each and combine their outputs. The estimator is a binary
classifier or a regressor, or an ordinal threshold model in OPBE.

.. currentmodule:: skordinal.reduction

.. autosummary::
   :toctree: generated/

   OrdinalDecomposition
   RegressorWrapper
   CostSensitiveWrapper
   OPBE
