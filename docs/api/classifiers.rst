Classifiers
===========

Every classifier here models the order of the target rather than treating the
classes as unrelated labels. They do so in different ways: by placing learned
thresholds on a latent scale, by predicting the cumulative distribution over
classes, or by decomposing the problem into ordered binary subproblems. The
three meta-estimators at the end build an ordinal classifier out of an
ordinary classifier or regressor.

.. currentmodule:: skordinal.classifiers

.. autosummary::
   :toctree: generated/

   POM
   LogisticAT
   LogisticIT
   NNPOM
   NNOP
   ELMOP
   KDLOR
   REDSVM
   SVOREX
   SVORIM
   ORBoost
   OrdinalDecomposition
   RegressorWrapper
   CostSensitiveWrapper
