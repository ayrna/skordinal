Datasets
========

skordinal ships five classic ordinal benchmark datasets that can be loaded
without any external download. Each loader follows the scikit-learn
:class:`~sklearn.utils.Bunch` API: call with ``return_X_y=True`` to receive
``(X, y)`` NumPy arrays directly. Any other CSV dataset is resolved by name or
path with :func:`~skordinal.datasets.load_dataset`, its train/test resamples
are yielded by :func:`~skordinal.datasets.load_partitions`, and
:func:`~skordinal.datasets.make_ordinal_classification` generates synthetic
ordinal problems.

.. currentmodule:: skordinal.datasets

.. autosummary::
   :toctree: generated/

   load_balance_scale
   load_era
   load_esl
   load_lev
   load_swd
   load_dataset
   load_partitions
   make_ordinal_classification
   get_data_home
   clear_data_home
