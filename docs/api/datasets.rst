Datasets
========

skordinal ships five classic ordinal benchmark datasets that can be loaded
without any external download. Each loader follows the scikit-learn
:class:`~sklearn.utils.Bunch` API: call with ``return_X_y=True`` to receive
``(X, y)`` NumPy arrays directly. Any other CSV dataset is resolved by name or
path with :func:`~skordinal.datasets.load_dataset`, its train/test resamples
are yielded by :func:`~skordinal.datasets.load_partitions`, and
:func:`~skordinal.datasets.make_ordinal_classification` generates synthetic
ordinal problems. The TOC-UCO collection is fetched on demand and cached
under :func:`~skordinal.datasets.get_data_home`:
:func:`~skordinal.datasets.fetch_tocuco` loads one dataset,
:func:`~skordinal.datasets.fetch_tocuco_partition` one of its canonical
train/test partitions, :func:`~skordinal.datasets.download_tocuco`
prefetches the whole collection, and
:func:`~skordinal.datasets.list_tocuco_datasets` lists the cache.

The :func:`~skordinal.datasets.average_imbalance_ratio` and
:func:`~skordinal.datasets.maximum_imbalance_ratio` functions calculate class
imbalance from a one-dimensional target array. The average imbalance ratio
(AIR) averages the per-class imbalance ratios and is 1 for a balanced dataset,
and the maximum imbalance ratio (MIR) is the majority-class count divided by
the minority-class count. To use a dataset loaded by
:func:`~skordinal.datasets.load_dataset`, pass its ``target`` attribute. The
same functions work with labels from any other source.

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
   fetch_tocuco
   fetch_tocuco_partition
   download_tocuco
   list_tocuco_datasets
   make_ordinal_classification
   average_imbalance_ratio
   maximum_imbalance_ratio
   get_data_home
   clear_data_home
