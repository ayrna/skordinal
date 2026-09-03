Getting Started
===============

Installation
------------

Install the latest release from PyPI::

   pip install skordinal

Binary wheels are published for Linux x86-64. On every other platform pip
builds the three C extensions (``REDSVM``, ``SVOR``, ``ORBoost``)
from source, which requires a C/C++ compiler.

Progress bars during a benchmark run need one optional extra::

   pip install "skordinal[progress]"

For a development install, in editable mode and with every optional
dependency::

   git clone https://github.com/ayrna/skordinal.git
   cd skordinal
   pip install -e ".[dev,docs]"

Quickstart
----------

The example below loads a bundled ordinal dataset, trains a ``POM``
classifier, and evaluates it with two ordinal metrics::

   from sklearn.model_selection import train_test_split

   from skordinal.datasets import load_era
   from skordinal.classifiers import POM
   from skordinal.metrics import mean_absolute_error, weighted_kappa

   X, y = load_era(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=0, stratify=y
   )

   clf = POM()
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)

   print(f"MAE:   {mean_absolute_error(y_test, y_pred):.3f}")
   print(f"Kappa: {weighted_kappa(y_test, y_pred):.3f}")

Every classifier implements the standard scikit-learn ``fit`` / ``predict``
interface, so it composes with :class:`~sklearn.pipeline.Pipeline`,
:func:`~sklearn.model_selection.cross_val_score` and
:class:`~sklearn.model_selection.GridSearchCV` without adaptation. The
:doc:`api/index` documents every class and function.
