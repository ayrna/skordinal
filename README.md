# skordinal

[![Unit tests](https://github.com/ayrna/skordinal/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/ayrna/skordinal/actions/workflows/unit-tests.yml) [![Coverage](https://img.shields.io/codecov/c/github/ayrna/skordinal?logo=codecov)](https://codecov.io/gh/ayrna/skordinal) [![Documentation](https://readthedocs.org/projects/skordinal/badge/?version=latest)](https://skordinal.readthedocs.io/en/latest/) [![PyPI](https://img.shields.io/pypi/v/skordinal)](https://pypi.org/project/skordinal/) [![Python](https://img.shields.io/pypi/pyversions/skordinal)](https://pypi.org/project/skordinal/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![License](https://img.shields.io/pypi/l/skordinal)](https://github.com/ayrna/skordinal/blob/main/LICENSE)

**skordinal** is a scikit-learn compatible library for **ordinal classification**, the supervised task whose class labels have a natural order but no known distance between them, such as a rating from *poor* to *excellent* or a disease graded from *I* to *IV*. A nominal classifier ignores that order, so confusing *poor* with *excellent* costs it exactly as much as confusing *poor* with *fair*. The estimators here model the order instead, and the metrics score a prediction against it.

Every estimator follows the scikit-learn API, so it drops unchanged into a `Pipeline`, a `GridSearchCV` and the rest of the ecosystem. skordinal is developed by the [AYRNA](https://www.uco.es/ayrna/) research group at the University of Córdoba and continues the work of ORCA and ORCA-python.

## Installation

```bash
pip install skordinal
```

Binary wheels are published for Linux x86-64. On every other platform pip builds the three C extensions (`REDSVM`, `SVOR`, `ORBoost`) from source, which requires a C/C++ compiler. Progress bars during a benchmark run need one optional extra, `pip install "skordinal[progress]"`.

## What is inside

- **Classifiers**: ten ordinal methods, from threshold models (`POM`, `LogisticAT`, `LogisticIT`) and neural networks (`NNPOM`, `NNOP`, `ELMOP`) to a kernel discriminant (`KDLOR`), support vector machines (`SVOR`, `REDSVM`) and boosting (`ORBoost`), plus three meta-estimators (`OrdinalDecomposition`, `RegressorWrapper`, `CostSensitiveWrapper`) that build an ordinal classifier out of any scikit-learn one.
- **Metrics**: thirteen ways to judge an ordinal prediction, including distance to the true class (MAE, AMAE, MMAE), exact and off-by-one accuracy, per-class sensitivity, weighted kappa, rank correlation and the ranked probability score. All but the last are available as scikit-learn scorers through `get_ordinal_scorer`.
- **Datasets and experiments**: five bundled ordinal benchmark datasets, a loader for your own CSV files with reproducible resamples, and a `Benchmark` that fits every combination of model, dataset and resample and writes the aggregated results to disk.

## Documentation

The [documentation](https://skordinal.readthedocs.io/en/latest/) is the place to start. The [getting started guide](https://skordinal.readthedocs.io/en/latest/getting_started.html) installs the library and fits a first model, the [gallery of examples](https://skordinal.readthedocs.io/en/latest/auto_examples/index.html) works through the classifiers, the metrics and the benchmarking harness, and the [API reference](https://skordinal.readthedocs.io/en/latest/api/index.html) documents every class and function.

## Contributing

Bug reports and pull requests are welcome on the [issue tracker](https://github.com/ayrna/skordinal/issues). Please open an issue before starting a large change so the work can be aligned with the project. For a development install:

```bash
git clone https://github.com/ayrna/skordinal.git
cd skordinal
pip install -e ".[dev,docs]"
pre-commit install
pytest
```

## Citation

If you use skordinal in your research, please cite it as:

```bibtex
@misc{skordinal2026,
  title  = {{skordinal: A scikit-learn compatible Python package for ordinal classification}},
  author = {Sevilla Molina, {\'A}ngel and Guijo-Rubio, David and Vargas, V{\'i}ctor Manuel and Guti{\'e}rrez, Pedro A.},
  year   = {2026},
  url    = {https://github.com/ayrna/skordinal}
}
```

## License

skordinal is distributed under the [BSD 3-Clause License](https://github.com/ayrna/skordinal/blob/main/LICENSE).
