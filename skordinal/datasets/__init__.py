"""Load and generate ordinal classification datasets."""

from ._base import (
    clear_data_home,
    get_data_home,
    load_dataset,
    load_partitions,
)
from ._loaders import (
    load_balance_scale,
    load_era,
    load_esl,
    load_lev,
    load_swd,
)
from ._samples_generator import make_ordinal_classification

__all__ = [
    "clear_data_home",
    "get_data_home",
    "load_balance_scale",
    "load_dataset",
    "load_era",
    "load_esl",
    "load_lev",
    "load_partitions",
    "load_swd",
    "make_ordinal_classification",
]
