"""Load, fetch, and generate ordinal classification datasets."""

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
from ._tocuco import (
    download_tocuco,
    fetch_tocuco,
    fetch_tocuco_partition,
    list_tocuco_datasets,
)

__all__ = [
    "clear_data_home",
    "download_tocuco",
    "fetch_tocuco",
    "fetch_tocuco_partition",
    "get_data_home",
    "list_tocuco_datasets",
    "load_balance_scale",
    "load_dataset",
    "load_era",
    "load_esl",
    "load_lev",
    "load_partitions",
    "load_swd",
    "make_ordinal_classification",
]
