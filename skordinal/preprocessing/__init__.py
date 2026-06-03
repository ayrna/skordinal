"""Ordinal preprocessing utilities."""

from ._encodings import (
    binary_cumulative_to_ordinal,
    build_coding_matrix,
    ordinal_to_binary_cumulative,
)

__all__ = [
    "binary_cumulative_to_ordinal",
    "build_coding_matrix",
    "ordinal_to_binary_cumulative",
]
