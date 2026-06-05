"""Ordinal preprocessing utilities."""

from ._coding import (
    binary_cumulative_to_ordinal,
    build_coding_matrix,
    ordinal_to_binary_cumulative,
)

__all__ = [
    "binary_cumulative_to_ordinal",
    "build_coding_matrix",
    "ordinal_to_binary_cumulative",
]
