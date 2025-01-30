"""Tests for Converting Tabledicts"""

import pytest

from tables_io import types
from tables_io.convert.conv_tabledict import convert
from ..testUtils import compare_table_dicts, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp

# TODO: Docstrings for all these functions


@pytest.mark.parametrize(
    "mod",
    [tables, apTable, apDiffUtils, fits, h5py, pd, pq],
)
def test_deps(mod):
    assert check_deps([mod])
