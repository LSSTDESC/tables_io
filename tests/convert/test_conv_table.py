"""Tests for Converting Tables"""

import pytest

from tables_io import types
from tables_io.conv.conv_table import convert_table
from ..testUtils import compare_table_dicts, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp


@pytest.mark.skipif(
    not check_deps([apTable, pd]), reason="Missing panda or astropy.table"
)
@pytest.mark.parametrize(
    "tType1",
    [
        types.AP_TABLE,
        types.NUMPY_DICT,
        types.NUMPY_RECARRAY,
        types.PA_TABLE,
        types.PD_DATAFRAME,
    ],
)
@pytest.mark.parametrize(
    "tType2",
    [
        types.AP_TABLE,
        types.NUMPY_DICT,
        types.NUMPY_RECARRAY,
        types.PA_TABLE,
        types.PD_DATAFRAME,
    ],
)
def test_convert_table(data_tables, data_table, tType1, tType2):
    """Perform type conversion for tables."""
    t1 = convert_table(data_table, tType1)
    t2 = convert_table(t1, tType2)
    _ = convert_table(t2, types.AP_TABLE)
