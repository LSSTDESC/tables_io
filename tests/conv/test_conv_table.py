"""Tests for Converting Tables"""

import pytest

from tables_io import types
from tables_io.conv.conv_table import convert_table
from ..helpers.utilities import compare_table_dicts, check_deps
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
def test_convert_table(data_table, tType1, tType2):
    """Perform type conversion for tables."""
    t1 = convert_table(data_table, tType1)
    t2 = convert_table(t1, tType2)
    _ = convert_table(t2, types.AP_TABLE)


def test_bad_conversion(data_table):
    """Tests that the conversion fails as expected when given an incorrect table type."""

    # Testing Case Sensitivity
    with pytest.raises(TypeError) as e:
        bad = convert_table(data_table, "astropytable")

    # Testing against out of range key
    with pytest.raises(KeyError) as e:
        bad = convert_table(data_table, 500)

    # Testing Wrong Key
    with pytest.raises(TypeError) as e:
        bad = convert_table(data_table, "CSV")


@pytest.mark.skipif(
    not check_deps([apTable, pd]), reason="Missing panda or astropy.table"
)
def test_conversion_strings(data_table):
    """Tests that the conversion functions work when given table types as strings."""

    # test convert works with a string
    tType = types.TABULAR_FORMATS[0]
    tab = convert_table(data_table, tType)
