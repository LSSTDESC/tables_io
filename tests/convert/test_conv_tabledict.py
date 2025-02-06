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
def test_convert_table_dicts(data_tables, tType1, tType2):
    """Perform type conversion on the cross-product of all types."""
    odict_1 = convert(data_tables, tType1)
    odict_2 = convert(odict_1, tType2)
    tables_r = convert(odict_2, types.AP_TABLE)
    assert compare_table_dicts(data_tables, tables_r)


def test_bad_conversion(data_table):
    """Tests that the conversion fails as expected when given an incorrect table type."""
    with pytest.raises(TypeError) as e:
        bad = convert(data_table, "astropytable")
