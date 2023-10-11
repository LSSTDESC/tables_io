"""
Unit tests for io_layer module
"""

import pytest
from tables_io import types, convert, convertObj
from tables_io.testUtils import compare_table_dicts, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp


@pytest.mark.parametrize(
    "mod",
    [tables, apTable, apDiffUtils, fits, h5py, pd, pq],
)
def test_deps(mod):
    assert check_deps([mod])


@pytest.mark.skipif(not check_deps([jnp]), reason="Failed to load jax.numpy")
def test_deps_jnp():
    assert check_deps([jnp])


def test_bad_deps():
    dummy = 0
    assert not check_deps([dummy])
    

@pytest.mark.skipif(not check_deps([apTable, pd]), reason="Missing panda or astropy.table")
@pytest.mark.parametrize(
    "tType1",
    [types.AP_TABLE, types.NUMPY_DICT, types.NUMPY_RECARRAY, types.PD_DATAFRAME],
)
@pytest.mark.parametrize(
    "tType2",
    [types.AP_TABLE, types.NUMPY_DICT, types.NUMPY_RECARRAY, types.PD_DATAFRAME],
)
def test_type_conversion(data_tables, data_table, tType1, tType2):
    """Perform type conversion on the cross-product of all types."""
    odict_1 = convert(data_tables, tType1)
    odict_2 = convert(odict_1, tType2)
    tables_r = convert(odict_2, types.AP_TABLE)
    assert compare_table_dicts(data_tables, tables_r)
    t1 = convertObj(data_table, tType1)
    t2 = convertObj(t1, tType2)
    _ = convertObj(t2, types.AP_TABLE)
