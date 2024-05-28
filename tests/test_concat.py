"""
Unit tests for io_layer module
"""

import pytest
from tables_io import types, convert, concat, sliceObj, sliceObjs
from tables_io.testUtils import compare_tables, check_deps
from tables_io.lazy_modules import tables, apTable, apDiffUtils, fits, h5py, pd, pq, jnp


@pytest.mark.skipif(not check_deps([apTable, pd]), reason="Missing panda or astropy.table")
@pytest.mark.parametrize(
    "tType",
    [types.AP_TABLE, types.NUMPY_DICT, types.NUMPY_RECARRAY, types.PA_TABLE, types.PD_DATAFRAME],
)
def test_concat(data_tables, tType):
    """Perform type conversion on the cross-product of all types."""    
    odict_1 = convert(data_tables, tType)
    odict_2 = concat([odict_1, odict_1], tType)
    concat_data = odict_2['data']
    concat_md = odict_2['md']
    check_data_1 = sliceObj(concat_data, slice(0, 50))
    check_data_2 = sliceObj(concat_data, slice(1000, 1050))
    t_dict_1 = convert(check_data_1, types.AP_TABLE)
    t_dict_2 = convert(check_data_2, types.AP_TABLE)    
    test_dict = sliceObj(data_tables['data'], slice(0, 50))
    test_dict2 = sliceObjs(data_tables, slice(0, 10))
    assert compare_tables(test_dict, t_dict_1)
    assert compare_tables(test_dict, t_dict_2)
    assert compare_tables(test_dict2['data'], data_tables['data'][0:10])
