""" Unit tests for the fileIO module """

import numpy as np
import sys
import pytest
import unittest
from tables_io import arrayUtils, types
from tables_io.testUtils import check_deps
from tables_io.lazy_modules import lazyImport, apTable, pd


def test_array_length():
    """ Test the pandas reading """
    assert arrayUtils.arrayLength(4) == 0
    assert arrayUtils.arrayLength(np.ones(5)) == 5
    assert arrayUtils.arrayLength(np.ones((5, 5, 5))) == 5
    assert arrayUtils.arrayLength([3, 4, 4]) == 3

def test_force_to_pandasable():
    """ Test the force_to_pandasable function """
    rv = np.random.uniform(size=100)
    assert arrayUtils.forceToPandables(4) == 4
    assert np.allclose(arrayUtils.forceToPandables(rv), rv)
    try:
        arrayUtils.forceToPandables(rv, 95)
    except ValueError:
        pass
    else:
        raise ValueError("Failed to catch array length mismatch in arrayUtils.forceToPandables")
    rv2d = rv.reshape(10, 10)
    rv2d_check = np.vstack(arrayUtils.forceToPandables(rv2d))
    assert np.allclose(rv2d, rv2d_check)

    rv3d = rv.reshape(10, 2, 5)
    rv3d_check = np.vstack(arrayUtils.forceToPandables(rv3d)).reshape(rv3d.shape)
    assert np.allclose(rv3d, rv3d_check)


def test_slice_dict():
    """ Test the slice_dict method """
    test_data = dict(scalar=np.random.uniform(size=10),
                     vector=np.random.uniform(size=100).reshape(10, 10),
                     mat=np.random.uniform(size=1000).reshape(10, 10, 10))
    sliced = arrayUtils.sliceDict(test_data, 1)
    assert np.allclose(sliced['scalar'], test_data['scalar'][1])
    assert np.allclose(sliced['vector'], test_data['vector'][1])
    assert np.allclose(sliced['mat'], test_data['mat'][1])

    mask = np.zeros((10), bool)
    mask[1] = True
    mask[4] = True
    sliced = arrayUtils.sliceDict(test_data, mask)
    assert np.allclose(sliced['scalar'], test_data['scalar'][mask])
    assert np.allclose(sliced['vector'], test_data['vector'][mask])
    assert np.allclose(sliced['mat'], test_data['mat'][mask])


def test_print_dict_shape():
    """ Test the print_dict_shape method """
    test_data = dict(scalar=np.random.uniform(size=10),
                     vector=np.random.uniform(size=100).reshape(10, 10),
                     mat=np.random.uniform(size=1000).reshape(10, 10, 10))
    arrayUtils.printDictShape(test_data)


def test_concatenateDicts():
    """ Test the print_dict_shape method """
    test_data = dict(scalar=np.random.uniform(size=10),
                     vector=np.random.uniform(size=100).reshape(10, 10),
                     mat=np.random.uniform(size=1000).reshape(10, 10, 10))
    od = arrayUtils.concatenateDicts([test_data, test_data])
    assert np.allclose(od['scalar'][0:10], test_data['scalar'])
    assert np.allclose(od['vector'][0:10], test_data['vector'])
    assert np.allclose(od['mat'][0:10], test_data['mat'])
    assert np.allclose(od['scalar'][10:], test_data['scalar'])
    assert np.allclose(od['vector'][10:], test_data['vector'])
    assert np.allclose(od['mat'][10:], test_data['mat'])


def test_getInit():

    test_data = dict(scalar=np.random.uniform(size=10),
                     vector=np.random.uniform(size=100).reshape(10, 10),
                     mat=np.random.uniform(size=1000).reshape(10, 10, 10))

    dd = arrayUtils.getInitializationForODict(test_data)
    assert dd['scalar'][0] == (10,)
    assert dd['vector'][0] == (10, 10)
    assert dd['mat'][0] == (10, 10, 10)
    dd = arrayUtils.getInitializationForODict(test_data, 12)
    assert dd['scalar'][0] == (12,)
    assert dd['vector'][0] == (12, 10)
    assert dd['mat'][0] == (12, 10, 10)
    

def test_types():
    """ Test the typing functions"""
    assert not types.istablelike(4)
    assert types.istablelike(dict(a=np.ones(4)))
    assert not types.istablelike(dict(a=np.ones(4), b=np.ones(5)))
    assert not types.istabledictlike(4)
    assert not types.istabledictlike(dict(a=np.ones(4)))
    assert types.istabledictlike(dict(data=dict(a=np.ones(4))))
    try:
        types.fileType('xx.out')
    except KeyError:
        pass
    else:
        raise KeyError("Failed to catch unknown fileType")


@pytest.mark.skipif(not check_deps([apTable, pd]), reason="Missing an IO package")
def test_type_finders():
    """ Test the utils that identify apTables and data frames """
    import pandas as pd
    from astropy.table import Table

    class DataFrameSub(pd.DataFrame):
        pass

    class TableSub(Table):
        pass

    d1 = pd.DataFrame()
    d2 = DataFrameSub()
    t1 = Table()
    t2 = TableSub()

    assert types.isDataFrame(d1)
    assert types.isDataFrame(d2)
    assert not types.isDataFrame(t1)
    assert not types.isDataFrame(t2)
    assert not types.isDataFrame({})
    assert not types.isDataFrame(77)

    assert not types.isApTable(d1)
    assert not types.isApTable(d2)
    assert types.isApTable(t1)
    assert types.isApTable(t2)
    assert not types.isApTable({})
    assert not types.isApTable(77)




def test_lazy_load():
    """ Test that the lazy import works"""
    noModule = lazyImport('thisModuleDoesnotExist')
    try:
        noModule.d
    except ImportError:
        pass
    else:
        raise ImportError("lazyImport failed")

@unittest.skipIf("wave" in sys.modules, "Wave module already imported")
def test_lazy_load2():
    """ A second test that the lazy import works"""
    # I picked an obscure python module that is unlikely
    # to be loaded by anything else.

    wave = lazyImport("wave")
    # should not be loaded yet
    assert "wave" not in sys.modules

    # should trigger load
    assert "audioop" in dir(wave)
    assert wave.sys == sys

    assert "wave" in sys.modules
