""" Unit tests for the fileIO module """

import numpy as np

from tables_io import arrayUtils, types
from tables_io.lazy_modules import lazyImport


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



def test_lazy_load():
    """ Test that the lazy import works"""
    noModule = lazyImport('thisModuleDoesnotExist')
    try:
        noModule.d
    except ImportError:
        pass
    else:
        raise ImportError("lazyImport failed")
