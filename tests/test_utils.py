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
