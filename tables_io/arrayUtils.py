"""Array-related utility functions for tables_io"""

import numpy as np

def arrayLength(arr):
    """ Get the length of an array

    The works on scalars and arrays, so it is safe to use

    For scalars it returns 0
    For arrays it np.shape(arr)[0]

    Parameters
    ----------
    arr : `array-like`
        The input array

    Returns
    -------
    length : `int`
        The object length
    """
    shape = np.shape(arr)
    if not shape:
        return 0
    return shape[0]


def forceToPandables(arr, check_nrow=None):
    """
    Forces a  `numpy.array` into a format that pandas can handle

    Parameters
    ----------
    arr : `numpy.array`
        The input array

    check_nrow : `int` or `None`
        If not None, require that `arr.shape[0]` match this value

    Returns
    -------
    out : `numpy.array` or `list` of `numpy.array`
        Something that pandas can handle
    """
    ndim = np.ndim(arr)
    if ndim == 0:
        return arr
    shape = np.shape(arr)
    nrow = shape[0]
    if check_nrow is not None and check_nrow != nrow:
        raise ValueError("Number of rows does not match: %i != %i" % (nrow, check_nrow))
    if ndim == 1:
        return arr
    if ndim == 2:
        return list(arr)
    shape = np.shape(arr)
    ncol = np.product(shape[1:])
    return list(arr.reshape(nrow, ncol))


def getGroupInputDataLength(hg):
    """ Return the length of a HDF5 group

    Parameters
    ----------
    hg : `h5py.Group` or `h5py.File`
        The input data group

    Returns
    -------
    length : `int`
        The length of the data

    Notes
    -----
    For a multi-D array this return the length of the first axis
    and not the total size of the array.

    Normally that is what you want to be iterating over.

    The group is meant to represent a table, hence all child datasets
    should be the same length
    """
    firstkey = list(hg.keys())[0]
    nrows = len(hg[firstkey])
    firstname = hg[firstkey].name
    for value in hg.values():
        if len(value) != nrows:
            raise ValueError(f"Group does not represent a table. Length ({len(value)}) of column {value.name} not not match length ({nrows}) of first column {firstname}")
    return nrows
