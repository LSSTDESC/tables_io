"""Array-related utility functions for tables_io"""

from collections import OrderedDict
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


def printDictShape(in_dict):
    """Print the shape of arrays in a dictionary.
    This is useful for debugging `astropy.Table` creation.

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to print
    """
    for key, val in in_dict.items():
        print(key, np.shape(val))


def sliceDict(in_dict, subslice):
    """Create a new `dict` by taking a slice of of every array in a `dict`

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to extract from
    subslice : `int` or `slice`
        Used to slice the arrays

    Returns
    -------
    out_dict : `dict`
        The converted dicionary
    """

    out_dict = OrderedDict()
    for key, val in in_dict.items():
        try:
            out_dict[key] = val[subslice]
        except (KeyError, TypeError):  #pragma: no cover
            out_dict[key] = val
    return out_dict


def checkKeys(in_dicts):
    """Check that the keys in all the in_dicts match

    Parameters
    ----------
    in_dicts : `list`, (`OrderedDict`, (`str`, `numpy.array`))
        The dictionaries for which compare keys

    Raises KeyError if one does not match.
    """
    if not in_dicts:  #pragma: no cover
        return
    master_keys = in_dicts[0].keys()
    for in_dict in in_dicts[1:]:
        if in_dict.keys() != master_keys:  #pragma: no cover
            raise ValueError("Keys do not match: %s != %s" % (in_dict.keys(), master_keys))


def concatenateDicts(in_dicts):
    """Create a new `dict` by concatenating each array in `in_dicts`

    Parameters
    ----------
    in_dicts : `list`, (`OrderedDict`, (`str`, `numpy.array`))
        The dictionaries to stack

    Returns
    -------
    out_dict : `dict`
        The stacked dicionary
    """
    if not in_dicts:  #pragma: no cover
        return OrderedDict()
    checkKeys(in_dicts)
    out_dict = OrderedDict([(key, None) for key in in_dicts[0].keys()])
    for key in out_dict.keys():
        out_dict[key] = np.concatenate([in_dict[key] for in_dict in in_dicts])
    return out_dict
