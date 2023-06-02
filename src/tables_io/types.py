"""Type defintions for tables_io"""

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import numpy as np

from .arrayUtils import arrayLength

# Tabular data formats
AP_TABLE = 0
NUMPY_DICT = 1
NUMPY_RECARRAY = 2
PD_DATAFRAME = 3

TABULAR_FORMAT_NAMES = OrderedDict([
    ('astropyTable', AP_TABLE),
    ('numpyDict', NUMPY_DICT),
    ('numpyRecarray', NUMPY_RECARRAY),
    ('pandasDataFrame', PD_DATAFRAME)])

TABULAR_FORMATS = OrderedDict([(val, key) for key, val in TABULAR_FORMAT_NAMES.items()])


# File Formats
ASTROPY_FITS = 0
ASTROPY_HDF5 = 1
NUMPY_HDF5 = 2
NUMPY_FITS = 3
PANDAS_HDF5 = 4
PANDAS_PARQUET = 5

FILE_FORMAT_NAMES = OrderedDict([
    ('astropyFits', ASTROPY_FITS),
    ('astropyHdf5', ASTROPY_HDF5),
    ('numpyHdf5', NUMPY_HDF5),
    ('numpyFits', NUMPY_FITS),
    ('pandasHdf5', PANDAS_HDF5),
    ('pandaParquet', PANDAS_PARQUET)])

# Default suffixes for various file formats
FILE_FORMAT_SUFFIXS = OrderedDict([
    ('fits', ASTROPY_FITS),
    ('hf5', ASTROPY_HDF5),
    ('hdf5', NUMPY_HDF5),
    ('fit', NUMPY_FITS),
    ('h5', PANDAS_HDF5),
    ('parquet', PANDAS_PARQUET),
    ('parq', PANDAS_PARQUET),
    ('pq', PANDAS_PARQUET)])

DEFAULT_TABLE_KEY = OrderedDict([
    ('fits', ''),
    ('hf5', None),
    ('hdf5', None),
    ('fit', ''),
    ('h5', 'data'),
    ('parquet', ''),
    ('parq', ''),
    ('pq', '')])

FILE_FORMATS = OrderedDict([(val, key) for key, val in FILE_FORMAT_NAMES.items()])

FILE_FORMAT_SUFFIX_MAP = OrderedDict([(val, key) for key, val in FILE_FORMAT_SUFFIXS.items()])

# Default format to write various table types
NATIVE_FORMAT = OrderedDict([
    (AP_TABLE, ASTROPY_HDF5),
    (NUMPY_DICT, NUMPY_HDF5),
    (NUMPY_RECARRAY, NUMPY_FITS),
    (PD_DATAFRAME, PANDAS_PARQUET)])

NATIVE_TABLE_TYPE = OrderedDict([(val, key) for key, val in NATIVE_FORMAT.items()])

# Allowed formats to write various table types
ALLOWED_FORMATS = OrderedDict([
    (AP_TABLE, [ASTROPY_FITS, ASTROPY_HDF5]),
    (NUMPY_DICT, [NUMPY_HDF5]),
    (NUMPY_RECARRAY, [ASTROPY_FITS]),
    (PD_DATAFRAME, [PANDAS_PARQUET, PANDAS_HDF5])])


def isDataFrame(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "DataFrame" and c.__module__ == "pandas.core.frame":
            return True
    return False

def isApTable(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "Table" and c.__module__ == "astropy.table.table":
            return True
    return False



def tableType(obj):
    """ Identify the type of table we have

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    otype : `int`
        The object type, one of `TABULAR_FORMATS.keys()`

    Raises
    ------
    TypeError
        The object is not a supported type
    IndexError
        One of the columns in a Mapping is the wrong length
    """
    if isDataFrame(obj):
        return PD_DATAFRAME
    if isApTable(obj):
        return AP_TABLE
    if isinstance(obj, np.recarray):
        return NUMPY_RECARRAY
    if not isinstance(obj, Mapping):
        raise TypeError(f"Object of type {type(obj)} is not one of the supported types"
                        f"{list(TABULAR_FORMAT_NAMES.keys())}")

    nRow = None
    for key, val in obj.items():
        if istablelike(val):
            raise TypeError(f"Column {key} is a table of type {type(val)}")
        if not isinstance(val, Iterable):  #pragma: no cover
            raise TypeError(f"Column {key} of type {type(val)} is not iterable")
        if nRow is None:
            nRow = arrayLength(val)
        else:
            if arrayLength(val) != nRow:
                raise IndexError(f"Column {key} length {arrayLength(val)} != {nRow}") #pylint: disable=bad-string-format-type
    return NUMPY_DICT


def istablelike(obj):
    """ Test to see if an object is one of the supported table types

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    tablelike : `bool`
        True is the object is `Tablelike`, False otherwise
    """
    try:
        _ = tableType(obj)
    except (TypeError, IndexError):
        return False
    return True


def istabledictlike(obj):
    """ Test to see if an object is a `Mapping`, (`str`, `Tablelike`)

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    tabledict : `bool`
        True is the object is a `Mapping`, (`str`, `Tablelike`), False otherwise
    """
    if not isinstance(obj, Mapping):
        return False
    for val in obj.values():
        if not istablelike(val):
            return False
    return True


def fileType(filepath, fmt=None):
    """ Identify the type of file we have

    Parameters
    ----------
    filepath : `str`
        The path to the file

    fmt : `str` or `None`
        Overrides the file extension

    Returns
    -------
    otype : `int`
        The object type, one of `FILE_FORMATS.keys()`

    Raises
    ------
    KeyError
        The file format is not a support value
    """
    if fmt is None:
        fmt = os.path.splitext(filepath)[1][1:]
    try:
        return FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:
        raise KeyError(f"Unknown file format {fmt}, supported types are"
                       f"{list(FILE_FORMAT_SUFFIXS.keys())}") from msg
