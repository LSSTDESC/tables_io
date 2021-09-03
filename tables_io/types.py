"""Type defintions for tables_io"""

import os

from collections import OrderedDict
from collections.abc import Mapping, Iterable

import numpy as np

from .lazy_modules import apTable, pd
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
    ('pq', PANDAS_PARQUET)])

DEFAULT_TABLE_KEY = OrderedDict([
    ('fits', ''),
    ('hf5', None),
    ('hdf5', None),
    ('fit', ''),
    ('h5', 'data'),
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
    (AP_TABLE, [ASTROPY_HDF5, ASTROPY_HDF5]),
    (NUMPY_DICT, [NUMPY_HDF5]),
    (NUMPY_RECARRAY, [ASTROPY_FITS]),
    (PD_DATAFRAME, [PANDAS_PARQUET, PANDAS_HDF5])])


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
    TypeError : The object is not a supported type

    IndexError : One of the columns in a Mapping is the wrong length
    """
    if isinstance(obj, apTable.Table):
        return AP_TABLE
    if isinstance(obj, pd.DataFrame):
        return PD_DATAFRAME
    if isinstance(obj, np.recarray):
        return NUMPY_RECARRAY
    if not isinstance(obj, Mapping):
        raise TypeError("Object of type %s is not one of the supported types %s" %
                        (type(obj), list(TABULAR_FORMAT_NAMES.keys())))

    nRow = None
    for key, val in obj.items():
        if isinstance(val, (Mapping, apTable.Table, pd.DataFrame)):
            raise TypeError("Column %s of type Mapping %s" %
                            (key, type(val)))
        if isinstance(val, (np.recarray)):
            raise TypeError("Column %s of type np.recarray %s" %
                            (key, type(val)))
        if not isinstance(val, Iterable):  #pragma: no cover
            raise TypeError("Column %s of type %s is not iterable" %
                            (key, type(val)))
        if nRow is None:
            nRow = arrayLength(val)
        else:
            if arrayLength(val) != nRow:
                raise IndexError("Column %s length %i != %i" % (key, arrayLength(val), nRow)) #pylint: disable=bad-string-format-type
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
    KeyError : The file format is not a support value
    """
    if fmt is None:
        fmt = os.path.splitext(filepath)[1][1:]
    try:
        return FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:
        raise KeyError("Unknown file format %s, supported types are %s" %
                       (fmt, list(FILE_FORMAT_SUFFIXS.keys()))) from msg
