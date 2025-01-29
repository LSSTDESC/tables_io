"""Type definitions for tables_io"""

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Union

import numpy as np

from .utils.array_utils import array_length
from .lazy_modules import pa

# Tabular data formats
AP_TABLE = 0
NUMPY_DICT = 1
NUMPY_RECARRAY = 2
PD_DATAFRAME = 3
PA_TABLE = 4

TABULAR_FORMAT_NAMES = OrderedDict(
    [
        ("astropyTable", AP_TABLE),
        ("numpyDict", NUMPY_DICT),
        ("numpyRecarray", NUMPY_RECARRAY),
        ("pandasDataFrame", PD_DATAFRAME),
        ("pyarrowTable", PA_TABLE),
    ]
)

TABULAR_FORMATS = OrderedDict([(val, key) for key, val in TABULAR_FORMAT_NAMES.items()])


# File Formats
ASTROPY_FITS = 0
ASTROPY_HDF5 = 1
NUMPY_HDF5 = 2
NUMPY_FITS = 3
PANDAS_HDF5 = 4
PANDAS_PARQUET = 5
PYARROW_HDF5 = 6
PYARROW_PARQUET = 7


FILE_FORMAT_NAMES = OrderedDict(
    [
        ("astropyFits", ASTROPY_FITS),
        ("astropyHdf5", ASTROPY_HDF5),
        ("numpyHdf5", NUMPY_HDF5),
        ("numpyFits", NUMPY_FITS),
        ("pyarrowHdf5", PYARROW_HDF5),
        ("pandasHdf5", PANDAS_HDF5),
        ("pandaParquet", PANDAS_PARQUET),
        ("pyarrowParquet", PYARROW_PARQUET),
    ]
)

# Default suffixes for various file formats
FILE_FORMAT_SUFFIXS = OrderedDict(
    [
        ("fits", ASTROPY_FITS),
        ("hf5", ASTROPY_HDF5),
        ("hdf5", NUMPY_HDF5),
        ("fit", NUMPY_FITS),
        ("h5", PANDAS_HDF5),
        ("hd5", PYARROW_HDF5),
        ("parquet", PYARROW_PARQUET),
        ("parq", PANDAS_PARQUET),
        ("pq", PANDAS_PARQUET),
    ]
)

DEFAULT_TABLE_KEY = OrderedDict(
    [
        ("fits", ""),
        ("hf5", None),
        ("hdf5", None),
        ("hd5", "data"),
        ("fit", ""),
        ("h5", "data"),
        ("parquet", ""),
        ("parq", ""),
        ("pq", ""),
    ]
)

FILE_FORMATS = OrderedDict([(val, key) for key, val in FILE_FORMAT_NAMES.items()])

FILE_FORMAT_SUFFIX_MAP = OrderedDict(
    [(val, key) for key, val in FILE_FORMAT_SUFFIXS.items()]
)

# Default format to write various table types
NATIVE_FORMAT = OrderedDict(
    [
        (AP_TABLE, ASTROPY_HDF5),
        (NUMPY_DICT, NUMPY_HDF5),
        (NUMPY_RECARRAY, NUMPY_FITS),
        (PD_DATAFRAME, PANDAS_PARQUET),
        (PA_TABLE, PYARROW_PARQUET),
    ]
)

NATIVE_TABLE_TYPE = OrderedDict([(val, key) for key, val in NATIVE_FORMAT.items()])

# Allowed formats to write various table types
ALLOWED_FORMATS = OrderedDict(
    [
        (AP_TABLE, [ASTROPY_FITS, ASTROPY_HDF5]),
        (NUMPY_DICT, [NUMPY_HDF5]),
        (NUMPY_RECARRAY, [ASTROPY_FITS]),
        (PD_DATAFRAME, [PANDAS_PARQUET, PANDAS_HDF5]),
        (PA_TABLE, [PYARROW_PARQUET, PANDAS_PARQUET, PANDAS_HDF5]),
    ]
)


def is_dataframe(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "DataFrame" and c.__module__ == "pandas.core.frame":
            return True
    return False


def is_ap_table(obj):
    for c in obj.__class__.__mro__:
        if c.__name__ == "Table" and c.__module__ == "astropy.table.table":
            return True
    return False


def table_type(obj):
    """Identify the type of table we have

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
    if is_dataframe(obj):
        return PD_DATAFRAME
    if is_ap_table(obj):
        return AP_TABLE
    if isinstance(obj, pa.Table):
        return PA_TABLE
    if isinstance(obj, (np.recarray, np.ma.core.MaskedArray)):
        return NUMPY_RECARRAY
    if not isinstance(obj, Mapping):
        raise TypeError(
            f"Object of type {type(obj)} is not one of the supported types"
            f"Must be one of {list(TABULAR_FORMAT_NAMES.keys())}"
        )

    nRow = None
    for key, val in obj.items():
        if is_table_like(val):
            raise TypeError(f"Column {key} is a table of type {type(val)}")
        if not isinstance(val, Iterable):  # pragma: no cover
            raise TypeError(f"Column {key} of type {type(val)} is not iterable")
        if nRow is None:
            nRow = array_length(val)
        else:
            if array_length(val) != nRow:
                raise IndexError(
                    f"Column {key} length {array_length(val)} != {nRow}"
                )  # pylint: disable=bad-string-format-type
    return NUMPY_DICT


def is_table_like(obj):
    """Test to see if an object is one of the supported table types

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    table-like : `bool`
        True is the object is `Table-like`, False otherwise
    """
    try:
        _ = table_type(obj)
    except (TypeError, IndexError):
        return False
    return True


def is_tabledict_like(obj):
    """Test to see if an object is a `Mapping`, (`str`, `Table-like`),
    or `TableDict-like`.

    Parameters
    ----------
    obj : `object`
        The input object

    Returns
    -------
    tabledict : `bool`
        True is the object is a `Mapping`, (`str`, `Table-like`), False otherwise
    """
    if not isinstance(obj, Mapping):
        return False
    for val in obj.values():
        if not is_table_like(val):
            return False
    return True


def file_type(filepath, fmt=None):
    """Identify the type of file we have

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
        raise KeyError(
            f"Unknown file format {fmt}, supported types are"
            f"{list(FILE_FORMAT_SUFFIXS.keys())}"
        ) from msg


def tType_to_int(tType: Union[str, int]) -> int:
    """Takes table type as an `int` or `str`, and converts it to the corresponding
      `int` if it's a `str`.

    Parameters
    ----------
    tType : Union[str, int]
        The tabular format

    Returns
    -------
    int
        The number corresponding to the tabular format

    Raises
    ------
    TypeError
        Raised if the given `str` is not one of the available tabular format options.
    """

    if isinstance(tType, str):
        try:
            int_tType = TABULAR_FORMAT_NAMES[tType]
        except:
            raise TypeError(
                f"Unsupported tableType '{tType}', must be one of {TABULAR_FORMAT_NAMES}"
            )
    if isinstance(tType, int):
        int_tType = tType

    return int_tType
