"""IO Read Functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

from ..utils.arrayUtils import getGroupInputDataLength, forceToPandables
from ..convert.convMulti import convert
from ..convert.convSingle import dataFrameToDict, hdf5GroupToDict
from ..lazy_modules import apTable, fits, h5py, pa, pd, pq, ds
from ..types import (
    AP_TABLE,
    ASTROPY_FITS,
    ASTROPY_HDF5,
    DEFAULT_TABLE_KEY,
    FILE_FORMAT_SUFFIX_MAP,
    FILE_FORMAT_SUFFIXS,
    NATIVE_FORMAT,
    NATIVE_TABLE_TYPE,
    NUMPY_FITS,
    NUMPY_HDF5,
    PA_TABLE,
    PANDAS_HDF5,
    PANDAS_PARQUET,
    PYARROW_HDF5,
    PYARROW_PARQUET,
    PD_DATAFRAME,
    fileType,
    istabledictlike,
    istablelike,
    tableType,
)


# I. Top-level interface functions


def readSingle():
    # reads in a single table
    # gives an error if there are multiple tables?
    pass


def readMulti():
    # reads in multiple tables to an ordered dict
    # if there's only one table, reads it into an ordered dict
    pass


def read(filepath, tType=None, fmt=None, keys=None, allow_missing_keys=False, **kwargs):
    """Read a file to the corresponding table type

    Parameters
    ----------
    filepath : `str`
        File to load
    tType : `int` or `None`
        Table type, if `None` this will use `readNative`
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension
    keys : `list` or `None`
        For parquet files we must specify which keys to read, as each is in its own file
    allow_missing_keys : `bool`
        If False will raise FileNotFoundError if a key is missing
    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    data : `OrderedDict` ( `str` -> `Tablelike` )
        The data

    """
    odict = readNative(
        filepath, fmt, keys, allow_missing_keys, **kwargs
    )  # TODO: put a try except here or in each of the individual read functions
    if len(odict) == 1:
        # For special keys, use the table alone without an enclosing dictionary.
        single_dict_key = list(odict.keys())[0]
        if single_dict_key in ["", None, "__astropy_table__", "data"]:
            odict = odict[single_dict_key]
    if tType is None:  # pragma: no cover
        return odict
    return convert(odict, tType)


def readNative(filepath, fmt=None, keys=None, allow_missing_keys=False, **kwargs):
    """Read a file to the corresponding table type

    Parameters
    ----------
    filepath : `str`
        File to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension
    keys : `list` or `None`
        For parquet files we must specify with keys to read, as each is in its own file
    allow_missing_keys : `bool`
        If False will raise FileNotFoundError if a key is missing
    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    data : `OrderedDict` ( `str` -> `Tablelike` )
        The data

    """
    fType = fileType(filepath, fmt)
    if fType == ASTROPY_FITS:
        return readFitsToApTables(filepath, keys=keys)
    if fType == ASTROPY_HDF5:
        return readHdf5ToApTables(filepath, keys=keys)
    if fType == NUMPY_HDF5:
        return readHdf5ToDicts(filepath, keys=keys)
    if fType == NUMPY_FITS:
        return readFitsToRecarrays(filepath, keys=keys)
    if fType == PANDAS_HDF5:
        return readH5ToDataFrames(filepath, keys=keys)
    if fType == PANDAS_PARQUET:
        return readPqToDataFrames(filepath, keys, allow_missing_keys, **kwargs)
    if fType == PYARROW_HDF5:
        return readHd5ToTables(filepath, keys)
    if fType == PYARROW_PARQUET:
        return readPqToTables(filepath, keys, allow_missing_keys, **kwargs)
    raise TypeError(f"Unsupported FileType {fType}")  # pragma: no cover


def io_open(filepath, fmt=None, **kwargs):
    """Open a file

    Parameters
    ----------
    filepath : `str`
        File to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension

    Returns
    -------
    file
    """
    fType = fileType(filepath, fmt)
    if fType in [ASTROPY_FITS, NUMPY_FITS]:
        return fits.open(filepath, **kwargs)
    if fType in [ASTROPY_HDF5, NUMPY_HDF5, PANDAS_HDF5, PYARROW_HDF5]:
        return h5py.File(filepath, **kwargs)
    if fType in [PYARROW_PARQUET, PANDAS_PARQUET]:
        # basepath = os.path.splitext(filepath)[0]
        return pq.ParquetFile(filepath, **kwargs)
    raise TypeError(f"Unsupported FileType {fType}")  # pragma: no cover


def check_columns(
    filepath, columns_to_check, fmt=None, parent_groupname=None, **kwargs
):
    """Read the file column names and check it against input list

    Parameters
    ----------
    filepath : `str`
        File name for the file to read. If there's no suffix, it will be applied based on the object type.
    columns_to_check: `list`
        A list of columns to be compared with the data
    fmt : `str` or `None`
        The output file format, If `None` this will use `writeNative`
    parent_groupname: `str` or `None`
        For hdf5 files, the groupname for the data
    """

    fType = fileType(filepath, fmt)

    # Read the file below:
    file = io_open(filepath, fmt=None, **kwargs)

    if fType in [ASTROPY_FITS, NUMPY_FITS]:
        col_list = []
        for hdu in file[1:]:
            columns = hdu.columns
            for col in columns:
                if col.name not in col_list:
                    col_list.append(col.name)

    elif fType in [ASTROPY_HDF5, NUMPY_HDF5, PANDAS_HDF5, PYARROW_HDF5]:
        col_list = readHdf5GroupNames(filepath, parent_groupname=parent_groupname)

    elif fType in [PYARROW_PARQUET, PANDAS_PARQUET]:
        col_list = file.schema.names
    else:
        raise TypeError(f"Unsupported FileType {fType}")  # pragma: no cover

    # check columns
    intersection = set(columns_to_check).intersection(col_list)
    if len(intersection) < len(columns_to_check):
        diff = set(columns_to_check) - intersection
        raise KeyError("The following columns are not found: ", diff)


# II. Reading Files


# II A. Reading `astropy.table.Table` from FITS files


def readFitsToApTables(filepath, keys=None):
    """
    Reads `astropy.table.Table` objects from a FITS file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables
    """
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        if keys is not None:
            if hdu.name.lower() not in keys:
                continue
        tables[hdu.name.lower()] = apTable.Table.read(filepath, hdu=hdu.name)
    return tables


# II B Reading `np.recarray` from FITS files


def readFitsToRecarrays(filepath, keys=None):
    """
    Reads `np.recarray` objects from a FITS file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    tables : `OrderedDict` of `np.recarray`
        Keys will be HDU names, values will be tables
    """
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        if keys is not None and hdu.name.lower() not in keys:
            continue
        tables[hdu.name.lower()] = hdu.data
    return tables


# II C Reading `astropy.table.Table` from HDF5 file


def readHdf5ToApTables(filepath, keys=None):
    """
    Reads `astropy.table.Table` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be 'paths', values will be tables
    """
    fin = h5py.File(filepath)
    tables = OrderedDict()
    for k in fin.keys():
        if keys is not None and k not in keys:
            continue
        tables[k] = apTable.Table.read(filepath, path=k, format="hdf5")
    return tables


## II D. Reading `OrderedDict` (`str`, `numpy.array`) and `np.array` from HDF5 file


def readHdf5Group(filepath, groupname=None):
    """Read and return group from an hdf5 file.

    Parameters
    ----------
    filepath : `str`
        File in question
    groupname : `str` or `None`
        For hdf5 files, the groupname for the data

    Returns
    -------
    grp : `h5py.Group` or `h5py.File`
        The requested group
    infp : `h5py.File`
        The input file (returned so that the used can explicitly close the file)
    """
    infp = h5py.File(filepath, "r")
    if groupname is None or not groupname:  # pragma: no cover
        return infp, infp
    return infp[groupname], infp


def readHdf5GroupToDict(hg, start=None, end=None):
    """
    Reads `numpy.array` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `numpy.array`
        Keys will be 'paths', values will be tables
    """
    # pylint: disable=unused-argument
    if isinstance(hg, h5py.Dataset):
        return readHdf5DatasetToArray(hg, start, end)
    return OrderedDict(
        [(key, readHdf5DatasetToArray(val, start, end)) for key, val in hg.items()]
    )


def readHdf5GroupNames(filepath, parent_groupname=None):
    """Read and return group from an hdf5 file.

    Parameters
    ----------
    filepath : `str`
        File in question
    parent_groupname : `str` or `None`
        For hdf5 files, the parent groupname. All group names under this will be
        returned. If `None`, return the top level group names.

    Returns
    -------
    names : `list` of `str`
        The names of the groups in the file
    """
    infp = h5py.File(filepath, "r")
    if parent_groupname is None:  # pragma: no cover
        return list(infp.keys())

    try:
        subgroups = infp[parent_groupname].keys()
    except KeyError as msg:
        raise KeyError(
            f"Group {parent_groupname} not found in file {filepath}"
        ) from msg
    return list(subgroups)


def readHdf5ToDicts(filepath, keys=None):
    """
    Reads `numpy.array` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    dicts : `OrderedDict`, (`str`, `OrderedDict`, (`str`, `numpy.array`) )
        The data
    """
    fin = h5py.File(filepath)
    l_out = []
    for key, val in fin.items():
        if keys is not None and key not in keys:
            continue
        l_out.append((key, readHdf5GroupToDict(val)))
    return OrderedDict(l_out)


def readHdf5DatasetToArray(dataset, start=None, end=None):
    """Reads part of a hdf5 dataset into a `numpy.array`

    Parameters
    ----------
    dataset : `h5py.Dataset`
        The input dataset

    start : `int` or `None`
        Starting row

    end : `int` or `None`
        Ending row

    Returns
    -------
    out : `numpy.array` or `list` of `numpy.array`
        Something that pandas can handle
    """
    if start is None or end is None:
        return np.array(dataset)
    return np.array(dataset[start:end])


# II D. Reading `pandas.DataFrame` from HDF5


def readHdf5ToDataFrame(filepath, key=None):
    """
    Reads `pandas.DataFrame` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file
    key : `str` or `None`
        The key in the hdf5 file

    Returns
    -------
    df : `pandas.DataFrame`
        The dataframe
    """
    return pd.read_hdf(filepath, key)


def readH5ToDataFrames(filepath, keys=None):
    """Open an h5 file and and return a dictionary of `pandas.DataFrame`

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    tab : `OrderedDict` (`str` : `pandas.DataFrame`)
       The data

    Notes
    -----
    We are using the file suffix 'h5' to specify 'hdf5' files written from DataFrames using `pandas`
    They have a different structure than 'hdf5' files written with `h5py` or `astropy.table`
    """
    fin = h5py.File(filepath)
    l_out = []
    for key in fin.keys():
        if keys is not None and key not in keys:
            continue
        l_out.append((key, readHdf5ToDataFrame(filepath, key=key)))
    return OrderedDict(l_out)


# II E Reading `pandas.DataFrame` from parquet file


def readPqToDataFrame(filepath, columns=None, **kwargs):
    """
    Reads a `pandas.DataFrame` object from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns
    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    df : `pandas.DataFrame`
        The data frame
    """
    return pd.read_parquet(filepath, engine="pyarrow", columns=columns, **kwargs)


def readPqToDataFrames(
    filepath, keys=None, allow_missing_keys=False, columns=None, **kwargs
):
    """
    Reads `pandas.DataFrame` objects from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    allow_missing_keys: `bool`
        If False will raise FileNotFoundError if a key is missing

    columns : `dict` of `list (str)`, `list` (`str`), or `None`
        Names of the columns to read.
            - if a dictionary, keys are the `keys`, and values are a list of string column names.
                for each keyed table, only the columns in the value list will be loaded.
                if the key is not found, all columns will be loaded.
            - if a list, only the columns in the list will be loaded.
            - `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tables : `OrderedDict` of `pandas.DataFrame`
        Keys will be taken from keys
    """
    if keys is None:  # pragma: no cover
        keys = [""]
    dataframes = OrderedDict()
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for key in keys:
        try:
            column_list = None
            if pd.api.types.is_dict_like(columns):
                column_list = columns[key]
            elif pd.api.types.is_list_like(columns):
                column_list = columns
            print("column_list", column_list)

            dataframes[key] = readPqToDataFrame(
                f"{basepath}{key}{ext}", columns=column_list, **kwargs
            )
        except FileNotFoundError as msg:  # pragma: no cover
            if allow_missing_keys:
                continue
            raise msg
    return dataframes


# II F. Reading `OrderedDict` (`str`, `numpy.array`) from parquet file


def readPqToDict(filepath, columns=None, **kwargs):
    """Open a parquet file and return a dictionary of `numpy.array`

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns
    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data
    """
    tab = pq.read_table(filepath, columns=columns, **kwargs)
    return OrderedDict(
        [
            (c_name, col.to_numpy())
            for c_name, col in zip(tab.column_names, tab.itercolumns())
        ]
    )


def readH5ToDict(filepath, groupname=None):
    """Open an h5 file and and return a dictionary of `numpy.array`

    Parameters
    ----------
    filepath: `str`
        Path to input file

    groupname : `str` or `None`
        The group with the data

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data

    Notes
    -----
    We are using the file suffix 'h5' to specify 'hdf5' files written from DataFrames using `pandas`
    They have a different structure than 'hdf5' files written with `h5py` or `astropy.table`
    """
    df = readHdf5ToDataFrame(filepath, groupname)
    return dataFrameToDict(df)


def readHdf5ToDict(filepath, groupname=None):
    """Read in h5py hdf5 data, return a dictionary of all of the keys

    Parameters
    ----------
    filepath: `str`
        Path to input file

    groupname : `str` or `None`
        The groupname for the data

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data

    Notes
    -----
    We are using the file suffix 'hdf5' to specify 'hdf5' files written with `h5py` or `astropy.table`
    They have a different structure than 'h5' files written `panda`
    """
    hg, infp = readHdf5Group(filepath, groupname)
    data = hdf5GroupToDict(hg)
    infp.close()
    return data


# II G. Reading `pyarrow.Table` from HDF5 file


def readHd5ToTable(filepath, key=None):
    """
    Reads `pyarrow.Table` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file
    key : `str` or `None`
        The key in the hdf5 file

    Returns
    -------
    table : `pyarrow.Table`
        The table
    """
    pydict = readHdf5ToDicts(filepath, [key])[key]
    t_dict = {}
    for key, val in pydict.items():
        t_dict[key] = forceToPandables(
            val
        )  # TODO: add a try except around this to raise a more understandable error?
    return pa.Table.from_pydict(t_dict)


def readHd5ToTables(filepath, keys=None):
    """Open an h5 file and and return a dictionary of `pyarrow.Table`

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list` or `None`
        Which tables to read

    Returns
    -------
    tab : `OrderedDict` (`str` : `pyarrow.Table`)
       The data

    """
    fin = h5py.File(filepath)
    l_out = []
    for key in fin.keys():
        if keys is not None and key not in keys:  # pragma: no cover
            continue
        l_out.append((key, readHd5ToTable(filepath, key=key)))
    return OrderedDict(l_out)


# II H. Reading `pyarrow.Table` from parquet file


def readPqToTable(filepath, **kwargs):
    """
    Reads a `pyarrow.Table` object from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns
    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    table : `pyarrow.Table`
        The table
    """
    return pq.read_table(filepath, **kwargs)


def readPqToTables(
    filepath, keys=None, allow_missing_keys=False, columns=None, **kwargs
):
    """
    Reads `pyarrow.Table` objects from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    allow_missing_keys: `bool`
        If False will raise FileNotFoundError if a key is missing

    columns : `dict` of `list (str)`, `list` (`str`), or `None`
        Names of the columns to read.
            - if a dictionary, keys are the `keys`, and values are a list of string column names.
                for each keyed table, only the columns in the value list will be loaded.
                if the key is not found, all columns will be loaded.
            - if a list, only the columns in the list will be loaded.
            - `None` will read all the columns

    **kwargs : additional arguments to pass to the native file reader

    Returns
    -------
    tables : `OrderedDict` of `pyarrow.Table`
        Keys will be taken from keys
    """
    if keys is None:  # pragma: no cover
        keys = [""]
    tables = OrderedDict()
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for key in keys:
        try:
            column_list = None
            if pd.api.types.is_dict_like(columns):  # pragma: no cover
                column_list = columns[key]
            elif pd.api.types.is_list_like(columns):  # pragma: no cover
                column_list = columns
            print("column_list", column_list)

            tables[key] = readPqToTable(
                f"{basepath}{key}{ext}", columns=column_list, **kwargs
            )
        except FileNotFoundError as msg:  # pragma: no cover
            if allow_missing_keys:
                continue
            raise msg
    return tables
