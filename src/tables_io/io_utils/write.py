"""Io write functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

from ..convert.conv_tabledict import convert
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


def write(obj, filepath, fmt=None):
    """Write a file or files with tables

    Parameters
    ----------
    obj : `Tablelike` or `TableDictLike`
        The data to write
    filepath : `str`
        File name for the file to write. If there's no suffix, it will be applied based on the object type.
    fmt : `str` or `None`
        The output file format, If `None` this will use `write_native`
    """
    if fmt is None:
        splitpath = os.path.splitext(filepath)
        if not splitpath[1]:
            return write_native(obj, filepath)
        fmt = splitpath[1][1:]

    try:
        fType = FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(
            f"Unknown file format {fmt} from {filepath}, options are {list(FILE_FORMAT_SUFFIXS.keys())}"
        ) from msg

    if istablelike(obj):
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], obj)])
    elif istabledictlike(obj):
        odict = obj
    elif not obj:
        return None
    else:
        raise TypeError(f"Can not write object of type {type(obj)}")

    if fType in [ASTROPY_HDF5, NUMPY_HDF5, NUMPY_FITS, PANDAS_PARQUET, PYARROW_PARQUET]:
        try:
            nativeTType = NATIVE_TABLE_TYPE[fType]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Native file type not known for {fmt}") from msg

        forcedOdict = convert(odict, nativeTType)
        if os.path.splitext(filepath)[1]:
            fullpath = filepath
        else:
            fullpath = f"{filepath}.{fmt}"

        return write_native(forcedOdict, fullpath)

    if not os.path.splitext(filepath)[1]:
        filepath = filepath + "." + fmt
    if fType == ASTROPY_FITS:
        forcedOdict = convert(odict, AP_TABLE)
        writeApTablesToFits(forcedOdict, filepath)
        return filepath
    if fType == PANDAS_HDF5:
        forcedOdict = convert(odict, PD_DATAFRAME)
        writeDataFramesToH5(forcedOdict, filepath)
        return filepath
    if fType == PYARROW_HDF5:
        forcedPaTables = convert(odict, PA_TABLE)
        writeTablesToHd5(forcedPaTables, filepath)
        return filepath

    raise TypeError(f"Unsupported File type {fType}")  # pragma: no cover


def write_native(odict, filepath):
    """Write a file or files with tables

    Parameters
    ----------
    odict : `OrderedDict`, (`str`, `Tablelike`)
        The data to write
    filepath : `str`
        File name for the file to write. If there's no suffix, it will be applied based on the object type.
    """
    istable = False
    if istablelike(odict):
        istable = True
        tType = tableType(odict)
    elif istabledictlike(odict):
        tType = tableType(list(odict.values())[0])
    elif not odict:  # pragma: no cover
        return None
    else:  # pragma: no cover
        raise TypeError(f"Can not write object of type {type(odict)}")

    try:
        fType = NATIVE_FORMAT[tType]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(f"No native format for table type {tType}") from msg
    fmt = FILE_FORMAT_SUFFIX_MAP[fType]
    if not os.path.splitext(filepath)[1]:
        filepath = filepath + "." + fmt

    if istable:
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], odict)])

    try:
        os.unlink(filepath)
    except FileNotFoundError:
        pass

    if fType == ASTROPY_HDF5:
        writeApTablesToHdf5(odict, filepath)
        return filepath
    if fType == NUMPY_HDF5:
        writeDictsToHdf5(odict, filepath)
        return filepath
    if fType == NUMPY_FITS:
        writeRecarraysToFits(odict, filepath)
        return filepath
    if fType == PANDAS_PARQUET:
        writeDataFramesToPq(odict, filepath)
        return filepath
    if fType == PYARROW_PARQUET:
        writeTablesToPq(odict, filepath)
        return filepath
    raise TypeError(f"Unsupported Native file type {fType}")  # pragma: no cover


# II. Writing Files

# II A. Writing HDF5 files


def initializeHdf5WriteSingle(filepath, groupname=None, comm=None, **kwds):
    """Prepares an hdf5 file for output

    Parameters
    ----------
    filepath : `str`
        The output file name
    groupname : `str` or `None`
        The output group name

    Returns
    -------
    group : `h5py.File` or `h5py.Group`
        The group to write to
    fout : `h5py.File`
        The output file

    Notes
    -----
    The keywords should be used to create_datasets within the hdf5 file.
    Each keyword should provide a tuple of ( (shape), (dtype) )

    shape : `tuple` ( `int` )
        The shape of the data for this dataset
    dtype : `str`
        The data type for this dataset

    For example
    ``initializeHdf5WriteSingle('test.hdf5', data = dict(scalar=((100000,), 'f4'), vect=((100000, 3), 'f4'))``
    Would initialize an hdf5 file with two datasets, with shapes and data types as given

    """
    outdir = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(outdir):  # pragma: no cover
        os.makedirs(outdir, exist_ok=True)
    if comm is None:
        outf = h5py.File(filepath, "w")
    else:
        if not h5py.get_config().mpi:
            raise TypeError(
                "hdf5py module not prepared for parallel writing."
            )  # pragma: no cover
        outf = h5py.File(filepath, "w", driver="mpio", comm=comm)
    if groupname is None:
        group = outf
    else:
        group = outf.create_group(groupname)
    for key, shape in kwds.items():
        group.create_dataset(key, shape[0], shape[1])
    return group, outf


def initializeHdf5Write(filepath, comm=None, **kwds):
    """Prepares an hdf5 file for output

    Parameters
    ----------
    filepath : `str`
        The output file name
    comm: `communicator`
        MPI communicator to do parallel writing

    Returns
    -------
    group : `h5py.File` or `h5py.Group`
        The group to write to
    fout : `h5py.File`
        The output file

    Notes
    -----
    The keywords should be used to create groups within the hdf5 file.
    Each keyword should provide a dictionary with the data set information of the form:
    ``group = {'data1' : ( (shape1), (dtype1) ), 'data2' : ( (shape2), (dtype2) )}``

    group : `str`
        Name of the Hdf5 group
    data  : `str`
        Name of the column to be written
    shape : `tuple` ( `int` )
        The shape of the data for this dataset
    dtype : `str`
        The data type for this dataset

    For example
    ``initializeHdf5Write('test.hdf5', data = dict(scalar=((100000,), 'f4'), vect=((100000, 3), 'f4'))``

    Would initialize an hdf5 file with one group and two datasets, with shapes and data types as given
    """
    outdir = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(outdir):  # pragma: no cover
        os.makedirs(outdir, exist_ok=True)
    if comm is None:
        outf = h5py.File(filepath, "w")
    else:
        if not h5py.get_config().mpi:
            raise TypeError(
                "hdf5py module not prepared for parallel writing."
            )  # pragma: no cover
        outf = h5py.File(filepath, "w", driver="mpio", comm=comm)
    groups = {}
    for k, v in kwds.items():
        group = outf.create_group(k)
        groups[k] = group
        for key, shape in v.items():
            group.create_dataset(key, shape[0], shape[1])
    return groups, outf


def writeDictToHdf5ChunkSingle(fout, odict, start, end, **kwds):
    """Writes a data chunk to an hdf5 file

    Parameters
    ----------
    fout : `h5py.File`
        The file

    odict : `OrderedDict`, (`str`, `numpy.array`)
        The data being written

    start : `int`
        Starting row number to place the data

    end : `int`
        Ending row number to place the data

    Notes
    -----
    The kwds can be used to control the output locations, i.e., to
    rename the columns in data_dict when they good into the output file.

    For each item in data_dict, the output location is set as

    ``k_out = kwds.get(key, key)``

    This will check the kwds to see if they contain `key` and if so, will
    return the corresponding value.  Otherwise it will just return `key`.

    I.e., if `key` is present in kwds in will override the name.
    """
    for key, val in odict.items():
        k_out = kwds.get(key, key)
        fout[k_out][start:end] = val


def writeDictToHdf5Chunk(groups, odict, start, end, **kwds):
    """Writes a data chunk to an hdf5 file

    Parameters
    ----------
    fout : `h5py.File`
        The file

    odict : `OrderedDict`, (`str`, `numpy.array`)
        The data being written

    start : `int`
        Starting row number to place the data

    end : `int`
        Ending row number to place the data

    Notes
    -----
    The kwds can be used to control the output locations, i.e., to
    rename the columns in data_dict when they good into the output file.

    For each item in data_dict, the output location is set as

    `k_out = kwds.get(key, key)`

    This will check the kwds to see if they contain `key` and if so, will
    return the corresponding value.  Otherwise it will just return `key`.

    I.e., if `key` is present in kwds in will override the name.
    """
    for group_name, group in groups.items():
        for key, val in odict[group_name].items():
            k_out = kwds.get(key, key)
            group[k_out][start:end] = val


def finalizeHdf5Write(fout, groupname=None, **kwds):
    """Write any last data and closes an hdf5 file

    Parameters
    ----------
    fout : `h5py.File`
        The file

    Notes
    -----
    The keywords can be used to write additional data
    """
    if groupname is None:  # pragma: no cover
        group = fout
    else:
        group = fout.create_group(groupname)
    for k, v in kwds.items():
        group[k] = v
    fout.close()


# II B. Writing `astropy.table.Table`


def writeApTablesToFits(tables, filepath, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single FITS file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables
    filepath: `str`
        Path to output file
    kwargs:
        kwargs are passed to `astropy.io.fits.writeto` call.
    """
    out_list = [fits.PrimaryHDU()]
    for k, v in tables.items():
        hdu = fits.table_to_hdu(v)
        hdu.name = k
        out_list.append(hdu)
    hdu_list = fits.HDUList(out_list)
    hdu_list.writeto(filepath, **kwargs)


def writeApTablesToHdf5(tables, filepath, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file
    kwargs:
        kwargs are passed to `astropy.table.Table` call.
    """
    for k, v in tables.items():
        v.write(filepath, path=k, append=True, format="hdf5", **kwargs)


# II C. Writing `numpy.recarray`


def writeRecarraysToFits(recarrays, filepath, **kwargs):
    """
    Writes a dictionary of `np.recarray` to a single FITS file

    Parameters
    ----------
    recarrays  : `dict` of `np.recarray`
        Keys will be HDU names, values will be tables

    filepath: `str`
        Path to output file
    kwargs:
        kwargs are passed to `astropy.io.fits.writeto` call.
    """
    out_list = [fits.PrimaryHDU()]
    for k, v in recarrays.items():
        hdu = fits.BinTableHDU.from_columns(v.columns)
        hdu.name = k
        out_list.append(hdu)
    hdu_list = fits.HDUList(out_list)
    hdu_list.writeto(filepath, **kwargs)


# II D. Writing `OrderedDict`, (`str`, `numpy.array`) to `hdf5`


def writeDictToHdf5(odict, filepath, groupname, **kwargs):
    """
    Writes a dictionary of `numpy.array` or `jaxlib.xla_extension.DeviceArray`
    to a single hdf5 file

    Parameters
    ----------
    odict : `Mapping`, (`str`, `numpy.array` or `jaxlib.xla_extension.DeviceArray`)
        The data being written

    filepath: `str`
        Path to output file

    groupname : `str` or `None`
        The groupname for the data
    """
    # pylint: disable=unused-argument
    fout = h5py.File(filepath, "a")
    if groupname is None:  # pragma: no cover
        group = fout
    else:
        group = fout.require_group(groupname)
    for key, val in odict.items():
        try:
            if isinstance(val, np.ndarray):
                group.create_dataset(key, dtype=val.dtype, data=val.data)
            elif isinstance(val, list):
                arr = np.array(val)
                group.create_dataset(key, dtype=arr.dtype, data=arr)
            elif isinstance(val, dict):
                writeDictToHdf5(val, filepath, f"{group.name}/{key}")
            else:
                # In the future, it may be better to specifically case for
                # jaxlib.xla_extension.DeviceArray here. For now, we're
                # resorting to duck typing so we don't have to import jax just
                # to check the type.
                group.create_dataset(key, dtype=val.dtype, data=val.addressable_data(0))
        except Exception as msg:  # pragma: no cover
            print(f"Warning.  Failed to convert column {str(msg)}")
    fout.close()


def writeDictsToHdf5(odicts, filepath):
    """
    Writes a dictionary of `numpy.array` to a single hdf5 file

    Parameters
    ----------
    odicts : `OrderedDict`, (`str`, `Tablelike`)
        The data being written

    filepath: `str`
        Path to output file
    """
    try:
        os.unlink(filepath)
    except FileNotFoundError:
        pass
    for key, val in odicts.items():
        writeDictToHdf5(val, filepath, key)


# II E. Writing `pandas.DataFrame`


def writeDataFramesToH5(dataFrames, filepath):
    """
    Writes a dictionary of `pandas.DataFrame` to a single hdf5 file

    Parameters
    ----------
    dataFrames : `dict` of `pandas.DataFrame`
        Keys will be passed to 'key' parameter

    filepath: `str`
        Path to output file
    """
    for key, val in dataFrames.items():
        val.to_hdf(filepath, key)


def writeDataFramesToPq(dataFrames, filepath, **kwargs):
    """
    Writes a dictionary of `pandas.DataFrame` to a parquet files

    Parameters
    ----------
    tables : `dict` of `pandas.DataFrame`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for k, v in dataFrames.items():
        _ = v.to_parquet(f"{basepath}{k}{ext}", **kwargs)


# II F. Writing pyarrow table(s)


def writeTableToHd5(table, filepath, key):
    """
    Writes a `pyarrow.Table` to a single hdf5 file

    Parameters
    ----------
    table : `dict` of `pyarrow.Table`
        Keys will be passed to 'key' parameter

    filepath: `str`
        Path to output file

    key: `str`
        The hdf5 groupname
    """
    writeDictToHdf5(table.to_pydict(), filepath, key)


def writeTablesToHd5(tables, filepath):
    """
    Writes a dictionary of `pyarrow.Table` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `pyarrow.Table`
        Keys will be passed to 'key' parameter

    filepath: `str`
        Path to output file
    """
    for key, val in tables.items():
        writeTableToHd5(val, filepath, key)


def writeTablesToPq(tables, filepath, **kwargs):
    """
    Writes a dictionary of `pyarrow.Table` to a parquet files

    Parameters
    ----------
    tables : `dict` of `pyarrow.Table`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    basepath, ext = os.path.splitext(filepath)
    if not ext:  # pragma: no cover
        ext = "." + FILE_FORMAT_SUFFIX_MAP[PANDAS_PARQUET]
    for k, v in tables.items():
        pq.write_table(v, f"{basepath}{k}{ext}", **kwargs)
