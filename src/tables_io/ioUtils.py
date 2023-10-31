"""IO Functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

from .arrayUtils import getGroupInputDataLength
from .convUtils import convert, dataFrameToDict, hdf5GroupToDict
from .lazy_modules import apTable, fits, h5py, pa, pd, pq
from .types import (
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
    PANDAS_HDF5,
    PANDAS_PARQUET,
    PD_DATAFRAME,
    fileType,
    istabledictlike,
    istablelike,
    tableType,
)

### I. Iteration functions


### I A. HDF5 partial read/write functions


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


def getInputDataLengthHdf5(filepath, groupname=None):
    """Open an HDF5 file and return the size of a group

    Parameters
    ----------
    filepath: `str`
        Path to input file

    groupname : `str` or `None`
        The groupname for the data


    Returns
    -------
    length : `int`
        The length of the data

    Notes
    -----
    For a multi-D array this return the length of the first axis
    and not the total size of the array.

    Normally that is what you want to be iterating over.
    """
    hg, infp = readHdf5Group(filepath, groupname)
    nrow = getGroupInputDataLength(hg)
    infp.close()
    return nrow


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
            raise TypeError("hdf5py module not prepared for parallel writing.")  # pragma: no cover
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
        MPI commuticator to do parallel writing

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
            raise TypeError("hdf5py module not prepared for parallel writing.")  # pragma: no cover
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


def split_tasks_by_rank(tasks, parallel_size, rank):
    """Iterate through a list of items, yielding ones this process is responsible for/

    Tasks are allocated in a round-robin way.

    Parameters
    ----------
    tasks: iterator
        Tasks to split up
    parallel_size: int
        the number of processes under MPI
    rank: int
        the rank of this process under MPI

    Returns
    -------
    output: iterator
        number of the first task for this process
    """
    for i, task in enumerate(tasks):
        if i % parallel_size == rank:
            yield task


def data_ranges_by_rank(n_rows, chunk_rows, parallel_size, rank):
    """Split a number of rows by process.

    Given a total number of rows to read and a chunk size, yield
    the ranges within them that this process should handle.

    Parameters
    ----------
    n_rows: int
        Total number of rows to split up
    chunk_rows: int
        Size of each chunk to be read
    parallel_size: int
        the number of processes under MPI
    rank: int
        the rank of this process under MPI

    Yields
    ------
    start: int
        start index
    end: int
        ending index
    """
    n_chunks = n_rows // chunk_rows
    if n_chunks * chunk_rows < n_rows:  # pragma: no cover
        n_chunks += 1
    it = split_tasks_by_rank(range(n_chunks), parallel_size, rank)
    for i in it:
        start = i * chunk_rows
        end = min((i + 1) * chunk_rows, n_rows)
        yield start, end


def iterHdf5ToDict(filepath, chunk_size=100_000, groupname=None, rank=0, parallel_size=1):
    """
    iterator for sending chunks of data in hdf5.

    Currently only implemented for hdf5, returns `tuple`

    Parameters
    ----------
    filepath: str
        input file name
    chunk_size: int
        size of chunk to iterate over
    rank: int
        the rank of this process under MPI
    parallel_size: int
        the number of processes under MPI

    Yields
    -------
    start: int
        start index
    end: int
        ending index
    data: dict
        dictionary of all data from start:end
    """
    if rank >= parallel_size:
        raise TypeError(
            f"MPI rank {rank} larger than the total " f"number of processes {parallel_size}"
        )  # pragma: no cover
    f, infp = readHdf5Group(filepath, groupname)
    num_rows = getGroupInputDataLength(f)
    ranges = data_ranges_by_rank(num_rows, chunk_size, parallel_size, rank)
    data = OrderedDict()
    for start, end in ranges:
        for key, val in f.items():
            data[key] = readHdf5DatasetToArray(val, start, end)
        yield start, end, data
    infp.close()


def iterH5ToDataFrame(filepath, chunk_size=100_000, groupname=None):
    """
    iterator for sending chunks of data in hdf5.

    Parameters
    ----------
    filepath: str
        input file name
    chunk_size: int
        size of chunk to iterate over

    Returns
    -------
    output:
        iterator chunk
    """
    raise NotImplementedError("iterH5ToDataFrame")

    # This does't work b/c of the difference in structure
    # f, infp = readHdf5Group(filepath, groupname)
    # num_rows = getGroupInputDataLength(f)
    # for i in range(0, num_rows, chunk_size):
    #    start = i
    #    end = i+chunk_size
    #    if end > num_rows:
    #        end = num_rows
    #    data = pd.read_hdf(filepath, start=start, stop=end)
    #    yield start, end, data
    # infp.close()


def iterPqToDataFrame(filepath, chunk_size=100_000, columns=None, **kwargs):
    """
    iterator for sending chunks of data in parquet

    Parameters
    ----------
    filepath: str
        input file name
    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns
    **kwargs : additional arguments to pass to the native file reader

    Yields
    ------
    start: int
        start index
    end: int
        ending index
    data: `pandas.DataFrame`
        data frame of all data from start:end
    """
    parquet_file = pq.read_table(filepath, columns=columns, **kwargs)
    start = 0
    end = 0
    for table_chunk in parquet_file.to_batches(max_chunksize=chunk_size):
        data = pa.Table.from_batches([table_chunk]).to_pandas()
        num_rows = len(data)
        end += num_rows
        yield start, end, data
        start += num_rows


### II.   Reading and Writing Files

### II A.  Reading and Writing `astropy.table.Table` to/from FITS files


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


def readFitsToApTables(filepath):
    """
    Reads `astropy.table.Table` objects from a FITS file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables
    """
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        tables[hdu.name.lower()] = apTable.Table.read(filepath, hdu=hdu.name)
    return tables


### II A'.  Reading and Writing `` to/from FITS files


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


def readFitsToRecarrays(filepath):
    """
    Reads `np.recarray` objects from a FITS file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `np.recarray`
        Keys will be HDU names, values will be tables
    """
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        tables[hdu.name.lower()] = hdu.data
    return tables


### II B.  Reading and Writing `astropy.table.Table` to/from `hdf5`


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


def readHdf5ToApTables(filepath):
    """
    Reads `astropy.table.Table` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    tables : `OrderedDict` of `astropy.table.Table`
        Keys will be 'paths', values will be tables
    """
    fin = h5py.File(filepath)
    tables = OrderedDict()
    for k in fin.keys():
        tables[k] = apTable.Table.read(filepath, path=k, format="hdf5")
    return tables


### II C.  Reading and Writing `OrderedDict`, (`str`, `numpy.array`) to/from `hdf5`


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
    return OrderedDict([(key, readHdf5DatasetToArray(val, start, end)) for key, val in hg.items()])


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
        group = fout.create_group(groupname)
    for key, val in odict.items():
        try:
            if isinstance(val, np.ndarray):
                group.create_dataset(key, dtype=val.dtype, data=val.data)
            else:
                # In the future, it may be better to specifically case for
                # jaxlib.xla_extension.DeviceArray here. For now, we're
                # resorting to duck typing so we don't have to import jax just
                # to check the type.
                group.create_dataset(key, dtype=val.dtype, data=val.device_buffer)
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


def readHdf5ToDicts(filepath):
    """
    Reads `numpy.array` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    dicts : `OrderedDict`, (`str`, `OrderedDict`, (`str`, `numpy.array`) )
        The data
    """
    fin = h5py.File(filepath)
    return OrderedDict([(key, readHdf5GroupToDict(val)) for key, val in fin.items()])


### II C.  Reading and Writing `pandas.DataFrame` to/from `hdf5`


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


def readH5ToDataFrames(filepath):
    """Open an h5 file and and return a dictionary of `pandas.DataFrame`

    Parameters
    ----------
    filepath: `str`
        Path to input file

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
    return OrderedDict([(key, readHdf5ToDataFrame(filepath, key=key)) for key in fin.keys()])


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


### II E.  Reading and Writing `pandas.DataFrame` to/from `parquet`


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
    for k, v in dataFrames.items():
        _ = v.to_parquet(f"{filepath}{k}.pq", **kwargs)


def readPqToDataFrames(basepath, keys=None, allow_missing_keys=False, columns=None, **kwargs):
    """
    Reads `pandas.DataFrame` objects from an parquet file.

    Parameters
    ----------
    basepath: `str`
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
    for key in keys:
        try:
            column_list = None
            if pd.api.types.is_dict_like(columns):
                column_list = columns[key]
            elif pd.api.types.is_list_like(columns):
                column_list = columns
            print("column_list", column_list)

            dataframes[key] = readPqToDataFrame(f"{basepath}{key}.pq", columns=column_list, **kwargs)
        except FileNotFoundError as msg:  # pragma: no cover
            if allow_missing_keys:
                continue
            raise msg
    return dataframes


### II F.  Reading and Writing to `OrderedDict`, (`str`, `numpy.array`)


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
    return OrderedDict([(c_name, col.to_numpy()) for c_name, col in zip(tab.column_names, tab.itercolumns())])


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


### II G.  Top-level interface functions


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
    if fType in [ASTROPY_HDF5, NUMPY_HDF5, PANDAS_HDF5]:
        return h5py.File(filepath, **kwargs)
    if fType in [PANDAS_PARQUET]:
        # basepath = os.path.splitext(filepath)[0]
        return pq.ParquetFile(filepath, **kwargs)
    raise TypeError(f"Unsupported FileType {fType}")  # pragma: no cover


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
        return readFitsToApTables(filepath)
    if fType == ASTROPY_HDF5:
        return readHdf5ToApTables(filepath)
    if fType == NUMPY_HDF5:
        return readHdf5ToDicts(filepath)
    if fType == NUMPY_FITS:
        return readFitsToRecarrays(filepath)
    if fType == PANDAS_HDF5:
        return readH5ToDataFrames(filepath)
    if fType == PANDAS_PARQUET:
        basepath = os.path.splitext(filepath)[0]
        return readPqToDataFrames(basepath, keys, allow_missing_keys, **kwargs)
    raise TypeError(f"Unsupported FileType {fType}")  # pragma: no cover


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
    odict = readNative(filepath, fmt, keys, allow_missing_keys, **kwargs)
    if len(odict) == 1:
        # For special keys, use the table alone without an enclosing dictionary.
        single_dict_key = list(odict.keys())[0]
        if single_dict_key in ["", None, "__astropy_table__", "data"]:
            odict = odict[single_dict_key]
    if tType is None:  # pragma: no cover
        return odict
    return convert(odict, tType)


def iteratorNative(filepath, fmt=None, **kwargs):
    """Read a file to the corresponding table type and iterate over the file

    Parameters
    ----------
    filepath : `str`
        File to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension

    Returns
    -------
    data : `TableLike`
        The data

    Notes
    -----
    The kwargs are used passed to the specific iterator type

    """
    fType = fileType(filepath, fmt)
    funcDict = {NUMPY_HDF5: iterHdf5ToDict, PANDAS_HDF5: iterH5ToDataFrame, PANDAS_PARQUET: iterPqToDataFrame}

    try:
        theFunc = funcDict[fType]
        return theFunc(filepath, **kwargs)
    except KeyError as msg:
        raise NotImplementedError(
            f"Unsupported FileType for iterateNative {fType}"
        ) from msg  # pragma: no cover


def iterator(filepath, tType=None, fmt=None, **kwargs):
    """Read a file to the corresponding table type iterate over the file

    Parameters
    ----------
    filepath : `str`
        File to load
    tType : `int` or `None`
        Table type, if `None` this will use `readNative`
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension
    groupname : `str` or `None`
        For hdf5 files, the groupname for the data

    Returns
    -------
    data : `OrderedDict` ( `str` -> `Tablelike` )
        The data

    """
    for start, stop, data in iteratorNative(filepath, fmt, **kwargs):
        yield start, stop, convert(data, tType)


def writeNative(odict, basename):
    """Write a file or files with tables

    Parameters
    ----------
    odict : `OrderedDict`, (`str`, `Tablelike`)
        The data to write
    basename : `str`
        Basename for the file to write.  The suffix will be applied based on the object type.
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
    filepath = basename + "." + fmt

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
        writeDataFramesToPq(odict, basename)
        return basename
    raise TypeError(f"Unsupported Native file type {fType}")  # pragma: no cover


def write(obj, basename, fmt=None):
    """Write a file or files with tables

    Parameters
    ----------
    obj : `Tablelike` or `TableDictLike`
        The data to write
    basename : `str`
        Basename for the file to write.  The suffix will be applied based on the object type.
    fmt : `str` or `None`
        The output file format, If `None` this will use `writeNative`
    """
    if fmt is None:
        splitpath = os.path.splitext(basename)
        if not splitpath[1]:
            return writeNative(obj, basename)
        basename = splitpath[0]
        fmt = splitpath[1][1:]

    try:
        fType = FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(f"Unknown file format {fmt}, options are {list(FILE_FORMAT_SUFFIXS.keys())}") from msg

    if istablelike(obj):
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], obj)])
    elif istabledictlike(obj):
        odict = obj
    elif not obj:
        return None
    else:
        raise TypeError(f"Can not write object of type {type(obj)}")

    if fType in [ASTROPY_HDF5, NUMPY_HDF5, NUMPY_FITS, PANDAS_PARQUET]:
        try:
            nativeTType = NATIVE_TABLE_TYPE[fType]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Native file type not known for {fmt}") from msg

        forcedOdict = convert(odict, nativeTType)
        return writeNative(forcedOdict, basename)

    if fType == ASTROPY_FITS:
        forcedOdict = convert(odict, AP_TABLE)
        filepath = f"{basename}.fits"
        writeApTablesToFits(forcedOdict, filepath)
        return filepath
    if fType == PANDAS_HDF5:
        forcedOdict = convert(odict, PD_DATAFRAME)
        filepath = f"{basename}.h5"
        writeDataFramesToH5(forcedOdict, filepath)
        return filepath

    raise TypeError(f"Unsupported File type {fType}")  # pragma: no cover
