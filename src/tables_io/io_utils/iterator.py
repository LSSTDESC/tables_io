"""IO Iterator Read Functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

from .read import read_HDF5_group, read_HDF5_dataset_to_array
from ..utils.array_utils import get_group_input_data_length, force_to_pandables
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
    file_type,
    is_tabledict_like,
    is_table_like,
    table_type,
)


# I. Top Level Interface Functions


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
    for start, stop, data in iterator_native(filepath, fmt, **kwargs):
        yield start, stop, convert(data, tType)


def iterator_native(filepath, fmt=None, **kwargs):
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
    fType = file_type(filepath, fmt)
    funcDict = {
        NUMPY_HDF5: iter_HDF5_to_dict,
        PANDAS_HDF5: iter_HDF5_to_dataframe,
        PANDAS_PARQUET: iter_pq_to_dataframe,
        PYARROW_PARQUET: iter_ds_to_table,
        PYARROW_HDF5: iter_ds_to_table,
    }

    try:
        theFunc = funcDict[fType]
        return theFunc(filepath, **kwargs)
    except KeyError as msg:
        raise NotImplementedError(
            f"Unsupported FileType for iterateNative {fType}"
        ) from msg  # pragma: no cover


def get_input_data_length(filepath, fmt=None, **kwargs):
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
    fType = file_type(filepath, fmt)
    funcDict = {
        NUMPY_HDF5: get_input_data_length_HDF5,
        PANDAS_HDF5: get_input_data_length_HDF5,
        PANDAS_PARQUET: get_input_data_length_pq,
        PYARROW_PARQUET: get_input_data_length_ds,
    }

    try:
        theFunc = funcDict[fType]
        return theFunc(filepath, **kwargs)
    except KeyError as msg:
        raise NotImplementedError(
            f"Unsupported FileType for getInputDataLength {fType}"
        ) from msg  # pragma: no cover


# II. Iteration sub functions

# II A. HDF5 partial read functions


def get_input_data_length_HDF5(filepath, groupname=None):
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
    hg, infp = read_HDF5_group(filepath, groupname)
    nrow = get_group_input_data_length(hg)
    infp.close()
    return nrow


def iter_HDF5_to_dict(
    filepath, chunk_size=100_000, groupname=None, rank=0, parallel_size=1
):
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
            f"MPI rank {rank} larger than the total "
            f"number of processes {parallel_size}"
        )  # pragma: no cover
    f, infp = read_HDF5_group(filepath, groupname)
    num_rows = get_group_input_data_length(f)
    ranges = data_ranges_by_rank(num_rows, chunk_size, parallel_size, rank)
    data = OrderedDict()
    for start, end in ranges:
        for key, val in f.items():
            data[key] = read_HDF5_dataset_to_array(val, start, end)
        yield start, end, data
    infp.close()


def iter_HDF5_to_dataframe(filepath, chunk_size=100_000, groupname=None):
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


# II B. Parquet partial read functions


def iter_pq_to_dataframe(
    filepath, chunk_size=100_000, columns=None, rank=0, parallel_size=1, **kwargs
):
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
    if rank >= parallel_size:
        raise TypeError(
            f"MPI rank {rank} larger than the total "
            f"number of processes {parallel_size}"
        )  # pragma: no cover

    num_rows = get_input_data_length_pq(filepath, columns=columns)
    _ranges = data_ranges_by_rank(num_rows, chunk_size, parallel_size, rank)

    parquet_file = pq.read_table(filepath, columns=columns)
    start = 0
    end = 0

    batches = parquet_file.to_batches(max_chunksize=chunk_size)

    for table_chunk in batches:
        data = pa.Table.from_batches([table_chunk]).to_pandas()
        num_rows = len(data)
        end += num_rows
        yield start, end, data
        start += num_rows


def get_input_data_length_pq(filepath, columns=None, **kwargs):
    """Open a Parquet file and return the size of a group

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
    tab = pq.read_table(filepath, columns=columns)
    nrow = len(tab[tab.column_names[0]])
    return nrow


# II C. Parquet dataset partial read functions


def iter_ds_to_table(source, columns=None, **kwargs):
    """
    iterator for sending chunks of data in parquet

    Parameters
    ----------
    dataset: str
        input file name
    **kwargs : additional arguments to pass to the native file reader

    Yields
    ------
    start: int
        start index
    end: int
        ending index
    data: `pyarrow.Table`
        table of all data from start:end
    """
    start = 0
    end = 0
    dataset = ds.dataset(source)

    for batch in dataset.to_batches(columns=columns):
        data = pa.Table.from_pydict(batch.to_pydict())
        num_rows = len(data)
        end += num_rows
        yield start, end, data
        start += num_rows


def get_input_data_length_ds(source, **kwargs):
    """Open a dataset and return the size of a group

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    length : `int`
        The length of the data

    Notes
    -----
    Normally that is what you want to be iterating over.
    """
    dataset = ds.dataset(source, **kwargs)
    nrows = dataset.count_rows()
    return nrows


# II D. Iteration utility functions


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
