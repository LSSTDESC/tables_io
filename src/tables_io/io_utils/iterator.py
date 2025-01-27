"""IO Iterator Read Functions for tables_io"""

import os
from collections import OrderedDict
from collections.abc import Iterator, Iterable
from typing import Optional, Union, Mapping, List

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
    TABULAR_FORMAT_NAMES,
    file_type,
    is_tabledict_like,
    is_table_like,
    table_type,
)


# I. Top Level Interface Functions


def iterator(
    filepath: str,
    tType: Optional[int] = None,
    fmt: Optional[str] = None,
    chunk_size: Optional[int] = 100_000,
    rank: Optional[int] = 0,
    parallel_size: Optional[int] = 1,
    **kwargs,
):
    """Iterates through the data in a given file. The data is yielded (along with
    the start and stop index) as a `Tablelike` object. If no `tType` is given,
    the tabular format will be the default for that file type. If `tType` is given,
    the tabular format will be converted to that table type.

    For a given file type, there are additional arguments that can be supplied to
    the native file reader. The main arguments that are needed are `groupname` for
    HDF5 files, and `columns` for parquet files. Other arguments for reading parquet
    files can be found in the documentation of `pyarrow.parquet.read_table` or
    `pyarrow.dataset.dataset`.

    Parameters
    ----------
    filepath : `str`
        File to load
    tType : `int` or `None`
        Table type, if `None` this will use `readNative`
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension
    chunk_size: int
        The size of data chunk to iterate over, by default 100 000
    rank: int
        The rank of this process under MPI, by default 0
    parallel_size: int
        The number of processes under MPI, by default 1

    Optional **kwargs
    -----------------
    groupname : `str` or `None`
        For HDF5 files, the group name where the data is
    columns : `str` or `None`
        For parquet files, the names of the columns to read.
        `None` will read all the columns

    Returns
    -------
    start: int
        The starting index for the data.
    stop: int
        The end index for the data.
    data : Tablelike
        The data from [start:stop]. The format will be the native tabular format for the file
        if no `tType` is given. Otherwise, the data will be in the tabular format `tType`.

    """
    if isinstance(tType, str):
        try:
            int_tType = TABULAR_FORMAT_NAMES[tType]
        except:
            raise TypeError(
                f"Unsupported tableType '{tType}', must be one of {TABULAR_FORMAT_NAMES.keys()}"
            )
    if isinstance(tType, int):
        int_tType = tType

    for start, stop, data in iterator_native(
        filepath, fmt, chunk_size, rank, parallel_size, **kwargs
    ):
        yield start, stop, convert(data, int_tType)


def iterator_native(
    filepath: str,
    fmt: Optional[str] = None,
    chunk_size: Optional[int] = 100_000,
    rank: Optional[int] = 0,
    parallel_size: Optional[int] = 1,
    **kwargs,
):
    """Iterates through the data in a given file. The data is yielded (along with
    the start and stop index) as a `Tablelike` object that has the default format
    for the given file type.

    Any **kwargs are passed to the specific iterator function for the file type.

    Parameters
    ----------
    filepath : `str`
        File to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension. By default `None`.
    chunk_size: int
        The size of data chunk to iterate over, by default 100 000
    rank: int
        The rank of this process under MPI, by default 0
    parallel_size: int
        The number of processes under MPI, by default 1

    Returns
    -------
    start: int
        Data start index
    stop: int
        Data ending index
    data : `TableLike`
        The data in the native type for that file, from [start:stop]


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
        return theFunc(
            filepath,
            chunk_size=chunk_size,
            rank=rank,
            parallel_size=parallel_size,
            **kwargs,
        )
    except KeyError as msg:
        raise NotImplementedError(
            f"Unsupported FileType for iterateNative {fType}"
        ) from msg  # pragma: no cover


def get_input_data_length(filepath: str, fmt: Optional[str] = None, **kwargs):
    """Opens the given file and gets the length of data in that file. If the data is multi-dimensional, the
    function will give the length of the first axis of the data, which is typically the axis that you want
    to iterate over.

    Parameters
    ----------
    filepath : `str`
        File to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension.

    Returns
    -------
    data : `TableLike`
        The data

    Notes
    -----
    The kwargs are passed to the specific iterator type.

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


def get_input_data_length_HDF5(filepath: str, groupname: Optional[str] = None) -> int:
    """Open an HDF5 file and return the size of a group

    Parameters
    ----------
    filepath: `str`
        The input filepath.

    groupname : `str` or `None`
        The group name where the data is, by default None.


    Returns
    -------
    length : `int`
        The length of the data. In the case of a multi-dimensional array,
        this is the length of the first axis.

    Notes
    -----
    For a multi-D array this returns the length of the first axis
    and not the total size of the array.

    Normally that is what you want to be iterating over.
    """
    hg, infp = read_HDF5_group(filepath, groupname)
    nrow = get_group_input_data_length(hg)
    infp.close()
    return nrow


def iter_HDF5_to_dict(
    filepath: str,
    groupname: Optional[str] = None,
    chunk_size: int = 100_000,
    rank: int = 0,
    parallel_size: int = 1,
) -> Iterator[int, int, Mapping]:
    """
    Iterates through an `HDF5` file, yielding one chunk of data at a time
    as an `OrderedDict` of `np.array` objects.

    Parameters
    ----------
    filepath: str
        The input filepath
    groupname: str
        The group name where the data is, by default None.
    chunk_size: int
        The size of data chunk to iterate over, by default 100 000
    rank: int
        The rank of this process under MPI, by default 0
    parallel_size: int
        The number of processes under MPI, by default 1

    Yields
    -------
    start: int
        Data start index
    end: int
        Data ending index
    data: dict
        `OrderedDict` of `np.array` of all data from start:end
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


def iter_HDF5_to_dataframe(
    filepath: str,
    chunk_size: Optional[int] = 100_000,
    groupname=None,
    rank: Optional[int] = 0,
    parallel_size: Optional[int] = 1,
):
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
    filepath: str,
    chunk_size: int = 100_000,
    columns: Optional[List[str]] = None,
    rank: int = 0,
    parallel_size: int = 1,
    **kwargs,
):
    """
    Iterates through a parquet file, yielding one chunk of data at a time.

    Parameters
    ----------
    filepath: str
        path to input file
    chunk_size: int
        The maximum chunk size of the data, by default = 100_000
    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns
    rank: int
        The rank of this process under MPI, by default 0
    parallel_size: int
        The number of processes under MPI, by default 1
    **kwargs : additional arguments to pass to the parquet read_table function

    Yields
    ------
    start: int
        Data start index
    end: int
        Data ending index
    data: `pandas.DataFrame`
        DataFrame of all data from start:end
    """
    if rank >= parallel_size:
        raise TypeError(
            f"MPI rank {rank} larger than the total "
            f"number of processes {parallel_size}"
        )  # pragma: no cover

    num_rows = get_input_data_length_pq(filepath, columns=columns)
    _ranges = data_ranges_by_rank(num_rows, chunk_size, parallel_size, rank)

    parquet_file = pq.read_table(filepath, columns=columns, **kwargs)
    start = 0
    end = 0

    batches = parquet_file.to_batches(max_chunksize=chunk_size)

    for table_chunk in batches:
        data = pa.Table.from_batches([table_chunk]).to_pandas()
        num_rows = len(data)
        end += num_rows
        yield start, end, data
        start += num_rows


def get_input_data_length_pq(
    filepath: str, columns: Optional[List[str]] = None, **kwargs
) -> int:
    """Open a Parquet file and return the size of a group

    Parameters
    ----------
    filepath: `str`
        Path to input file
    columns : `List[str]` or `None`
        The groupname for the data
    **kwargs: additional arguments to pass to the pyarrow.parquet.read_table function


    Returns
    -------
    nrow : `int`
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


def iter_ds_to_table(source, columns: Optional[List[str]] = None, **kwargs):
    """
    Iterator for sending chunks of data in parquet

    Parameters
    ----------
    source: str
        input file name
    columns: List[str], default None
        The list of columns to use
    **kwargs : additional arguments to pass to the pyarrow.dataset.to_batches() function

    Yields
    ------
    start: int
        Data start index
    end: int
        Data ending index
    data: `pyarrow.Table`
        table of all data from start:end
    """
    start = 0
    end = 0
    dataset = ds.dataset(source, **kwargs)

    for batch in dataset.to_batches(columns=columns):
        data = pa.Table.from_pydict(batch.to_pydict())
        num_rows = len(data)
        end += num_rows
        yield start, end, data
        start += num_rows


def get_input_data_length_ds(source, **kwargs) -> int:
    """Open a dataset and return the number of rows in a group

    Parameters
    ----------
    source: `str`
        Path to input file or directory
    **kwargs:
        **kwargs are passed to pyarrow.dataset.dataset()

    Returns
    -------
    nrows : `int`
        The length of the data

    """
    dataset = ds.dataset(source, **kwargs)
    nrows = dataset.count_rows()
    return nrows


# II D. Iteration utility functions


def split_tasks_by_rank(
    tasks: Iterable[int], parallel_size: int, rank: int
) -> Iterator[int]:
    """Iterate through a list of tasks, yielding ones this process is responsible for.

    Tasks are allocated in a round-robin way.

    Parameters
    ----------
    tasks: iterator
        Tasks to split up
    parallel_size: int
        The number of processes under MPI
    rank: int
        The rank of this process under MPI

    Yields
    -------
    task: int
        The number of the first task for this process
    """
    for i, task in enumerate(tasks):
        if i % parallel_size == rank:
            yield task


def data_ranges_by_rank(
    n_rows: int, chunk_rows: int, parallel_size: int, rank: int
) -> Iterator[int, int]:
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
        The number of processes under MPI
    rank: int
        The rank of this process under MPI

    Yields
    ------
    start: int
        Data start index
    end: int
        Data ending index
    """
    n_chunks = n_rows // chunk_rows
    if n_chunks * chunk_rows < n_rows:  # pragma: no cover
        n_chunks += 1
    it = split_tasks_by_rank(range(n_chunks), parallel_size, rank)
    for i in it:
        start = i * chunk_rows
        end = min((i + 1) * chunk_rows, n_rows)
        yield start, end
