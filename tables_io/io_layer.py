"""IO Functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

try:
    from astropy.table import Table as apTable
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:  #pragma: no cover
    print("Warning: astropy.table is not available.  tables_io will still work with other formats.")
    HAS_ASTROPY = False

try:
    import h5py
    HAS_HDF5 = True
except ImportError:  #pragma: no cover
    print("Warning: h5py is not available.  tables_io will still work with other formats.")
    HAS_HDF5 = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  #pragma: no cover
    print("Warning: pandas is not available.  tables_io will still work with other formats.")
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PQ = True
except ImportError:  #pragma: no cover
    print("Warning: parquet is not available.  tables_io will still work with other formats.")
    HAS_PQ = False


def forceToPandables(arr, check_nrow=None):
    """
    Forces a  `numpy.array` into a format that panda can handle

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
    shape = np.shape(arr)
    nrow = shape[0]
    if check_nrow is not None and check_nrow != nrow:  # pragma: no cover
        raise ValueError("Number of rows does not match: %i != %i" % (nrow, check_nrow))
    if ndim == 1:
        return arr
    if ndim == 2:
        return list(arr)
    shape = np.shape(arr)  #pragma: no cover
    ncol = np.product(shape[1:])  #pragma: no cover
    return list(arr.reshape(nrow, ncol))  #pragma: no cover


def tableToDataframe(tab):
    """
    Convert an `astropy.table.Table` to a `pandas.DataFrame`

    Parameters
    ----------
    tab : `astropy.table.Table`
        The table

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    if not HAS_PANDAS:  #pragma: no cover
        raise ImportError("pandas is not available, can't make DataFrame")

    o_dict = OrderedDict()
    for colname in tab.columns:
        col = tab[colname]
        o_dict[colname] = forceToPandables(col.data)
    df = pd.DataFrame(o_dict)
    for k, v in tab.meta.items():
        df.attrs[k] = v  #pragma: no cover
    return df


def dataframeToArrayDict(df):
    """
    Convert a `pandas.DataFrame` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returnes
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key in df.keys():
        data[key] = np.array(df[key])
    return data


def hdf5GroupToArrayDict(hg):
    """
    Convert a `hdf5` object to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    hg :  FIXME
        The hdf5 object

    Returnes
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key in hg.keys():
        data[key] = np.array(hg[key])
    return data


def arraysToDataframe(array_dict, meta=None):  #pragma: no cover
    """
    Convert a `dict` of  `numpy.array` to a `pandas.DataFrame`

    Parameters
    ----------
    array_dict : `astropy.table.Table`
        The arrays

    meta : `dict` or `None`
        Optional dictionary of metadata

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is not available, can't make DataFrame")

    o_dict = OrderedDict()
    for k, v in array_dict:
        o_dict[k] = forceToPandables(v)
    df =  pd.DataFrame(o_dict)
    if meta is not None:
        for k, v in meta.items():
            df.attrs[k] = v
    return df


def dataframeToTable(df):
    """
    Convert a `pandas.DataFrame` to an `astropy.table.Table`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    if not HAS_ASTROPY:  #pragma: no cover
        raise ImportError("Astropy is not available, can't make astropy tables")

    o_dict = OrderedDict()
    for colname in df.columns:
        col = df[colname]
        if col.dtype.name == 'object':
            o_dict[colname] = np.vstack(col.to_numpy())
        else:
            o_dict[colname] = col.to_numpy()
    tab = apTable(o_dict)
    for k, v in df.attrs.items():
        tab.meta[k] = v  #pragma: no cover
    return tab


def tablesToDataframes(tables):
    """
    Convert several `astropy.table.Table` to `pandas.DataFrame`

    Parameters
    ----------
    tab : `dict` of `astropy.table.Table`
        The tables

    Returns
    -------
    df :  `OrderedDict` of `pandas.DataFrame`
        The dataframes
    """
    return OrderedDict([(k, tableToDataframe(v)) for k, v in tables.items()])


def dataframesToTables(dataframes):
    """
    Convert several `pandas.DataFrame` to `astropy.table.Table`

    Parameters
    ----------
    datafarmes :  `dict` of `pandas.DataFrame`
        The dataframes

    Returns
    -------
    tabs : `OrderedDict` of `astropy.table.Table`
        The tables
    """
    return OrderedDict([(k, dataframeToTable(v)) for k, v in dataframes.items()])


def writeTablesToFits(tables, filepath, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single FITS file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be HDU names, values will be tables

    filepath: `str`
        Path to output file

    kwargs are passed to `astropy.io.fits.writeto` call.
    """
    if not HAS_ASTROPY:  #pragma: no cover
        raise ImportError("Astropy is not available, can't save to FITS")
    out_list = [fits.PrimaryHDU()]
    for k, v in tables.items():
        hdu = fits.table_to_hdu(v)
        hdu.name = k
        out_list.append(hdu)
    hdu_list = fits.HDUList(out_list)
    hdu_list.writeto(filepath, **kwargs)


def readFitsToTables(filepath):
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
    if not HAS_ASTROPY:  #pragma: no cover
        raise ImportError("Astropy is not available, can't read FITS")
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        tables[hdu.name.lower()] = apTable.read(filepath, hdu=hdu.name)
    return tables


def writeTablesToHdf5(tables, filepath, **kwargs):
    """
    Writes a dictionary of `astropy.table.Table` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `astropy.table.Table`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    kwargs are passed to `astropy.table.Table` call.
    """
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't save to hdf5")

    for k, v in tables.items():
        v.write(filepath, path=k, append=True, **kwargs)


def readHdf5ToTables(filepath):
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
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't read hdf5")
    fin = h5py.File(filepath)
    tables = OrderedDict()
    for k in fin.keys():
        tables[k] = apTable.read(filepath, path=k)
    return tables


def readHdf5ToDataframe(filepath):
    """
    Reads `pandas.DataFrame` objects from an hdf5 file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    df : `pandas.DataFrame`
        The dataframe
    """
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't read hdf5")
    if not HAS_PANDAS:  #pragma: no cover
        raise ImportError("pandas is not available, can't read hdf5 to dataframe")
    return pd.read_hdf(filepath)


def readHdf5Group(infile, groupname=None):
    """ Get the size of data in a particular group in an hdf5 file

    Parameters
    ----------
    infile : `str`
        File in question
    groupname : `str` or `None`
        For hdf5 files, the groupname for the data

    Returns
    -------
    FIXME
    """
    infp = h5py.File(infile, "r")
    if groupname != 'None':  #pragma: no cover
        return infp[groupname], infp
    return infp, infp


def writeArraysToHdf5(arrays, filepath, **kwargs):
    """
    Writes a dictionary of `numpy.array` to a single hdf5 file

    Parameters
    ----------
    tables : `dict` of `numpy.array`
        Keys will be passed to 'path' parameter

    filepath: `str`
        Path to output file

    """
    # pylint: disable=unused-argument
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't save to hdf5")
    raise NotImplementedError("writeArraysToHdf5")  #pragma: no cover


def readHdf5ToArrays(filepath):
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
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't read hdf5")
    raise NotImplementedError("writeArraysToHdf5")  #pragma: no cover


def writeDataframesToPq(dataFrames, filepath, **kwargs):
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
        _ = v.to_parquet("%s%s.pq" % (filepath, k), **kwargs)


def readPqToDataframe(infile):
    """
    Reads a `pandas.DataFrame` object from an parquet file.

    Parameters
    ----------
    infile: `str`
        Path to input file

    Returns
    -------
    df : `pandas.DataFrame`
        The data frame
    """
    return pd.read_parquet(infile, engine='pyarrow')


def readPqToDataframes(basepath, keys=None, **kwargs):
    """
    Reads `pandas.DataFrame` objects from an parquet file.

    Parameters
    ----------
    basepath: `str`
        Path to input file

    keys : `list`
        Keys for the input objects.  Used to complete filepaths

    Returns
    -------
    tables : `OrderedDict` of `pandas.DataFrame`
        Keys will be taken from keys
    """

    if keys is None:  #pragma: no cover
        keys = [""]
    dataframes = OrderedDict()
    for key in keys:
        try:
            pqtab = pq.read_table("%s%s.pq" % (basepath, key), **kwargs)
            dataframes[key] = pqtab.to_pandas()
        except Exception:  #pragma: no cover
            pass
    return dataframes


def readPqToArrayDict(infile):
    """ Open a parquet file and return a dictionary of `numpy.array`
    """
    df = readPqToDataframe(infile)
    return dataframeToArrayDict(df)


def readH5ToArrayDict(infile):
    """ Open an h5 file and and return a dictionary of `numpy.array`
    """
    df = readHdf5ToDataframe(infile)
    return dataframeToArrayDict(df)


def readHdf5ToArrayDict(infile, groupname='None'):
    """ Read in h5py hdf5 data, return a dictionary of all of the keys
    """
    hg, infp = readHdf5Group(infile, groupname)
    data = hdf5GroupToArrayDict(hg)
    infp.close()
    return data


def readFileToArrayDict(filename, fmt=None, groupname='None'):
    """ Load training data

    Parameters
    ----------
    filename : `str`
        File to load
    fmt : `str`
        File format
    groupname : `str` or `None`
        For hdf5 files, the groupname for the data

    Returns
    -------
    data : `dict` ( `str` -> `numpy.array` )
        The data
    """
    if fmt is None:
        _, ext = os.path.splitext(filename)
        fmt = ext[1:]

    fmtlist = ['hdf5', 'parquet', 'h5']
    if fmt not in fmtlist:
        raise NotImplementedError(f"File format {fmt} not implemented")
    if fmt == 'hdf5':
        data = readHdf5ToArrayDict(filename, groupname)
    if fmt == 'parquet':
        data = readPqToArrayDict(filename)
    if fmt == 'h5':
        data = readH5ToArrayDict(filename)
    return data


def getGroupInputDataSize(hg):
    """ Return the size of a HDF5 group """
    firstkey = list(hg.keys())[0]
    nrows = len(hg[firstkey])
    return nrows


def getInputDataSizeHdf5(infile, groupname='None'):
    """ Open an HDF5 file and return  the size of a group """
    hg, infp = readHdf5Group(infile, groupname)
    nrow = getGroupInputDataSize(hg)
    infp.close()
    return nrow


def initializeHdf5Writeout(outfile, **kwds):
    """ Prepares an hdf5 files for parallel output

    Parameters
    ----------
    outfile : `str`
        The output file name

    Notes
    -----
    The keywords should be used to create_datasets within the hdf5 file.

    Each keyword should provide a tuple of ( (shape), (dtype) )

    shape : `tuple` ( `int` )
        The shape of the data for this dataset
    dtype : `str`
        The data type for this dataset

    For exmaple
    `initialize_writeout('test.hdf5', scalar=((100000,), 'f4'), vect=((100000, 3), 'f4'))`

    Would initialize an hdf5 file with two datasets, which shapes and data types as given
    """
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):  #pragma: no cover
        os.makedirs(outdir, exist_ok=True)
    outf = h5py.File(outfile, "w")
    for k, v in kwds.items():
        outf.create_dataset(k, v[0], v[1])
    return outf


def writeoutHdf5Chunk(outf, data_dict, start, end, **kwds):
    """ Writes a data chunk to an hdf5 file

    Parameters
    ----------
    outf : `h5py.File`
        The file

    data_dict : `dict`
        The data being written

    start : `int`
        Starting row number to place the data

    end : `int`
        Ending row number to place the data

    Notes
    -----
    The kwds can be used to control the output locations

    For each item in data_dict, the output location is set as

    `k_out = kwds.get(key, key)`
    """
    for key, val in data_dict.items():
        k_out = kwds.get(key, key)
        outf[k_out][start:end] = val


def finalizeHdf5Writeout(outf, **kwds):
    """ Write any last data and closes an hdf5 file

    Parameters
    ----------
    outf : `h5py.File`
        The file

    Notes
    -----
    The keywords can be used to write additional data
    """
    for k, v in kwds.items():
        outf[k] = v
    outf.close()


def iterChunkHdf5Data(infile, chunk_size=100_000, groupname='None'):
    """
    iterator for sending chunks of data in hdf5.

    Parameters
    ----------
      infile: input file name (str)
      chunk_size: size of chunk to iterate over (int)

    Returns
    -------
    output:
        iterator chunk

    Currently only implemented for hdf5, returns `tuple`
        start: start index (int)
        end: ending index (int)
        data: dictionary of all data from start:end (dict)
    """
    f, infp = readHdf5Group(infile, groupname)
    num_rows = getGroupInputDataSize(f)
    data = OrderedDict()
    for i in range(0, num_rows, chunk_size):
        start = i
        end = i+chunk_size
        if end > num_rows:
            end = num_rows
        for key in f.keys():
            data[key] = np.array(f[key][start:end])
        yield start, end, data
    infp.close()
