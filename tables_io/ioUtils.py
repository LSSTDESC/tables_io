"""IO Functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

from .lazy_modules import pd, pq, h5py, apTable, fits
from .lazy_modules import HAS_TABLES, HAS_HDF5

from .arrayUtils import getGroupInputDataLength

from .types import ASTROPY_FITS, ASTROPY_HDF5, NUMPY_HDF5, PANDAS_HDF5, PANDAS_PARQUET,\
     NATIVE_FORMAT, FILE_FORMAT_SUFFIXS, FILE_FORMAT_SUFFIX_MAP, DEFAULT_TABLE_KEY,\
     NATIVE_TABLE_TYPE, AP_TABLE, PD_DATAFRAME,\
     fileType, tableType, istablelike, istabledictlike

from .convUtils import dataFrameToDict, hdf5GroupToDict, convert



### I. Iteration functions


### I A. HDF5 partial read/write functions

def readHdf5DatasetToArray(dataset, start=None, end=None):
    """
    Reads part of a hdf5 dataset into a `numpy.array`

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
    """ Open an HDF5 file and return the size of a group

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


def initializeHdf5Write(filepath, groupname=None, **kwds):
    """ Prepares an hdf5 file for output

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

    For exmaple
    `initialize_writeout('test.hdf5', scalar=((100000,), 'f4'), vect=((100000, 3), 'f4'))`

    Would initialize an hdf5 file with two datasets, with shapes and data types as given
    """
    outdir = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(outdir):  #pragma: no cover
        os.makedirs(outdir, exist_ok=True)
    outf = h5py.File(filepath, "w")
    if groupname is None:  #pragma: no cover
        group = outf
    else:
        group = outf.create_group(groupname)

    for k, v in kwds.items():
        group.create_dataset(k, v[0], v[1])
    return group, outf


def writeDictToHdf5Chunk(fout, odict, start, end, **kwds):
    """ Writes a data chunk to an hdf5 file

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
    for key, val in odict.items():
        k_out = kwds.get(key, key)
        fout[k_out][start:end] = val


def finalizeHdf5Write(fout, groupname=None, **kwds):
    """ Write any last data and closes an hdf5 file

    Parameters
    ----------
    fout : `h5py.File`
        The file

    Notes
    -----
    The keywords can be used to write additional data
    """
    if groupname is None:  #pragma: no cover
        group = fout
    else:
        group = fout.create_group(groupname)
    for k, v in kwds.items():
        group[k] = v
    fout.close()


def iterHdf5ToDict(filepath, chunk_size=100_000, groupname=None):
    """
    iterator for sending chunks of data in hdf5.

    Parameters
    ----------
      filepath: input file name (str)
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
    f, infp = readHdf5Group(filepath, groupname)
    num_rows = getGroupInputDataLength(f)
    data = OrderedDict()
    for i in range(0, num_rows, chunk_size):
        start = i
        end = i+chunk_size
        if end > num_rows:
            end = num_rows
        for key, val in f.items():
            data[key] = readHdf5DatasetToArray(val, start, end)
        yield start, end, data
    infp.close()


def iterH5ToDataFrame(filepath, chunk_size=100_000, groupname=None):
    """
    iterator for sending chunks of data in hdf5.

    Parameters
    ----------
      filepath: input file name (str)
      chunk_size: size of chunk to iterate over (int)

    Returns
    -------
    output:
        iterator chunk

    Currently only implemented for hdf5, returns `tuple`
        start: start index (int)
        end: ending index (int)
        data: pandas.DataFrame of all data from start:end (dict)
    """
    raise NotImplementedError("iterH5ToDataFrame")

    # This does't work b/c of the difference in structure
    #f, infp = readHdf5Group(filepath, groupname)
    #num_rows = getGroupInputDataLength(f)
    #for i in range(0, num_rows, chunk_size):
    #    start = i
    #    end = i+chunk_size
    #    if end > num_rows:
    #        end = num_rows
    #    data = pd.read_hdf(filepath, start=start, stop=end)
    #    yield start, end, data
    #infp.close()


def iterPqToDataFrame(filepath):
    """
    iterator for sending chunks of data in parquet

    Parameters
    ----------
      filepath: input file name (str)

    Returns
    -------
    output:
        iterator chunk

    Currently only implemented for hdf5, returns `tuple`
        start: start index (int)
        end: ending index (int)
        data: `pandas.DataFrame` of all data from start:end (dict)
    """
    raise NotImplementedError("iterPqToDataFrame")


### II.   Reading and Writing Files

### II A.  Reading and Writing `astropy.table.Table` to/from FITS files

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
    fin = fits.open(filepath)
    tables = OrderedDict()
    for hdu in fin[1:]:
        tables[hdu.name.lower()] = apTable.Table.read(filepath, hdu=hdu.name)
    return tables


### II B.  Reading and Writing `astropy.table.Table` to/from `hdf5`

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
        v.write(filepath, path=k, append=True, format='hdf5', **kwargs)


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
    fin = h5py.File(filepath)
    tables = OrderedDict()
    for k in fin.keys():
        tables[k] = apTable.Table.read(filepath, path=k, format='hdf5')
    return tables


### II C.  Reading and Writing `OrderedDict`, (`str`, `numpy.array`) to/from `hdf5`

def readHdf5Group(filepath, groupname=None):
    """ Read and return group from an hdf5 file.

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
    if groupname is None:  #pragma: no cover
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
    Writes a dictionary of `numpy.array` to a single hdf5 file

    Parameters
    ----------
    odict : `Mapping`, (`str`, `numpy.array`)
        The data being written

    filepath: `str`
        Path to output file

    groupname : `str` or `None`
        The groupname for the data
    """
    # pylint: disable=unused-argument
    fout = h5py.File(filepath, 'a')
    if groupname is None:  #pragma: no cover
        group = fout
    else:
        group = fout.create_group(groupname)
    for key, val in odict.items():
        try:
            group.create_dataset(key, dtype=val.dtype, data=val.data)
        except Exception as msg:  #pragma: no cover
            print("Warning failed to convert column %s" % msg)
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
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't read hdf5")
    if not HAS_TABLES:  #pragma: no cover
        raise ImportError("tables is not available, can't read hdf5 to dataframe")

    return pd.read_hdf(filepath, key)


def readH5ToDataFrames(filepath):
    """ Open an h5 file and and return a dictionary of `pandas.DataFrame`

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
    if not HAS_TABLES:  #pragma: no cover
        raise ImportError("tables is not available, can't read hdf5 to dataframe")
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
    if not HAS_HDF5:  #pragma: no cover
        raise ImportError("h5py is not available, can't write hdf5")
    if not HAS_TABLES:  #pragma: no cover
        raise ImportError("tables is not available, can't write hdf5")

    for key, val in dataFrames.items():
        val.to_hdf(filepath, key)




### II E.  Reading and Writing `pandas.DataFrame` to/from `parquet`

def readPqToDataFrame(filepath):
    """
    Reads a `pandas.DataFrame` object from an parquet file.

    Parameters
    ----------
    filepath: `str`
        Path to input file

    Returns
    -------
    df : `pandas.DataFrame`
        The data frame
    """
    return pd.read_parquet(filepath, engine='pyarrow')


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
        _ = v.to_parquet("%s%s.pq" % (filepath, k), **kwargs)


def readPqToDataFrames(basepath, keys=None):
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
        dataframes[key] = readPqToDataFrame("%s%s.pq" % (basepath, key))
    return dataframes


### II F.  Reading and Writing to `OrderedDict`, (`str`, `numpy.array`)

def readPqToDict(filepath, columns=None):
    """ Open a parquet file and return a dictionary of `numpy.array`

    Parameters
    ----------
    filepath: `str`
        Path to input file

    columns : `list` (`str`) or `None`
        Names of the columns to read, `None` will read all the columns

    Returns
    -------
    tab : `OrderedDict` (`str` : `numpy.array`)
       The data
    """
    tab = pq.read_table(filepath, columns=columns)
    return OrderedDict([(c_name, col.to_numpy()) for c_name, col in zip(tab.column_names, tab.itercolumns())])


def readH5ToDict(filepath, groupname=None):
    """ Open an h5 file and and return a dictionary of `numpy.array`

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
    """ Read in h5py hdf5 data, return a dictionary of all of the keys

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

def readNative(filepath, fmt=None, keys=None):
    """ Read a file to the corresponding table type

    Parameters
    ----------
    filepath : `str`
        File to load
    fmt : `str` or `None`
        File format, if `None` it will be taken from the file extension
    groupname : `str` or `None`
        For hdf5 files, the groupname for the data

    Returns
    -------
    data : `OrderedDict` ( `str` -> `Tablelike` )
        The data

    """
    fType = fileType(filepath, fmt)
    if fType == ASTROPY_FITS:
        return readFitsToTables(filepath)
    if fType == ASTROPY_HDF5:
        return readHdf5ToTables(filepath)
    if fType == NUMPY_HDF5:
        return readHdf5ToDicts(filepath)
    if fType == PANDAS_HDF5:
        return readH5ToDataFrames(filepath)
    if fType == PANDAS_PARQUET:
        basepath = os.path.splitext(filepath)[0]
        return readPqToDataFrames(basepath, keys)
    raise TypeError("Unsupported FileType %i" % fType)  #pragma: no cover


def read(filepath, tType=None, fmt=None, keys=None):
    """ Read a file to the corresponding table type

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
    odict = readNative(filepath, fmt, keys)
    if len(odict) == 1:
        for defName in ['', None, '__astropy_table__', 'data']:
            if defName in odict:
                odict = odict[defName]
    if tType is None:  #pragma: no cover
        return odict
    return convert(odict, tType)


def iterateNative(filepath, fmt=None, **kwargs):
    """ Read a file to the corresponding table type and iterate over the file

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
    funcDict = {NUMPY_HDF5:iterHdf5ToDict,
                PANDAS_HDF5:iterH5ToDataFrame,
                PANDAS_PARQUET:iterPqToDataFrame}

    try:
        theFunc = funcDict[fType]
        return theFunc(filepath, **kwargs)
    except KeyError as msg:
        raise NotImplementedError("Unsupported FileType for iterateNative %i" % fType) from msg #pragma: no cover


def iterate(filepath, tType=None, fmt=None, **kwargs):
    """ Read a file to the corresponding table type iterate over the file

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
    for start, stop, data in iterateNative(filepath, fmt, **kwargs):
        yield start, stop, convert(data, tType)


def writeNative(odict, basename):
    """ Write a file or files with tables

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
    elif not odict:  #pragma: no cover
        return None
    else:  #pragma: no cover
        raise TypeError("Can not write object of type %s" % type(odict))

    try:
        fType = NATIVE_FORMAT[tType]
    except KeyError as msg:  #pragma: no cover
        raise KeyError("No native format for table type %i" % tType) from msg
    fmt = FILE_FORMAT_SUFFIX_MAP[fType]
    filepath = basename + '.' + fmt

    if istable:
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], odict)])

    try:
        os.unlink(filepath)
    except FileNotFoundError:
        pass

    if fType == ASTROPY_HDF5:
        writeTablesToHdf5(odict, filepath)
        return filepath
    if fType == NUMPY_HDF5:
        writeDictsToHdf5(odict, filepath)
        return filepath
    if fType == PANDAS_PARQUET:
        writeDataFramesToPq(odict, basename)
        return basename
    raise TypeError("Unsupported Native file type %i" % fType)  #pragma: no cover


def write(obj, basename, fmt=None):
    """ Write a file or files with tables

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
        return writeNative(obj, basename)
    try:
        fType = FILE_FORMAT_SUFFIXS[fmt]
    except KeyError as msg:  #pragma: no cover
        raise KeyError("Unknown file format %s, options are %s" % (fmt, list(FILE_FORMAT_SUFFIXS.keys()))) from msg

    if istablelike(obj):
        odict = OrderedDict([(DEFAULT_TABLE_KEY[fmt], obj)])
    elif istabledictlike(obj):
        odict = obj
    elif not obj:
        return None
    else:
        raise TypeError("Can not write object of type %s" % type(obj))

    if fType in [ASTROPY_HDF5, NUMPY_HDF5, PANDAS_PARQUET]:
        try:
            nativeTType = NATIVE_TABLE_TYPE[fType]
        except KeyError as msg:  #pragma: no cover
            raise KeyError("Native file type not known for %s" % (fmt)) from msg

        forcedOdict = convert(odict, nativeTType)
        return writeNative(forcedOdict, basename)

    if fType == ASTROPY_FITS:
        forcedOdict = convert(odict, AP_TABLE)
        filepath = "%s.fits" % basename
        writeTablesToFits(forcedOdict, filepath)
        return filepath
    if fType == PANDAS_HDF5:
        forcedOdict = convert(odict, PD_DATAFRAME)
        filepath = "%s.h5" % basename
        writeDataFramesToH5(forcedOdict, filepath)
        return filepath

    raise TypeError("Unsupported File type %i" % fType)  #pragma: no cover
