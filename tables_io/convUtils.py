"""IO Functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .lazy_modules import pd, apTable, fits
from .lazy_modules import HAS_ASTROPY, HAS_PANDAS

from .arrayUtils import forceToPandables

from .types import AP_TABLE, NUMPY_DICT, NUMPY_RECARRAY, PD_DATAFRAME, tableType, istablelike


### I. Single `Tablelike` conversions

### I A. Converting to `astropy.table.Table`

def dataFrameToTable(df):
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
    tab = apTable.Table(o_dict)
    for k, v in df.attrs.items():
        tab.meta[k] = v  #pragma: no cover
    return tab


def convertToTable(obj):
    """
    Convert an object to an `astropy.table.Table`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    tType = tableType(obj)
    if tType == AP_TABLE:
        return obj
    if tType == NUMPY_DICT:
        return apTable.Table(obj)
    if tType == NUMPY_RECARRAY:
        return apTable.Table(obj)
    if tType == PD_DATAFRAME:
        # try this: apTable.from_pandas(obj)
        return dataFrameToTable(obj)
    raise TypeError("Unsupported TableType %i" % tType)  #pragma: no cover



### I B. Converting to `OrderedDict`, (`str`, `numpy.array`)

def tableToDict(tab):
    """
    Convert an `astropy.table.Table` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    tab :  `astropy.table.Table`
        The table

    Returnes
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key, val in zip(tab.colnames, tab.itercols()):
        data[key] = np.array(val)
    return data


def recarrayToDict(rec):
    """
    Convert an `np.recarray` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    rec :  `np.recarray`
        The input recarray

    Returnes
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    return OrderedDict([(colName, rec[colName]) for colName in rec.dtype.names])


def dataFrameToDict(df):
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
        col = df[key]
        if col.dtype.name == 'object':
            data[key] = np.vstack(col.to_numpy())
        else:
            data[key] = np.array(col)
    return data


def hdf5GroupToDict(hg):
    """
    Convert a `hdf5` object to an `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    hg :  `h5py.File` or `h5py.Group`
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


def convertToDict(obj):
    """
    Convert an object to an `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    tType = tableType(obj)
    if tType == AP_TABLE:
        return tableToDict(obj)
    if tType == NUMPY_DICT:
        return obj
    if tType == NUMPY_RECARRAY:
        return recarrayToDict(obj)
    if tType == PD_DATAFRAME:
        return dataFrameToDict(obj)
    raise TypeError("Unsupported TableType %i" % tType)  #pragma: no cover



### I C. Converting to `np.recarray`

def tableToRecarray(tab):
    """
    Convert an `astropy.table.Table` to an `numpy.recarray`

    Parameters
    ----------
    tab :  `astropy.table.Table`
        The table

    Returnes
    --------
    rec : `numpy.recarray`
        The output rec array
    """
    return fits.table_to_hdu(tab).data


def convertToRecarray(obj):
    """
    Convert an object to an `numpy.recarray`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    rec : `numpy.recarray`
        The output recarray
    """
    tType = tableType(obj)
    if tType == AP_TABLE:
        return tableToRecarray(obj)
    if tType == NUMPY_DICT:
        return tableToRecarray(apTable.Table(obj))
    if tType == NUMPY_RECARRAY:
        return obj
    if tType == PD_DATAFRAME:
        return tableToRecarray(dataFrameToTable(obj))
    raise TypeError("Unsupported TableType %i" % tType)  #pragma: no cover



### I D. Converting to `pandas.DataFrame`

def tableToDataFrame(tab):
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


def dictToDataFrame(odict, meta=None):
    """
    Convert an `OrderedDict`, (`str`, `numpy.array`) to a `pandas.DataFrame`

    Parameters
    ----------
    odict : `OrderedDict`, (`str`, `numpy.array`)
        The dict

    meta : `dict` or `None`
        Optional dictionary of metadata

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    outdict = OrderedDict()
    for k, v in odict.items():
        outdict[k] = forceToPandables(v)
    df = pd.DataFrame(outdict)
    if meta is not None:  #pragma: no cover
        for k, v in meta.items():
            df.attrs[k] = v
    return df


def convertToDataFrame(obj):
    """
    Convert an object to a `pandas.DataFrame`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    df :  `pandas.DataFrame`
        The dataframe
    """
    tType = tableType(obj)
    if tType == AP_TABLE:
        return tableToDataFrame(obj)
    if tType == NUMPY_DICT:
        return dictToDataFrame(obj)
    if tType == NUMPY_RECARRAY:
        odict = recarrayToDict(obj)
        return dictToDataFrame(odict)
    if tType == PD_DATAFRAME:
        return obj
    raise TypeError("Unsupported tableType %i" % tType)  #pragma: no cover


# I E. Generic `convert`

def convertObj(obj, tType):
    """
    Convert an object to a specific type of `Tablelike`

    Parameters
    ----------
    obj : `object`
       The object being converted
    tType : `int`
       The type of object to convert to, one of `TABULAR_FORMAT_NAMES`

    Returns
    -------
    out :  `Tablelike`
        The converted object
    """
    if tType == AP_TABLE:
        return convertToTable(obj)
    if tType == NUMPY_DICT:
        return convertToDict(obj)
    if tType == NUMPY_RECARRAY:
        return convertToRecarray(obj)
    if tType == PD_DATAFRAME:
        return convertToDataFrame(obj)
    raise TypeError("Unsupported tableType %i" % tType)  #pragma: no cover



### II.  Multi-table conversion utilities

def convertToTables(odict):
    """
    Convert several `objects` to `astropy.table.Table`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Tablelike`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `astropy.table.Table`
        The tables
    """
    return OrderedDict([(k, convertToTable(v)) for k, v in odict.items()])


def convertToDicts(odict):
    """
    Convert several `objects` to `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Tablelike`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `OrderedDict`, (`str`, `numpy.array`)
        The tables
    """
    return OrderedDict([(k, convertToDict(v)) for k, v in odict.items()])


def convertToRecarrays(odict):
    """
    Convert several `objects` to `np.recarray`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Tablelike`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `np.recarray`
        The tables
    """
    return OrderedDict([(k, convertToRecarray(v)) for k, v in odict.items()])


def convertToDataFrames(odict):
    """
    Convert several `objects` to `pandas.DataFrame`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Tablelike`)
        The input objects

    Returns
    -------
    df :  `OrderedDict` of `pandas.DataFrame`
        The dataframes
    """
    return OrderedDict([(k, convertToDataFrame(v)) for k, v in odict.items()])


def convert(obj, tType):
    """
    Convert several `objects` to a specific type

    Parameters
    ----------
    obj :  'Tablelike` or `TableDictlike`
        The input object

    tType : `int`
        One of `TABULAR_FORMAT_NAMES.keys()`

    Returns
    -------
    out :  `Tablelike` or `TableDictlike`
        The converted data
    """
    if istablelike(obj):
        return convertObj(obj, tType)

    funcMap = {AP_TABLE:convertToTables,
               NUMPY_DICT:convertToDicts,
               NUMPY_RECARRAY:convertToRecarrays,
               PD_DATAFRAME:convertToDataFrames}
    try:
        theFunc = funcMap[tType]
    except KeyError as msg:  #pragma: no cover
        raise KeyError("Unsupported type %i" % tType) from msg
    return theFunc(obj)
