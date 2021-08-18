"""IO Functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .lazy_modules import pd, apTable
from .lazy_modules import HAS_ASTROPY, HAS_PANDAS

from .arrayUtils import forceToPandables

from .types import AP_TABLE, NUMPY_DICT, PD_DATAFRAME, tableType


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


def forceToTable(obj):
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
    if tType == PD_DATAFRAME:
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


def forceToDict(obj):
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
    if tType == PD_DATAFRAME:
        return dataFrameToDict(obj)
    raise TypeError("Unsupported TableType %i" % tType)  #pragma: no cover



### I C. Converting to `pandas.DataFrame`

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
    if not HAS_PANDAS:  #pragma: no cover
        raise ImportError("pandas is not available, can't make DataFrame")

    outdict = OrderedDict()
    for k, v in odict.items():
        outdict[k] = forceToPandables(v)
    df = pd.DataFrame(outdict)
    if meta is not None:  #pragma: no cover
        for k, v in meta.items():
            df.attrs[k] = v
    return df


def forceToDataFrame(obj):
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
    if tType == PD_DATAFRAME:
        return obj
    raise TypeError("Unsupported tableType %i" % tType)  #pragma: no cover


# I D. Generic `forceTo`

def forceObjTo(obj, tType):
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
        return forceToTable(obj)
    if tType == NUMPY_DICT:
        return forceToDict(obj)
    if tType == PD_DATAFRAME:
        return forceToDataFrame(obj)
    raise TypeError("Unsupported tableType %i" % tType)  #pragma: no cover



### II.  Multi-table conversion utilities

def forceToTables(odict):
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
    return OrderedDict([(k, forceToTable(v)) for k, v in odict.items()])


def forceToDicts(odict):
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
    return OrderedDict([(k, forceToDict(v)) for k, v in odict.items()])


def forceToDataFrames(odict):
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
    return OrderedDict([(k, forceToDataFrame(v)) for k, v in odict.items()])


def forceTo(odict, tType):
    """
    Convert several `objects` to a specific type

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Tablelike`)
        The input objects

    tType : `int`
        One of `TABULAR_FORMAT_NAMES.keys()`

    Returns
    -------
    out :  `OrderedDict`, (`str`, `Tablelike`)
        The converted data
    """
    funcMap = {AP_TABLE:forceToTables,
               NUMPY_DICT:forceToDicts,
               PD_DATAFRAME:forceToDataFrames}
    try:
        theFunc = funcMap[tType]
    except KeyError as msg:  #pragma: no cover
        raise KeyError("Unsupported type %i" % tType) from msg
    return theFunc(odict)
