"""IO Functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .arrayUtils import forceToPandables
from .lazy_modules import apTable, fits, pd, pa
from .types import AP_TABLE, NUMPY_DICT, NUMPY_RECARRAY, PD_DATAFRAME, PA_TABLE, istablelike, tableType

### I. Single `Tablelike` conversions

### I A. Converting to `astropy.table.Table`

def paTableToApTable(table):
    """
    Convert a `pyarrow.Table` to an `astropy.table.Table`

    Parameters
    ----------
    table :  `pyarrow.Table`
        The table

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    df = table.to_pandas()
    return dataFrameToApTable(df)


def dataFrameToApTable(df):
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
    o_dict = OrderedDict()
    for colname in df.columns:
        col = df[colname]
        if col.dtype.name == "object":
            o_dict[colname] = np.vstack(col.to_numpy())
        else:
            o_dict[colname] = col.to_numpy()
    tab = apTable.Table(o_dict)
    for k, v in df.attrs.items():
        tab.meta[k] = v  # pragma: no cover
    return tab


def convertToApTable(obj):
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
    if tType == PA_TABLE:
        return paTableToApTable(obj)
    if tType == NUMPY_RECARRAY:
        return apTable.Table(obj)
    if tType == PD_DATAFRAME:
        # try this: apTable.from_pandas(obj)
        return dataFrameToApTable(obj)
    raise TypeError(f"Unsupported TableType {tType}")  # pragma: no cover


### I B. Converting to `OrderedDict`, (`str`, `numpy.array`)


def apTableToDict(tab):
    """
    Convert an `astropy.table.Table` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    tab :  `astropy.table.Table`
        The table

    Returns
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

    Returns
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

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    data = OrderedDict()
    for key in df.keys():
        col = df[key]
        if col.dtype.name == "object":
            data[key] = np.vstack(col.to_numpy())
        else:
            data[key] = np.array(col)
    return data


def paTableToDict(rec):
    """
    Convert an `pa.Table` to an `OrderedDict` of `str` : `numpy.array`

    Parameters
    ----------
    rec :  `pa.Table`
        The input table

    Returns
    --------
    data : `OrderedDict`,  (`str` : `numpy.array`)
        The tabledata
    """
    return OrderedDict([(colName, rec[colName].to_numpy()) for colName in rec.schema.names])


def hdf5GroupToDict(hg):
    """
    Convert a `hdf5` object to an `OrderedDict`, (`str`, `numpy.array`)

    Parameters
    ----------
    hg :  `h5py.File` or `h5py.Group`
        The hdf5 object

    Returns
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
        return apTableToDict(obj)
    if tType == PA_TABLE:
        return paTableToDict(obj)
    if tType == NUMPY_DICT:
        return obj
    if tType == NUMPY_RECARRAY:
        return recarrayToDict(obj)
    if tType == PD_DATAFRAME:
        return dataFrameToDict(obj)
    raise TypeError(f"Unsupported TableType {tType}")  # pragma: no cover


### I C. Converting to `np.recarray`


# TODO
def paTableToRecarray(tab):
    """
    Convert an `pyarrow.Table` to an `numpy.recarray`

    Parameters
    ----------
    tab :  `pyarrow.Table`
        The table

    Returns
    --------
    rec : `numpy.recarray`
        The output rec array
    """
    raise NotImplementedError()  #pragma: no cover


def apTableToRecarray(tab):
    """
    Convert an `astropy.table.Table` to an `numpy.recarray`

    Parameters
    ----------
    tab :  `astropy.table.Table`
        The table

    Returns
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
        return apTableToRecarray(obj)
    if tType == NUMPY_DICT:
        return apTableToRecarray(apTable.Table(obj))
    if tType == NUMPY_RECARRAY:
        return obj
    if tType == PD_DATAFRAME:
        return apTableToRecarray(dataFrameToApTable(obj))
    if tType == PA_TABLE:
        return apTableToRecarray(paTableToApTable(obj))
    raise TypeError(f"Unsupported TableType {tType}")  # pragma: no cover


### I D. Converting to `pandas.DataFrame`


def apTableToDataFrame(tab):
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
    o_dict = OrderedDict()
    for colname in tab.columns:
        col = tab[colname]
        o_dict[colname] = forceToPandables(col.data)
    df = pd.DataFrame(o_dict)
    for k, v in tab.meta.items():
        df.attrs[k] = v  # pragma: no cover
    return df


def paTableToDataFrame(table):
    df = table.to_pandas()
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
    if meta is not None:  # pragma: no cover
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
        return apTableToDataFrame(obj)
    if tType == NUMPY_DICT:
        return dictToDataFrame(obj)
    if tType == NUMPY_RECARRAY:
        odict = recarrayToDict(obj)
        return dictToDataFrame(odict)
    if tType == PA_TABLE:
        return paTableToDataFrame(obj)
    if tType == PD_DATAFRAME:
        return obj
    raise TypeError(f"Unsupported tableType {tType}")  # pragma: no cover


### I E. Converting to `pa.Table`


def apTableToPaTable(tab):
    """
    Convert an `astropy.table.Table` to a `pa.Table`

    Parameters
    ----------
    tab : `astropy.table.Table`
        The table

    Returns
    -------
    table :  `pa.Table`
        The output table
    """
    o_dict = OrderedDict()
    for colname in tab.columns:
        col = tab[colname]
        ndim = len(col.data.shape)
        if ndim == 1:
            o_dict[colname] = col.data
        elif ndim > 1:
            o_dict[colname] = forceToPandables(col.data)

    metadata = {k: str(v) for k, v in tab.meta.items()}
    table = pa.Table.from_pydict(o_dict, metadata=metadata)
    return table


def dataFrameToPaTable(df):
    """
    Convert a `pandas.DataFrame` to an `pa.Table`

    Parameters
    ----------
    df :  `pandas.DataFrame`
        The dataframe

    Returns
    -------
    table : `pa.Table`
        The table
    """
    table = pa.Table.from_pandas(df)
    return table


def dictToPaTable(odict, meta=None):
    """
    Convert an `OrderedDict`, (`str`, `numpy.array`) to a `pa.Table`

    Parameters
    ----------
    odict : `OrderedDict`, (`str`, `numpy.array`)
        The dict

    meta : `dict` or `None`
        Optional dictionary of metadata

    Returns
    -------
    table :  `pa.Table`
        The table
    """
    out_dict = {key: forceToPandables(val) for key, val in odict.items()}
    if meta is not None:  # pragma: no cover
        metadata = {k: str(v) for k, v in meta.items()}
    else:
        metadata = None

    table = pa.Table.from_pydict(out_dict, metadata=metadata)
    return table


def convertToPaTable(obj):
    """
    Convert an object to a `pa.Table`

    Parameters
    ----------
    obj : `object`
       The object being converted

    Returns
    -------
    table :  `pa.Table`
        The table
    """
    tType = tableType(obj)
    if tType == AP_TABLE:
        return apTableToPaTable(obj)
    if tType == NUMPY_DICT:
        return dictToPaTable(obj)
    if tType == NUMPY_RECARRAY:
        odict = recarrayToDict(obj)
        return dictToDataFrame(odict)
    if tType == PD_DATAFRAME:
        return dataFrameToPaTable(obj)
    if tType == PA_TABLE:
        return obj
    raise TypeError(f"Unsupported tableType {tType}")  # pragma: no cover


# I F. Generic `convert`


def convertObj(obj, tType):
    """
    Convert an object to a specific type of `Tablelike`

    Parameters
    ----------
    obj : `object`
       The object being converted
    tType : `int`
       The type of object to convert to, one of `TABULAR_FORMAT_NAMES.keys()`

    Returns
    -------
    out :  `Tablelike`
        The converted object
    """
    if tType == AP_TABLE:
        return convertToApTable(obj)
    if tType == NUMPY_DICT:
        return convertToDict(obj)
    if tType == NUMPY_RECARRAY:
        return convertToRecarray(obj)
    if tType == PA_TABLE:
        return convertToPaTable(obj)
    if tType == PD_DATAFRAME:
        return convertToDataFrame(obj)
    raise TypeError(f"Unsupported tableType {tType}")  # pragma: no cover


### II.  Multi-table conversion utilities


def convertToApTables(odict):
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
    return OrderedDict([(k, convertToApTable(v)) for k, v in odict.items()])


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


def convertToPaTables(odict):
    """
    Convert several `objects` to `pa.Table`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `Tablelike`)
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `np.recarray`
        The tables
    """
    return OrderedDict([(k, convertToPaTable(v)) for k, v in odict.items()])


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

    funcMap = {
        AP_TABLE: convertToApTables,
        NUMPY_DICT: convertToDicts,
        NUMPY_RECARRAY: convertToRecarrays,
        PA_TABLE: convertToPaTables,
        PD_DATAFRAME: convertToDataFrames,
    }
    try:
        theFunc = funcMap[tType]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(f"Unsupported type {tType}") from msg
    return theFunc(obj)
