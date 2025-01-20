"""Multi-table Convert Functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .convSingle import (
    convertObj,
    convertToApTable,
    convertToDataFrame,
    convertToDict,
    convertToPaTable,
    convertToRecarray,
)
from ..utils.arrayUtils import forceToPandables
from ..lazy_modules import apTable, fits, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    istablelike,
    tableType,
)


### II.  Multi-table conversion utilities
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
