"""Multi-table Convert Functions for tables_io"""

from collections import OrderedDict

import numpy as np
from typing import Mapping

from .conv_table import (
    convert_obj,
    convert_to_ap_table,
    convert_to_dataframe,
    convert_to_dict,
    convert_to_pa_table,
    convert_to_recarray,
)
from ..utils.array_utils import force_to_pandables
from ..lazy_modules import apTable, fits, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    TABULAR_FORMAT_NAMES,
    TABULAR_FORMATS,
    is_table_like,
    table_type,
    tType_to_int,
)


### II.  Multi-table conversion utilities
# TODO: Make tType also accept strings or ints
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
    if is_table_like(obj):
        return convert_obj(obj, tType)

    funcMap = {
        AP_TABLE: convert_to_ap_tables,
        NUMPY_DICT: convert_to_dicts,
        NUMPY_RECARRAY: convert_to_recarrays,
        PA_TABLE: convert_to_pa_tables,
        PD_DATAFRAME: convert_to_dataframes,
    }

    # Convert tType to int if necessary
    int_tType = tType_to_int(tType)

    try:
        theFunc = funcMap[int_tType]
    except KeyError as msg:  # pragma: no cover
        raise KeyError(
            f"Unsupported tabular type {int_tType} ({TABULAR_FORMATS[int_tType]}, must be one of {TABULAR_FORMAT_NAMES})"
        ) from msg
    return theFunc(obj)


def convert_to_ap_tables(odict: Mapping):
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
    return OrderedDict([(k, convert_to_ap_table(v)) for k, v in odict.items()])


def convert_to_dicts(odict: Mapping):
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
    return OrderedDict([(k, convert_to_dict(v)) for k, v in odict.items()])


def convert_to_recarrays(odict: Mapping):
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
    return OrderedDict([(k, convert_to_recarray(v)) for k, v in odict.items()])


def convert_to_pa_tables(odict: Mapping):
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
    return OrderedDict([(k, convert_to_pa_table(v)) for k, v in odict.items()])


def convert_to_dataframes(odict: Mapping):
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
    return OrderedDict([(k, convert_to_dataframe(v)) for k, v in odict.items()])
