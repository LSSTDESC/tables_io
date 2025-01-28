"""Concatanation functions for tables_io"""

from collections import OrderedDict
from typing import Union, Optional, List, Mapping

import numpy as np

from .array_utils import concatenate_dicts
from ..lazy_modules import apTable, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    TABULAR_FORMAT_NAMES,
)


### I. concatanating list of table-like objects


# I A. Generic `concat`
def concat_objs(tableList: List, tType: Union[str, int]):
    """
    Concatanate a list of `table-like` objects

    Parameters
    ----------
    tablelist :  `list`
        The tables

    tType: `str` or `int`
        The tabular format of the tables given.

    Returns
    -------
    tab : `Tablelike`
        The concatenated table
    """
    funcDict = {
        AP_TABLE: concat_ap_tables,
        NUMPY_DICT: concat_numpy_dicts,
        NUMPY_RECARRAY: concat_numpy_recarrays,
        PD_DATAFRAME: concat_dataframes,
        PA_TABLE: concat_pa_tables,
    }

    # convert tType to int if necessary
    if isinstance(tType, str):
        try:
            int_tType = TABULAR_FORMAT_NAMES[tType]
        except:
            raise TypeError(
                f"Unsupported tableType '{tType}', must be one of {TABULAR_FORMAT_NAMES.keys()}"
            )
    if isinstance(tType, int):
        int_tType = tType

    try:
        theFunc = funcDict[int_tType]
        return theFunc(tableList)
    except KeyError as msg:  # pragma: no cover
        raise NotImplementedError(
            f"Unsupported FileType for concatObjs {tType}"
        ) from msg  # pragma: no cover


### I B.  Multi-table concatanating


def concat(odictlist: List[Mapping], tType: Union[str, int]) -> Mapping:
    """
    Vertically concatanates a list of `TableDict-like` objects.

    Parameters
    ----------
    odictlist :  `list`, 'TableDict-like'
        The input objects
    tType: `str` or `int`
        The tabular format of the tables given.

    Returns
    -------
    tabs : `OrderedDict` of `table-like`
        A `TableDict-like` object of the concatenated `Tablelike` objects
    """
    odict_in = OrderedDict()
    first = True
    for odict_ in odictlist:
        for key, val in odict_.items():
            if first:
                odict_in[key] = [val]
            else:
                odict_in[key].append(val)
        first = False

    return OrderedDict([(k, concat_objs(v, tType)) for k, v in odict_in.items()])


### II. Concatenating specific data tables
### II A. Concatanating `astropy.table.Table`


def concat_ap_tables(tablelist: List):
    """
    Concatanate a list of `astropy.table.Table`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """
    return apTable.vstack(tablelist)


### II B. Concatanating dicts of numpy arrays
def concat_numpy_dicts(tablelist: List):
    """
    Concatanate a list of `dicts` of `np.array`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """
    return concatenate_dicts(tablelist)


### II C. Concatanating numpy recarrays
def concat_numpy_recarrays(tablelist: List):
    """
    Concatanate a list of `dicts` of `np.recarray`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """
    return np.lib.recfunctions.stack_arrays(tablelist)


### II D. Concatanating pandas dataframes
def concat_dataframes(tablelist: List):
    """
    Concatanate a list of `pandas.DataFrame`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """
    return pd.concat(tablelist)


### II E. Concatanating pyarrow tables
def concat_pa_tables(tablelist: List):
    """
    Concatanate a list of `pyarrow.Table`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """
    return pa.concat_tables(tablelist)
