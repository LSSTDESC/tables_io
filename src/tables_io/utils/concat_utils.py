"""Concatanation functions for tables_io"""

from collections import OrderedDict
from typing import Union, Optional, List, Mapping

import numpy as np
from numpy.lib import recfunctions as rfn

from .array_utils import concatenate_dicts
from ..lazy_modules import apTable, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    tType_to_int,
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
    int_tType = tType_to_int(tType)

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
    Vertically concatenates a list of `TableDict-like` objects. Each `Tablelike` object
    in a `TableDict-like` object will be concatenated with any matching `Tablelike` objects
    in the other `TableDict-like` objects (where matching means they have the same key). The
    final `TableDict-like` object will contain all unique `Tablelike` objects (those with unique
    keys).

    The concatenation will be of join type `outer`, which means that no data will be lost.

    Note: If concatenating `NUMPY_RECARRAY` objects, the output arrays will be masked arrays if
    any fill values are required by the concatenation.

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


    Example
    -------
    ```
    import tables_io
    tabledict_1 = {'data1': data, 'data2': data}
    tabledict_2 = {'data2': data, 'data3': data}
    tables_io.concat([tabledict1, tabledict_2],)
    ```
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
        The list of tables

    Returns
    -------
    tab : `astropy.table.Table`
        The concatenated table
    """
    return apTable.vstack(tablelist, join_type="outer")


### II B. Concatanating dicts of numpy arrays
def concat_numpy_dicts(tablelist: List):
    """
    Concatanate a list of `dicts` of `np.array`

    Parameters
    ----------
    tablelist :  `list`
        The list of tables

    Returns
    -------
    tab : `dict`
        The concatenated table
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
    return rfn.stack_arrays(tablelist)


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
    tab : `pandas.DataFrame`
        The concatenated table
    """
    return pd.concat(tablelist, join="outer", axis=0, ignore_index=True)


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
    tab : `pyarrow.Table`
        The concatenated table
    """
    return pa.concat_tables(tablelist)
