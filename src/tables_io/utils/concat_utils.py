"""Concatanation functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .array_utils import concatenate_dicts
from ..lazy_modules import apTable, pd, pa
from ..types import AP_TABLE, NUMPY_DICT, NUMPY_RECARRAY, PD_DATAFRAME, PA_TABLE


### I. concatanating list of table-like objects


# I A. Generic `concat`
def concat_objs(tableList, tType):
    """
    Concatanate a list of `table-like` objects

    Parameters
    ----------
    tablelist :  `list`
        The tables

    tType:
        What type of tables we expect the objects to be

    Returns
    -------
    tab : `dict`
        The table
    """
    funcDict = {
        AP_TABLE: concat_ap_tables,
        NUMPY_DICT: concat_numpy_dicts,
        NUMPY_RECARRAY: concat_numpy_recarrays,
        PD_DATAFRAME: concat_dataframes,
        PA_TABLE: concat_pa_tables,
    }

    try:
        theFunc = funcDict[tType]
        return theFunc(tableList)
    except KeyError as msg:  # pragma: no cover
        raise NotImplementedError(
            f"Unsupported FileType for concatObjs {tType}"
        ) from msg  # pragma: no cover


### I B.  Multi-table concatanating


def concat(odictlist, tType):
    """
    Concatanate all the tables in a list of dicts

    Parameters
    ----------
    odictlist :  `list`, 'tableDict-like'
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `table-like`
        The tables
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


def concat_ap_tables(tablelist):
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
def concat_numpy_dicts(tablelist):
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
def concat_numpy_recarrays(tablelist):
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
def concat_dataframes(tablelist):
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
def concat_pa_tables(tablelist):
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
