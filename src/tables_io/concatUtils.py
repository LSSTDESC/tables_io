"""Concatanation functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .arrayUtils import forceToPandables, concatenateDicts
from .lazy_modules import apTable, fits, pd, pa
from .types import AP_TABLE, NUMPY_DICT, NUMPY_RECARRAY, PD_DATAFRAME, PA_TABLE, istablelike, tableType


### I. concatanating list of table-like objects

### I A. Concatanating `astropy.table.Table`

def concatApTables(tablelist):
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

    
### I B. Concatanating dicts of numpy arrays
def concatNumpyDicts(tablelist):
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
    return concatenateDicts(tablelist)


### I C. Concatanating numpy recarrays
def concatNumpyRecarrays(tablelist):
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
    

### I D. Concatanating pandas dataframes
def concatDataframes(tablelist):
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


### I E. Concatanating pyarrow tables
def concatPATables(tablelist):
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


# I F. Generic `concat`
def concatObjs(tableList, tType):
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
        AP_TABLE:concatApTables,
        NUMPY_DICT:concatNumpyDicts,
        NUMPY_RECARRAY:concatNumpyRecarrays,
        PD_DATAFRAME:concatDataframes,
        PA_TABLE:concatPATables,
    }
    
    try:
        theFunc = funcDict[tType]
        return theFunc(tableList)
    except KeyError as msg:
        raise NotImplementedError(
            f"Unsupported FileType for concatObjs {fType}"
        ) from msg  # pragma: no cover



### II.  Multi-table concatanating

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
    
    return OrderedDict([(k, concatObjs(v, tType)) for k, v in odict_in.items()])

