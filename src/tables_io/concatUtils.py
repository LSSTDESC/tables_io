"""Concatanation functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .arrayUtils import concatenateDicts
from .lazy_modules import apTable, pd, pa
from .types import AP_TABLE, NUMPY_DICT, NUMPY_RECARRAY, PD_DATAFRAME, PA_TABLE, tableType, istablelike, istabledictlike


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
    if not tableList:  # pragma: no cover
        return OrderedDict()

    funcDict = {
        AP_TABLE:concatApTables,
        NUMPY_DICT:concatNumpyDicts,
        NUMPY_RECARRAY:concatNumpyRecarrays,
        PD_DATAFRAME:concatDataframes,
        PA_TABLE:concatPATables,
    }
        
    if tType is None:
        firstOdict = tableList[0]
        try:
            tType = tableType(firstOdict)
        except TypeError:  # pragma: no cover
            firstTable = list(firstOdict.values())[0]
            tType = tableType(firstTable)
        
    try:
        theFunc = funcDict[tType]
    except KeyError as msg:  # pragma: no cover
        raise NotImplementedError(
            f"Unsupported tableType for concatObjs {tType}"
        ) from msg  # pragma: no cover
    return theFunc(tableList)



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
    if not odictlist:  # pragma: no cover
        return OrderedDict()
    first = odictlist[0]
    if not istabledictlike(first):  # pragma: no cover
        if not istablelike(first):
            raise TypeError(f"odictlist is of {type(first)}, and not tablelike or tabledictlike")
        else:
            return concatObjs(odictlist, tType)
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

