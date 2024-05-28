"""IO Functions for tables_io"""

from collections import OrderedDict

import numpy as np

from .arrayUtils import forceToPandables
from .lazy_modules import apTable, fits, pd, pa
from .types import AP_TABLE, NUMPY_DICT, NUMPY_RECARRAY, PD_DATAFRAME, PA_TABLE, istablelike, tableType


### I. merging list of table-like objects

### I A. Merging `astropy.table.Table`

def mergeApTables(tablelist):
    """
    Merge a list of `astropy.table.Table`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `astropy.table.Table`
        The table
    """

    
### I B. Merging dicts of numpy arrays
def mergeNumpyDicts(tablelist):
    """
    Merge a list of `dicts` of `np.array`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """


### I C. Merging numpy recarrays
def mergeNumpyRecarrays(tablelist):
    """
    Merge a list of `dicts` of `np.recarray`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """


### I D. Merging pandas dataframes
def mergeDataframes(tablelist):
    """
    Merge a list of `pandas.DataFrame`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """


### I E. Merging pyarrow tables
def mergePATables(tablelist):
    """
    Merge a list of `pyarrow.Table`

    Parameters
    ----------
    tablelist :  `list`
        The tables

    Returns
    -------
    tab : `dict`
        The table
    """


# I F. Generic `merge`
def mergeObjs(tableList, tType):
    """
     Merge a list of `table-like` objects

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
    pass
    

### II.  Multi-table mergingg

def merge(odict):
    """
    Convert several `objects` to `astropy.table.Table`

    Parameters
    ----------
    odict :  `Mapping`, (`str`, `list' (`Tablelike`) )
        The input objects

    Returns
    -------
    tabs : `OrderedDict` of `table-like`
        The tables
    """
    return OrderedDict([(k, merge(v)) for k, v in odict.items()])

