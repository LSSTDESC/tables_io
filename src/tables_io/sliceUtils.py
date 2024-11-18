"""Slicing functions for tables_io"""

from collections import OrderedDict


from .arrayUtils import sliceDict
from .types import NUMPY_DICT, tableType


# I F. Generic `slice`
def sliceObj(obj, the_slice):
    """
    Slice a `table-like` objects

    Parameters
    ----------
    obj :  `table_like`
        Table like object to slice

    the_slice: `slice` 
        Slice to make

    Returns
    -------
    tab : `table-like`
        The slice of the table
    """
    tType = tableType(obj)
    if tType is NUMPY_DICT:
        return sliceDict(obj, the_slice)
    return obj[the_slice]


def sliceObjs(odict, the_slice):
    """Slice many `table-like` objects

    Parameters
    ----------
    odict :  `table_like`
       Objects to slice

    the_slice: `slice` 
        Slice to make


    Returns
    -------
    odict : tableDict-like
        The sliced tables
    """
    return OrderedDict([(k, sliceObj(v, the_slice)) for k, v in odict.items()])

