"""Functions to store analysis results as astropy data tables  """

import os

from collections import OrderedDict
from collections.abc import Mapping

from .lazy_modules import pd, apTable
from .lazy_modules import HAS_ASTROPY, HAS_PANDAS


class TableDict(OrderedDict):
    """Object to collect various types of table-like objects

    This class is a dictionary mapping name to table-like
    and a few helper functions, e.g., to add new tables to the dictionary
    and to read and write files, either as FITS or HDF5 files.
    """
    legal_tabletypes = None
    
    @classmethod
    def is_table(cls, tab):
        """ Check to set if an item is something we can treat as a table 

        Parameters
        ----------
        tab : `?`
            The item in question
        
        Raises
        ------
        TypeError if it isn't a table.
        """
        if cls.legal_tabletypes is None:
            cls.legal_tabletypes = []
            if HAS_ASTROPY:
                cls.legal_tabletypes.append(apTable.Table)
            if HAS_PANDAS:
                cls.legal_tabletypes.append(pd.DataFrame)
        
        if isinstance(tab, legal_tabletypes):
            return
        if not isinstance(tab, Mapping):
            raise TypeError("%s is not a DataFrame, not a table and not Mapping" % type(tab))
        for key, val in tab.items():
            if not isinstance(val, (np.array, Iterable)):
                raise TypeError("%s -> %s is not an array and not Iterable" % (key, type(val)))

    
    def __setitem__(self, key, value):

        try:
            is_table(value)
        except TypeError as msg:
            raise TypeError("item %s was not recognized as a table.") from ms
        
        return OrderedDict.__setitem(self, key, value)
    

    def save_datatables(self, filepath, **kwargs):
        """Save all of the `Table` objects in this object to a file

        Parameters
        ----------
        filepath : `str`
            The file to save it to
        kwargs
            Passed to write functions

        Raises
        ------
        ValueError : If the output file type is not known.
        """
        extype = os.path.splitext(filepath)[1]
        if extype in HDF5_SUFFIXS:
            for key, val in self._table_dict.items():
                val.write(filepath, path=key, **kwargs)
        elif extype in FITS_SUFFIXS:
            if self._primary is None:
                hlist = [fits.PrimaryHDU()]
            else:
                hlist = [self._primary]
            for key, val in self._table_dict.items():
                hdu = fits.table_to_hdu(val)
                hdu.name = key
                hlist.append(hdu)
            hdulist = fits.HDUList(hlist)
            hdulist.writeto(filepath, overwrite=True, **kwargs)
        else:
            raise ValueError("Can only write pickle and hdf5 files for now, not %s" % extype)


    def load_datatables(self, filepath, **kwargs):
        """Read a set of `Table` objects from a file into this object

        Parameters
        ----------
        filepath : `str`
            The file to read
        kwargs
            Passed to reade functions

        Raises
        ------
        ValueError : If the input file type is not known.
        """
        extype = os.path.splitext(filepath)[1]
        tablelist = kwargs.get('tablelist', None)
        if extype in HDF5_SUFFIXS:
            hdffile = h5py.File(filepath)
            keys = hdffile.keys()
            for key in keys:
                if tablelist is None or key in tablelist:
                    self._table_dict[key] = Table.read(filepath, key, **kwargs)
        elif extype in FITS_SUFFIXS:
            hdulist = fits.open(filepath)
            for hdu in hdulist[1:]:
                if tablelist is None or hdu.name.lower() in tablelist:
                    self._table_dict[hdu.name.lower()] = Table.read(filepath, hdu=hdu.name)
        else:
            raise ValueError("Can only read pickle and hdf5 files for now, not %s" % extype)

