"""IO Functions for tables_io"""

import os
from collections import OrderedDict

import numpy as np

from ..utils.arrayUtils import getGroupInputDataLength, forceToPandables
from ..convert.convUtils import convert, dataFrameToDict, hdf5GroupToDict
from ..lazy_modules import apTable, fits, h5py, pa, pd, pq, ds
from ..types import (
    AP_TABLE,
    ASTROPY_FITS,
    ASTROPY_HDF5,
    DEFAULT_TABLE_KEY,
    FILE_FORMAT_SUFFIX_MAP,
    FILE_FORMAT_SUFFIXS,
    NATIVE_FORMAT,
    NATIVE_TABLE_TYPE,
    NUMPY_FITS,
    NUMPY_HDF5,
    PA_TABLE,
    PANDAS_HDF5,
    PANDAS_PARQUET,
    PYARROW_HDF5,
    PYARROW_PARQUET,
    PD_DATAFRAME,
    fileType,
    istabledictlike,
    istablelike,
    tableType,
)

### I. Iteration functions


### I A. HDF5 partial read/write functions


### I B. Parquet partial read/write functions


### I C. Parquet dataset partial read/write functions


### II.   Reading and Writing Files

### II A.  Reading and Writing `astropy.table.Table` to/from FITS files


### II A'.  Reading and Writing `` to/from FITS files


### II B.  Reading and Writing `astropy.table.Table` to/from `hdf5`


### II C.  Reading and Writing `OrderedDict`, (`str`, `numpy.array`) to/from `hdf5`


### II C.  Reading and Writing `pandas.DataFrame` to/from `hdf5`


### II E.  Reading and Writing `pandas.DataFrame` to/from `parquet`


### II F.  Reading and Writing to `OrderedDict`, (`str`, `numpy.array`)


### II G.  Reading and Writing `pandas.DataFrame` to/from `hdf5`


### II H.  Reading and Writing `pyarrow.Table` to/from `parquet`


### II I.  Top-level interface functions
