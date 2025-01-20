"""IO Functions for tables_io"""

from collections import OrderedDict

import numpy as np

from ..utils.arrayUtils import forceToPandables
from ..lazy_modules import apTable, fits, pd, pa
from ..types import (
    AP_TABLE,
    NUMPY_DICT,
    NUMPY_RECARRAY,
    PD_DATAFRAME,
    PA_TABLE,
    istablelike,
    tableType,
)
