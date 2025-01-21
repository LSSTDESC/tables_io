"""tables_io is a library of functions for input, output and conversion of tabular data formats"""

try:
    from ._version import version
except:  # pylint: disable=bare-except   #pragma: no cover
    version = "unknown"

from .lazy_modules import *

from .tableDict import TableDict

from . import convert as conv

from . import io_utils

from .utils import concatUtils

from .utils import sliceUtils

# Exposing the primary functions and interfaces for tables_io

convertObj = conv.convertObj

convert = conv.convert

writeNative = io_utils.write_native
"""This function is being deprecated, please see `write_native` instead"""

write_native = io_utils.write_native

write = io_utils.write

readNative = io_utils.read_native
"""This function is being deprecated, please see `read_native` instead"""

read_native = io_utils.read_native

read = io_utils.read

io_open = io_utils.io_open

iteratorNative = io_utils.iteratorNative

iterator = io_utils.iterator

concatObjs = concatUtils.concatObjs

concat = concatUtils.concat

sliceObj = sliceUtils.sliceObj

sliceObjs = sliceUtils.sliceObjs

check_columns = io_utils.check_columns
