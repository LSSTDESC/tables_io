"""tables_io is a library of functions for input, output and conversion of tabular data formats"""

try:
    from ._version import version
except:  # pylint: disable=bare-except   #pragma: no cover
    version = "unknown"

from .lazy_modules import *
from .table_dict import TableDict
from . import convert as conv
from . import io_utils
from .utils import concat_utils
from .utils import slice_utils

# Exposing the primary functions and interfaces for tables_io


#
# Conversion Functions
#

convert_table = conv.conv_table.convert_table
convertObj = conv.conv_table.convert_table
"""This function is being deprecated, please see `convert_table` instead"""
convert = conv.conv_tabledict.convert

#
# Write Functions
#

writeNative = io_utils.write.write_native
"""This function is being deprecated, please see `write_native` instead"""
write_native = io_utils.write.write_native
write = io_utils.write.write


#
# Read Functions
#

readNative = io_utils.read.read_native
"""This function is being deprecated, please see `read_native` instead"""
read_native = io_utils.read.read_native
read = io_utils.read.read
io_open = io_utils.read.io_open
check_columns = io_utils.read.check_columns

#
# Iteration Functions
#

iterator_native = io_utils.iterator.iterator_native
iteratorNative = io_utils.iterator.iterator_native
"""This function is being deprecated, please see `iterator_native` instead"""
iterator = io_utils.iterator.iterator

#
# Concatenation Functions
#

concat_table = concat_utils.concat_table
concatObjs = concat_utils.concat_table
"""This function is being deprecated, please see `concat_table` instead"""
concat = concat_utils.concat_tabledict
"""This function is being deprecated, please see `concat_tabledict` instead"""
concat_tabledict = concat_utils.concat_tabledict


#
# Slicing Functions
#

slice_table = slice_utils.slice_table
sliceObj = slice_utils.slice_table
"""This function is being deprecated, please see `slice_table` instead"""
sliceObjs = slice_utils.slice_tabledict
"""This function is being deprecated, please see `slice_table` instead"""
slice_tabledict = slice_utils.slice_tabledict
