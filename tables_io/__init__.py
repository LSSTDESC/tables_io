"""tables_io is a library of functions for input, output and conversion of tabular data formats"""

from ._version import version

from .lazy_modules import *

from . import types

from . import arrayUtils

from . import convUtils as conv

from . import ioUtils as io

from .tableDict import TableDict

convertObj = conv.convertObj

convert = conv.convert

writeNative = io.writeNative

write = io.write

readNative = io.readNative

read = io.read

#iterate = io.iterate
