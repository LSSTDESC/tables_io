"""tables_io is a library of functions for input, output and conversion of tabular data formats"""

try:
    from ._version import version
except:  # pylint: disable=bare-except   #pragma: no cover
    version = "unknown"

from .lazy_modules import *



from . import convUtils as conv

from . import ioUtils as io

from . import concatUtils

from . import sliceUtils


convertObj = conv.convertObj

convert = conv.convert

writeNative = io.writeNative

write = io.write

readNative = io.readNative

read = io.read

io_open = io.io_open

iteratorNative = io.iteratorNative

iterator = io.iterator

concatObjs = concatUtils.concatObjs

concat = concatUtils.concat

sliceObj = sliceUtils.sliceObj

sliceObjs = sliceUtils.sliceObjs
