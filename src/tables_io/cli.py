"""cli for tables_io.convert"""

import os
from functools import partial
from typing import Any

import click

from . import types
import tables_io


class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.option, *param_decls, cls=partial(click.Option), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.argument, *param_decls, cls=partial(click.Argument), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._partial(*args, **kwargs)


input = PartialOption(
    "--input",
    type=click.Path(),
    help=f"input filename; suffix should be one of {list(types.FILE_FORMAT_SUFFIXS.keys())}",
)

output = PartialOption(
    "--output",
    type=click.Path(),
    help=f"output filename; suffix should be one of {list(types.FILE_FORMAT_SUFFIXS.keys())}",
)

input_args = PartialArgument(
    "input_args",
    nargs=-1,
)


@click.group()
@click.version_option(tables_io._version)
def cli() -> None:
    """tables_io utility scripts"""


@cli.command()
@input()
@output()
def convert(input, output):  # pragma: no cover
    """Convert a file with tabular data from one format to another"""

    input_fname = input
    output_fname = output
    output_format = output_fname.split(".")[-1]

    print(f"Converting {input_fname} to {output_fname}")

    suffixes = types.FILE_FORMAT_SUFFIXS
    suffix = suffixes[output_format]

    t_in = tables_io.read(input_fname)
    t_out = tables_io.convert(t_in, suffix)
    _written = tables_io.write(t_out, output_fname)

    print("Done converting file")

    return 0


@cli.command()
@input_args()
@output(help="output filename; suffix should be 'idx'")
def make_index(input_args, output):  # pragma: no cover
    """Make an index file from a list of input files"""
    tables_io.createIndexFile(output, input_args)
    return 0


@cli.command()
@input_args()
@output()
def concat(input_args, output):  # pragma: no cover
    """Make an index file from a list of input files"""
    suffix = os.path.splitext(output)[1][1:]
    fType = types.FILE_FORMAT_SUFFIXS[suffix]
    tType = types.NATIVE_TABLE_TYPE[fType]
    
    odictlist = [tables_io.read(input_) for input_ in input_args]
    out_dict = tables_io.concat(odictlist, tType)
    tables_io.write(out_dict, output)
    return 0

