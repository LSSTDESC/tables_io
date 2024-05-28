"""cli for tables_io.convert"""

from functools import partial  #pragma: no cover
from typing import Any, Type, TypeVar  #pragma: no cover

import click  #pragma: no cover

from . import types
import tables_io


class PartialOption:
    """Wraps click.option with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.option, *param_decls, cls=partial(click.Option), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return self._partial(*args, **kwargs)


class PartialArgument:
    """Wraps click.argument with partial arguments for convenient reuse"""

    def __init__(self, *param_decls: Any, **kwargs: Any) -> None:
        self._partial = partial(
            click.argument, *param_decls, cls=partial(click.Argument), **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
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
def convert(input, output):
    """Convet a file with tabular data from one format to another"""

    input_fname = input
    output_fname = output
    output_format = output_fname.split(".")[-1]

    print(f"Converting {input_fname} to {output_fname}")

    suffixes = types.FILE_FORMAT_SUFFIXS
    suffix = suffixes[output_format]

    t_in = tables_io.read(input_fname)
    t_out = tables_io.convert(t_in, suffix)
    written = tables_io.write(t_out, output_fname)

    print(f"Done converting file")

    return 0


@cli.command()
@output()
@input_args()
def make_index(output, input_args):
    """Make an index file from a list of files"""

    tables_io.createIndexFile(output, input_args)
    
    return 0
