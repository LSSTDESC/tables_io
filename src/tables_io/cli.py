"""cli for tables_io.convert"""

import argparse

import tables_io


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help=f"input filename; suffix should be one of {list(tables_io.types.FILE_FORMAT_SUFFIXS.keys())}",
    )
    parser.add_argument(
        "output",
        type=str,
        help=f"output filename; suffix should be one of {list(tables_io.types.FILE_FORMAT_SUFFIXS.keys())}",
    )
    return parser.parse_args()


def main():
    args = get_args()

    input_fname = args.input
    output_fname = args.output
    output_format = output_fname.split(".")[-1]

    print(f"Converting {input_fname} to {output_fname}")

    suffixes = tables_io.types.FILE_FORMAT_SUFFIXS
    suffix = suffixes[output_format]

    t_in = tables_io.read(input_fname)
    t_out = tables_io.convert(t_in, suffix)
    written = tables_io.write(t_out, output_fname)

    print(f"Done converting file")

    return 0
