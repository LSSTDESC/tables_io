# Style Guide

## Documentation

All documentation is created using Sphinx. This lives in the `docs` folder, and the output is created in the `_readthedocs` folder in the main package. Documentation files are written in Markdown.

### Writing Documentation Pages

This is what you should do

### Building the Documentation Locally

To build the documentation locally, for instance, for preview or to manually upload to a specified location, run the following from the `docs/` directory with the appropriate conda environment activated:

```bash

# Remove any previous compiled documentation
make clean

# Make the HTML Documentation
make html

```

The HTML rendered documentation will live in the `_readthedocs/html` directory.

## Understanding the code

`tables_io` is organized such that the main functionalities are made available in the main `__init__.py`, and additionally `HDF5` functionalities are made available in the `hdf5` module. The diagram below displays the main functions available and some of the most relevant `HDF5` functions. For all of the available `HDF5` functions, see the function overview of the <project:functionoverview.md#hdf5-module>.

![code diagram](assets/tables_io_model.svg)

The code itself is split into three main components: `io_utils`, `convert`, and `utils`. `io_utils` contains all of the read and write functions. `convert` contains all conversion functions, and `utils` contains all functions related to `slice` and `concat` functionality, as well as code array utilities. `types.py` contains all the dictionaries relating table types and formats, as well as the functions that get information about what type a file or object falls into. `lazy_modules.py` handles loading in the necessary packages. See the diagram below to get a sense of the layout of the package.

```bash
tables_io
├── __init__.py
├── _version.py
├── cli.py
├── conv #conversion functions
│   ├── __init__.py
│   ├── conv_table.py
│   └── conv_tabledict.py
├── hdf5 #loads functions for chunked reads and writes of HDF5 files
│   ├── __init__.py
├── io_utils #io functions
│   ├── __init__.py
│   ├── iterator.py
│   ├── read.py
│   └── write.py
├── lazy_modules.py #handles loading of modules not required by installation
├── table_dict.py #deprecated
├── types.py #table types and format dictionaries and functions
└── utils/ #array utilities and slice and concat functions
├── ├── __init__.py
├── ├── array_utils.py
├── ├── concat_utils.py
└── └── slice_utils.py
```

Generally, the code files are formatted such that there are interface functions at the top of a given file. These interface functions then identify the input object types and call more specific functions, which are found further down in the file.

## Expectations

- naming conventions etc

  - objects that are single tables (i.e. a pandas DataFrame, numpy OrderedDict of arrays) are referred to as `Table-like`.
  - objects that are `OrderedDict` objects of `Table-like` objects are referred to as `TableDict-like` objects
  - functions are snake case (like_this)
  - classes use pascal case (LikeThis)

- test coverage etc

- expectations for updating documentation

- PR procedure

### Typing Recommendations:

When referring to a generic `TableDict` or `TableDict-like` object, use the `Mapping` type hint.

## Version Release and Deployment Procedures

### Creating a Pull Release for a Release Candidate

### Publishing Package on PyPi

### Making the Documentation on "Read The Docs"

This is how you do that

### Informing Developers of Downstream Packages

`tables_io` is a core package of the LSST DESC RAIL ecosystem. Consequently, the developers of the following packages should be informed about new versions:

- `qp`
- `rail_base`
- ...
