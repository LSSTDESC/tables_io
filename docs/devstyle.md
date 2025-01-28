# Style Guide

## Documentation

All documentation is created using Sphinx. This lives in the...

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

- diagram of how the code works
- short description of how code is organized, etc

## Expectations

- naming conventions etc

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
