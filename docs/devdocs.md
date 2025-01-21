# Developer Documentation

## Developer Environment Setup

For the installation of `tables_io` for the purpose of development, we recommend that you use a separate [Anaconda](https://docs.anaconda.com/anaconda/install/) virtual environment with `tables_io` installed in "editable" mode with all dev optional dependencies added.

In this guide, we will name this developer environment `tables_io_dev` and we will assume that an Anaconda with a minimum python version of 3.9 has been previously installed.

To install the developer environment:

```bash
# Creating the environment from the YAML
conda env create -n tables_io_dev -f environment.yml

# Activate the environment
conda activate tables_io_dev

# Install tables_io in editable mode with dev dependencies
pip install -e '.[dev]'
```

### Setting up Parallel HDF5

**TODO: CHECK IF ANYTHING ELSE NEEDS TO BE DONE**

## Running Tests

All tests are coordinated via `pytest`...

## Documentation

All documentation is created using Sphinx. This lives in the...

### Writing Documentation Pages

This is what you should do

### Building the Documentation Locally

This is how you do that!

## Understanding the code

- diagram of how the code works
- short description of how code is organized, etc

## Expectations

- naming conventions etc

- test coverage etc

- expectations for updating documentation

- PR procedure

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
