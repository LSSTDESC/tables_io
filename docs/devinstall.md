# Developer Installation

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
