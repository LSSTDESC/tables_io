from setuptools import setup

setup(
    name="tables_io",
    author="Eric Charles",
    author_email="charles@slac.stanford.edu",
    url = "https://github.com/LSSTDESC/tables_io",
    packages=["tables_io"],
    description="Input/output and conversion functions for tabular data",
    setup_requires=['setuptools_scm'],
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    use_scm_version={"write_to":"tables_io/_version.py"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["numpy>=1.21.0;python_version>='3.8'",
                      "numpy<1.22.*;python_version<'3.8'",
                      "astropy;python_version>='3.8'",
                      "astropy<5.*;python_version<'3.8'",
                      "h5py",
                      "pandas;python_version>='3.8'",
                      "pandas<1.4.*;python_version<'3.8'",
                      "pyarrow",
                      "tables"],
    tests_require=["h5py>=2.9=mpi*",
                   "mpi4py"]
)
