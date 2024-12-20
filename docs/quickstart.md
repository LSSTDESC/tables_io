# Quickstart 


## Installation

* basic installation 

* parallel installation

* also dev installation 

## What is tables_io?

* code intended to allow for the reading and writing of multiple different file formats using a single package 
* something about how all these file formats currently supported are not csv formats - not sure if there's a good way to describe them other than as currently used in astronomy 
* allows easy conversions between file formats 

## Read/write 


* how to do read
    * two options: multi read/write gives ordered dict, single read/write gives just the table format 
* recommended usage:
    * provide a format key when reading in a file to make sure you get the memory format you want 
    * keep in mind this could result in more memory usage due to conversions 
* how to write:
    * I think single write function works 
    * can provide output format again to make sure you get the file type you want 
    * keep in mind this could result in more memory usage due to conversions 



### Supported formats

* list supported formats to read files in from and write to, with the associated suffixes 


| File format | File suffix | 
|-------------|-------------|
| astropy_fits | .fits |

* table that associates those file formats with in memory formats that are available 

| Format in memory | File format read | File format write | 
|------------------|------------------|-------------------|
| astropy table | astropy_hdf5, astropy_fits | **astropy_hdf5**, astropy_fits | 





## Conversion

* how to convert a table outside of read/write functionality 


## Array operations

* maybe don't include, since these functions are pretty rough right now ?
* some functionality is available for dealing with tables_io tables 
* concatenating, slicing 