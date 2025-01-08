# Getting Started


## Installation

* basic installation 

* parallel installation



## Read/write 

* how to read in tables
    * two options: multi read/write gives ordered dict, single read/write gives just the table format 
* recommended usage:
    * provide a format key when reading in a file to make sure you get the memory format you want 
    * keep in mind this could result in more memory usage due to conversions 
* how to write tables:
    * use write()
    * can provide output format again to make sure you get the file type you want 
    * keep in mind this could result in more memory usage due to conversions
        * to minimize this, check the list of file types that can be written to for each table memory format below  

* see cookbook page for details on more complicated read/write operations, i.e. chunked read of hdf5 files 



### Supported table formats

* list supported formats to read files in from and write to, with the associated suffixes 


| File format | File suffix | 
|-------------|-------------|
| astropy_fits | .fits |

* table that associates those file formats with in memory formats that are available 

| Format in memory | File format read/write | Native written file | 
|------------------|------------------|-------------------|
| astropy table | astropy_hdf5, astropy_fits | astropy_hdf5 | 





## Conversion

* how to convert a table outside of read/write functionality 


## Array operations

* some functionality is available for dealing with tables_io tables 
* concatenating, slicing 