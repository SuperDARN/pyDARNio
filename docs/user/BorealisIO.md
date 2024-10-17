# Reading in Borealis HDF5 structured SuperDARN data files
---


HDF5 (Hierarchical Data Format v5) is a user-friendly data format that supports
heterogeneous data, allows for easy sharing, is cross platform, has fast I/O
with storage space optimizations, has no limit on dataset size inside the file,
and supports keeping metadata with the data in the file. For more on
HDF5, see the website of the [HDF Group](www.hdfgroup.org).

The Borealis software writes data files in HDF5 format. The files written on
site are written record-by-record, in a similar style to the SuperDARN standard
dmap format. As of Borealis v1.0, this is the only structure of HDF5 file supported by Borealis.

## Structure of the data

Prior to Borealis v1.0, two structures of data file were supported: "site" and "array". The site
structure stores data record-by-record, with some metadata fields redundantly stored in each
record of the file. To reduce file sizes, array-structured files were created. These files
group all like-data from across records into single datasets, zero-padding where necessary to 
create numpy arrays. Static metadata fields are stored only once. Site-structured files are denoted
with an additional `.site` suffix to the file extension. As of Borealis v1.0, the conventional file naming
has been changed, so all files end in the `.h5` file extension. This is both for brevity and to distinguish
between the vastly different formats by the file name alone. 

The restructuring process is fully built into the IO so that if you would like to see
the record-by-record data, you can simply return the record's attribute of the
IO class for any Borealis file. Similarly, if you would like to see the data in
the arrays format, return the arrays attribute. This works regardless of how
the original file was structured. For Borealis v1.0 files, I/O is only conducted when
either the `records` or `arrays` attribute is accessed, returning the data in the respective
structure in memory. 

## File types

In addition to file structure, there are various types of data sets (file types)
that can be produced by Borealis. The file types that can be produced are:

- `'rawrf'`
This is the raw samples at the receive bandwidth rate. This is rarely
produced and only would be done by request.

- `'antennas_iq'`
Downsampled data from individual antennas, i and q samples.

- `'bfiq'`
Beamformed i and q samples. Typically two array data sets are included,
for main array and interferometer array.

- `'rawacf'`
The correlated data given as lags x ranges, for the two arrays.

Borealis files can also be converted to the standard SuperDARN DMap formats
using pyDARNio.

## Reading with BorealisRead

BorealisRead class takes 3 parameters:

- `filename`,
- `borealis_filetype`, and
- `borealis_file_structure` (optional but recommended for v0.x data, unused for v1.0 onwards).

The BorealisRead class can return either array or site structured data,
regardless of the file's structure. Note that if you are returning the structure
that the file was not stored in, it will require some processing time.

Here's an example:

```python
import pydarnio

bfiq_site_filename = "path/to/bfiq_site_file"
borealis_reader = pydarnio.BorealisRead(bfiq_site_filename, 'bfiq', 'site')

# We can return the original data from the site file. This will be a dictionary
# of dictionaries.
record_data = borealis_reader.records

# For site structured data, it is often helpful to have the record names alone
# in order to retrieve the data from the fields within the record.
record_names = borealis_reader.record_names

# We can also get the data in array structured format. Beware that this
# will require some processing. This will be a dictionary.
array_data = borealis_reader.arrays
```

For v0.x data, if you don't supply the borealis_file_structure parameter, the reader will
attempt to read the file as array structured first (as this should be the most
common structure available to the user), and following failure will attempt to
read as site structured.

```python
import pydarnio

rawacf_array_filename = "path/to/rawacf_array_file"
borealis_reader = pydarnio.BorealisRead(rawacf_array_filename, 'rawacf')

print(borealis_reader.borealis_file_structure) # confirm it was array structured

# We can return the original data from the array file
array_data = borealis_reader.arrays

# We can also get the data in the site structured format. Again, beware that
# this will require some processing.
record_data = borealis_reader.records
```

## Accessing Data Fields in a Borealis Dataset

The method of accessing data fields will vary depending on if you have loaded
site data or array data. In both cases, you can use the `keys()` method.

For site files, to see all the data fields in the first record:
```python
record_names = borealis_reader.record_names
first_record_name = record_names[0]
print(record_data[first_record_name].keys())
```

For array files, to see all the data fields available for all the records:
```python
print(array_data.keys())
```

For more information on the data fields available in both array structured
and site structured files (they vary slightly), see the Borealis documentation
[here](https://borealis.readthedocs.io/en/latest/borealis_data.html).

## Writing with BorealisWrite

!!! Warning
    Writing Borealis v1.0 data is not supported

The BorealisWrite class takes 4 parameters:

- `filename`,
- `borealis_data`,
- `borealis_filetype`, and
- `borealis_file_structure` (optional but recommended).

Here's an example that will write `my_rawacf_data` to `my_file`:

```python
import pydarnio

my_rawacf_data = borealis_reader.arrays

my_file = "path/to/file"
writer = pydarnio.BorealisWrite(my_file, my_rawacf_data, 'rawacf', 'array')
```

Similar to reading files, if you don't supply the borealis_file_structure
parameter, the writer will attempt to write the file as array structured first,
and following failure will attempt to write as site structured.

```python
import pydarnio

my_rawacf_data = borealis_reader.arrays

my_file = "path/to/file"
writer = pydarnio.BorealisWrite(my_file, my_rawacf_data, 'rawacf')

print(writer.borealis_file_structure)  # to check the file structure written
```

## Reading data in with xarray

Borealis v1.0 data can be read in using the [xarray](https://docs.xarray.dev/en/stable/) library.
To do so, you must install pydarnio with the optional `xarray` dependencies.
This can be done using `pip install pydarnio[xarray]`.

Two flavours of xarray I/O are supported: record-by-record, or with field grouped together across records.

```python
import pydarnio
infile = "/path/to/data.rawacf.h5"

# Read in record-by-record (returns list[xarray.Dataset])
dsets = pydarnio.BorealisV1Read.records_as_xarray(infile)

# Read in with data grouped together across records (returns xarray.Dataset)
ds = pydarnio.BorealisV1Read.arrays_as_xarray(infile)
```

The xarray I/O provides an extremely useful interface for exploring data files,
selecting/slicing data, and plotting data.
