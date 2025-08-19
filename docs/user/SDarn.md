# DMap structured SuperDARN data file I/O

Data Map (DMap) is a binary self-describing format that was developed by Rob Barnes. 
This format is currently the primary format used by SuperDARN. 
For more information on DMap please see [RST Documentation](https://radar-software-toolkit-rst.readthedocs.io/en/latest/).
Types of files used by SuperDARN which are usually accessed in DMap format are:

- IQDAT
- RAWACF
- FITACF
- GRID/GRD
- MAP
- SND

This tutorial will focus on reading in DMap structured files using pyDARNio, including how to read compressed files and access common data fields.

## The basics

The basic code to read and write a DMap structured file is as follows:
```python
import pydarnio

file = "path/to/rawacf_file"
data = pydarnio.read_rawacf(file)
outfile = "path/to/outfile.rawacf"
pydarnio.write_rawacf(data, outfile)  # writes binary data to `outfile`
raw_bytes = pydarnio.write_rawacf(data)  # returns a `bytes` object
```
which puts the file contents into `data`, then writes out to `"path/to/outfile.rawacf". `data` will be a list of dictionaries, 
where each dictionary is a DMAP record. The supported reading functions are:

- `read_iqdat`, 
- `read_rawacf`, 
- `read_fitacf`, 
- `read_grid`, 
- `read_map`,  
- `read_snd`, and
- `read_dmap`.

The supported writing functions are:

- `write_iqdat`, 
- `write_rawacf`,
- `write_fitacf`, 
- `write_grid`, 
- `write_map`, 
- `write_snd`, and
- `write_dmap`.

### Accessing data fields
To see the names of the variables you've loaded in and now have access to, try using the `keys()` method:
```python
print(data[0].keys())
```
which will tell you all the variables in the first (zeroth) record.

Let's say you loaded in a MAP file, and wanted to grab the cross polar-cap potentials for each record:
```python
import pydarnio
file = "20150302.n.map"
map_data = pydarnio.read_map(file)

cpcps=[rec['pot.drop'] for rec in map_data]
```

## I/O on a bz2 compressed file

pyDARNio will handle compressing and decompressing `.bz2` files seamlessly, detecting the compression via the file extension. E.g.
```python
import pydarnio
fitacf_file = "path/to/file.bz2"
data = pydarnio.read_fitacf(fitacf_file)
pydarnio.write_fitacf(data, "temp.fitacf.bz2")
```
will read in the compressed file, then also write out a new compressed file. Note that compression on the writing side
will only be done when writing to file, as the detection is done based on the file extension of the output file.

## Generic I/O
pyDARNio supports generic DMap I/O, without verifying the field names and types. The file must still
be properly formatted as a DMap file, but otherwise no checks are conducted.

**NOTE:** When using the generic writing function `write_dmap`, scalar fields will possibly be resized; e.g., the `stid`
field may be stored as an 8-bit integer, as opposed to a 16-bit integer as usual. As such, reading with a specific method
(e.g. `read_fitacf`) on a file written using `write_dmap` will likely not pass the DMap consistency checks.

```python
import pydarnio
generic_file = "path/to/file"  # can be iqdat, rawacf, fitacf, grid, map, snd, and optionally .bz2 compressed
data = pydarnio.read_dmap(generic_file)
pydarnio.write_dmap(data, "temp.generic.fitacf")  # fitacf as an example
data2 = pydarnio.read_rawacf("temp.generic.fitacf")  # This will fail due to different types for scalar fields
```

## Handling corrupted data files
The self-describing data format of DMap files makes it susceptible to corruption. The metadata fields which describe
how to interpret the following bytes are very important, and so any corruption will lead to the remainder of the file being
effectively useless. The I/O functions described above will raise an error if a corrupted record is encountered. However,
in certain use cases, the non-corrupted data before the bad record may be of use. As such, a lax read can be done by
specifying the `mode` parameter to the reading functions. These functions will not error on corrupted records in `lax` mode, 
but rather return all the good records, as well as the byte where the corrupted records start.

```python
import pydarnio

corrupted_file = "path/to/file"
data, bad_byte = pydarnio.read_dmap(corrupted_file, mode="lax")
assert bad_byte > 0

good_file = "path/to/file"
data, bad_byte = pydarnio.read_dmap(good_file, mode="lax")
assert bad_byte is None
```
In both uses of the above example, `data` will be a list of all records extracted from the file, but may be
considerably smaller than the file.

## Stream I/O
pyDARNio also can conduct read/write operations from/to Python `bytes` objects directly. These bytes must be formatted in 
accordance with the DMap format. Simply pass in a `bytes` object to any of the `read_[type]` functions instead of a path
and the input will be parsed.

While not the recommended way to read data from a DMap file, the following example shows the use of these byte I/O functions:
```python
import pydarnio
file = "path/to/file"
with open(file, 'rb') as f:  # 'rb' specifies to open the binary (b) file as read-only (r)
    raw_bytes = f.read()  # reads the file in its entirety
data = pydarnio.read_dmap(raw_bytes)
binary_data = pydarnio.write_dmap(data)
assert binary_data == raw_bytes
```
As a note, this binary data can be compressed ~2x typically using zlib, or with another compression utility. This is quite 
useful if sending data over a network where speed and bandwidth must be considered. Note that the binary writing functions
don't compress automatically, an external package like `zlib` or `bzip2` must be used.

## File "sniffing"
If you only want to inspect a file, without actually needing access to all of the data, you can use the `read_[type]`
functions in `"sniff"` mode. This will only read in the first record from a file, and works on both compressed and 
non-compressed files. Note that this mode does not work with bytes objects directly.

```python
import pydarnio
path = "path/to/file"
first_rec = pydarnio.read_dmap(path, mode="sniff")
```

## Other Examples

Other examples of using pyDARNio with file reading is for reading in multiple 2-hour files, sorting them, and concatenating the data together.
For example, you may do something like this, using the **glob** library:

```python
import pydarnio 
from glob import glob

fitacf_files = glob('path/to/fitacf/files/<date>*<radar>*.fitacf.bz2')
data = []

# assuming they are named via date and time
fitacf_files.sort()
print("Reading in fitacf files")
for fitacf_file in fitacf_files:
    data += pydarnio.read_fitacf(fitacf_file)
print("Reading complete...")
pydarnio.write_fitacf(data, "path/to/fitacf/files/<date>.<radar>.fitacf.bz2")  # Write the concatenated data together
```
