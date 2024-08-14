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
```python3
import pydarnio

file = "path/to/rawacf_file"
data = pydarnio.read_rawacf(file)
outfile = "path/to/outfile.rawacf"
pydarnio.write_rawacf(data, outfile)
```
which puts the file contents into `data`, then writes out to `"path/to/outfile.rawacf". `data` will be a list of dictionaries, 
where each dictionary is a DMAP record. The supported reading functions are:

* `read_iqdat`, 
* `read_rawacf`, 
* `read_fitacf`, 
* `read_grid`, 
* `read_map`,  
* `read_snd`, and
* `read_dmap`.

The supported writing functions are:

* `write_iqdat`, 
* `write_rawacf`,
* `write_fitacf`, 
* `write_grid`, 
* `write_map`, 
* `write_snd`, and
* `write_dmap`.

### Accessing data fields
To see the names of the variables you've loaded in and now have access to, try using the `keys()` method:
```python3
print(data[0].keys())
```
which will tell you all the variables in the first [0th] record.

Let's say you loaded in a MAP file, and wanted to grab the cross polar-cap potentials for each record:
```python
import pydarnio
file = "20150302.n.map"
map_data = pydarnio.read_map(file)

cpcps=[rec['pot.drop'] for rec in map_data]
```

## I/O on a compressed file

pyDARNio will handle compressing and decompressing `.bz2` files seamlessly, detecting the compression via the file extension. E.g.
```python3
import pydarnio
fitacf_file = "path/to/file.bz2"
data = pydarnio.read_fitacf(fitacf_file)
dmap.write_fitacf(data, "temp.fitacf.bz2")
```
will read in the compressed file, then also write out a new compressed file.

## Generic I/O
pyDARNio supports generic DMap I/O, without verifying the field names and types. The file must still
be properly formatted as a DMap file, but otherwise no checks are conducted.

**NOTE:** When using the generic writing function `write_dmap`, scalar fields will possibly be resized; e.g., the `stid`
field may be stored as an 8-bit integer, as opposed to a 16-bit integer as usual. As such, reading with a specific method
(e.g. `read_fitacf`) on a file written using `write_dmap` will likely not pass the DMap consistency checks.
```python3
import pydarnio
generic_file = "path/to/file"  # can be iqdat, rawacf, fitacf, grid, map, snd, and optionally .bz2 compressed
data = pydarnio.read_dmap(generic_file)
pydarnio.write_dmap(data, "temp.generic.fitacf")  # fitacf as an example
data2 = pydarnio.read_rawacf("temp.generic.fitacf")  # This will likely fail due to different types for scalar fields
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
dmap.write_fitacf(data, "path/to/fitacf/files/<date>.<radar>.fitacf.bz2")  # Write the concatenated data together
```
