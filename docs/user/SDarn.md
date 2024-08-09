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

## Reading files

The basic code to read in a DMap structured file is as follows:
```python3
import pydarnio

file = "path/to/rawacf_file"
data = pydarnio.read_rawacf(file)
```
which puts the file contents into `data`. This will be a list of dictionaries, where each dictionary is a DMAP record.
The supported reading functions are:
* `read_iqdat`, 
* `read_rawacf`, 
* `read_fitacf`, 
* `read_grid`, 
* `read_map`, and 
* `read_snd`.

### Reading a compressed file

To read a compressed file like **bz2** (commonly used for SuperDARN data products), you will need to use [bz2 library](https://docs.python.org/3/library/bz2.html). 
```python3
import bz2
import pydarnio

fitacf_file = "path/to/file.bz2"
decompressed_file = "path/to/file"
with bz2.open(fitacf_file) as fp:
    fitacf_stream = fp.read()
with open(decompressed_file, 'w') as f:
    f.write(fitacf_stream)
data = pydarnio.read_fitacf(decompressed_file)
```

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

cpcps=[i['pot.drop'] for i in map_data]
```
### Other Examples

Other examples of using pyDARNio with file reading is for reading in multiple 2-hour files, sorting them, and concatenating the data together.
For example, you may do something like this, using the **glob** library:

```python
import bz2 
import pydarnio 

from glob import glob

fitacf_files = glob('path/to/fitacf/files/<date>*<radar>*.fitacf.bz2')
data = []

# assuming they are named via date and time
fitacf_files.sort()
print("Reading in fitacf files")
for fitacf_file in fitacf_files:
    with bz2.open(fitacf_file) as fp:
        fitacf_stream = fp.read()
    with open(fitacf_file.strip('.bz2'), 'w') as f:
        f.write(fitacf_stream)
    records = pydarnio.read_fitacf(fitacf_file.strip('.bz2'))
    data += records
print("Reading complete...")
```

## Writing files

Very similarly to reading, DMap files can be written to file with a few simple commands.

The basic code to write out a DMap structured file is as follows:
```python3
import pydarnio

infile = "path/to/infile.rawacf"
data = pydarnio.read_rawacf(infile) # First you must get the records from a file
outfile = "path/to/outfile.rawacf"
writer = pydarnio.write_rawacf(data, outfile)
```

where, similar to the reading operation, the writing supported functions are:
* `write_iqdat`, 
* `write_rawacf`,
* `write_fitacf`, 
* `write_grid`, 
* `write_map`, and 
* `write_snd`.
