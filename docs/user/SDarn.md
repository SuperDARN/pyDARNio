# DMap structured SuperDARN data file I/O
---

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

## Reading with SDarnRead

The basic code to read in a DMap structured file is as follows:
```python
import pydarnio

file = "path/to/file"
reader = pydarnio.SDarnRead(file)
```
which puts the file contents into a Python object called `reader`. 

Now you need to tell it what kind of file it is. For instance, if the file you were reading in is a FITACF file, you would write something like:
```python
fitacf_data = reader.read_fitacf()
```
where the named variable `fitacf_data` is a python dictionary list containing all the data in the file. If you were reading a different kind of file, you would need to use the methods `read_iqdat`, `read_rawacf`, `read_grid`, `read_map`, or `read_snd` for their respective filetypes.

### Reading a compressed file

To read a compressed file like **bz2** (commonly used for SuperDARN data products), you will need to use [bz2 library](https://docs.python.org/3/library/bz2.html). 
The `SDarnRead` class allows the user to provide the file data as a stream of data which is what the **bz2** returns when it reads a compressed file: 
```python
import bz2
import pydarnio

fitacf_file = "path/to/file.bz2"
with bz2.open(fitacf_file) as fp:
      fitacf_stream = fp.read()

reader = pydarnio.SDarnRead(fitacf_stream, True)
records = reader.read_fitacf()
```

### Accessing data fields
To see the names of the variables you've loaded in and now have access to, try using the `keys()` method:
```python
print(records[0].keys())
```
which will tell you all the variables in the first [0th] record.

Let's say you loaded in a MAP file, and wanted to grab the cross polar-cap potentials for each record:
```python
file = "20150302.n.map"
reader = pydarnio.SDarnRead(file)
map_data = reader.read_map()

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

    reader = pydarnio.SDarnRead(fitacf_stream, True)
    records = reader.read_fitacf()
    data += records
print("Reading complete...")
```

### DMapRead

In pyDARNio, there also exists a class called `DMapRead`, which you can use in place of SDarnRead to read in any generic DMap structured file. However, pyDARNio won't test its integrity as it doesn't know what file it's supposed to be. If you're reading a SuperDARN file from one of the official data mirrors, then it is best to use SDarnRead in general.

## Writing with SDarnWrite

Very similarly to reading, DMap files can be written to file with a few simple commands.

The basic code to write out a DMap structured file is as follows:
```python
import pydarnio

data = ... # First you must get the records from a file, using SDarnRead()
outfile = "path/to/file"
writer = pydarnio.SDarnWrite(data, outfile)
```
which conducts some basic data integrity checks and stores in the object `writer`. 

Now you need to tell it what kind of file it is. For instance, if the file you were reading in is a FITACF file, you would write something like:
```python
writer.write_fitacf()
```
If you were writing a different kind of file, you would need to use the methods `write_iqdat`, `write_rawacf`, `write_grid`, `write_map` or `write_snd` for their respective filetypes.
These method calls will also check to ensure that the `data` you pass in is valid for the given file type.
