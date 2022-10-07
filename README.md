![pyDARNio](https://raw.githubusercontent.com/SuperDARN/pyDARNio/master/docs/imgs/pydarnio_logo.png)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 
![GitHub release (latest by date)](https://img.shields.io/github/v/release/superdarn/pyDARNio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4009470.svg)](https://doi.org/10.5281/zenodo.4009470)

Python data IO library for the Super Dual Auroral Radar Network (SuperDARN).

## Changelog

## Version 1.1.1 - Release!

pyDARNio is released! This is a patch release to address the following issues:
- Bug fix with initializing empty arrays when converting HDF5 files from site- to array-structured
- Bug fixes with converting files from HDF5 to DMAP
  - correctly check blanked_samples
  - support multiple beams per record
  - replacement of far-range lag0 data in rawacf conversion

## Documentation

pyDARNio's documentation can found [here](https://pydarnio.readthedocs.io/en/latest/)

## Getting Started


`pip install pydarnio`

Or read the [installation guide](https://pydarnio.readthedocs.io/en/latest/user/install/).

If wish to get access to SuperDARN data please read the [SuperDARN data access documentation](https://pydarnio.readthedocs.io/en/latest/user/superdarn_data/).
Please make sure to also read the documentation on [**citing SuperDARN and pydarn**](https://pydarnio.readthedocs.io/en/latest/user/citing/). 

As a quick tutorial on using pyDARNio to read a non-compressed file: 
```python
import pydarnio

# read a non-compressed file
fitacf_file = '20180220.C0.rkn.stream.fitacf'

# pyDARNio functions to read a fitacf file
reader = pydarnio.SDarnRead(fitacf_file)
records = reader.read_fitacf()
```

or to read a compressed file:
``` python
import bz2
import pydarnio
# read in compressed file
fitacf_file = '20180220.C0.rkn.stream.fitacf.bz2'
with bz2.open(fitacf_file) as fp: 
      fitacf_stream = fp.read()

# pyDARNio functions to read a fitacf file stream
reader = pydarnio.SDarnRead(fitacf_stream, True)
records = reader.read_fitacf()
```

For more information and tutorials on pyDARNio please see the [tutorial section](https://pydarnio.readthedocs.io/en/latest/)

## Getting involved

pyDARNio is always looking for testers and developers keen on learning python, github, and/or SuperDARN data visualizations! 
Here are some ways to get started: 

  - **Testing Pull Request**: to determine which [pull requests](https://github.com/SuperDARN/pyDARNio/pulls) need to be tested right away, filter them by their milestones (v1.2.0 is currently highest priority).
  - **Getting involved in projects**: if you are looking to help in a specific area, look at pyDARNio's [projects tab](https://github.com/SuperDARN/pyDARNio/projects). The project you are interested in will give you information on what is needed to reach completion. This includes things currently in progress, and those awaiting reviews. 
  - **Answer questions**: if you want to try your hand at answering some pyDARNio questions, or adding to the discussion, look at pyDARNio's [issues](https://github.com/SuperDARN/pyDARNio/issues) and filter by labels.
  - **Become a developer**: if you want to practice those coding skills and add to the library, look at pyDARNio [issues](https://github.com/SuperDARN/pyDARNio/issues) and filter by milestone's to see what needs to get done right away. 

Please contact the leading developer, Marina Schmidt (marina.t.schmidt@gmail.com), if you would like to become a member of the team!
