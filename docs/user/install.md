<!--Copyright (C) SuperDARN Canada, University of Saskatchewan 
author(s) Marina Schmidt-->
# Installing pyDARNio 
---

[![License: LGPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) 
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) 
![GitHub release (latest by date)](https://img.shields.io/github/v/release/SuperDARN/pyDARNio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4009470.svg)](https://doi.org/10.5281/zenodo.4009470)



## Prerequisites

**python 3.8+**

| Ubuntu      | OpenSuse       | Fedora        | OSX           |
| ----------- | -------------- | ------------- | ------------- |
| libyaml-dev | python3-PyYAML | libyaml-devel | Xcode/pip     |

You can check your python version with  

`$ python --version` or 
`$ python3 --version`

## Dependencies

pyDARNio's setup will download the following dependencies: 

- [Git](https://git-scm.com/) (For developers)
- [darn-dmap](https://github.com/SuperDARNCanada/dmap)
- [NumPy](https://numpy.org/)
- [h5py](https://www.h5py.org/)

## Virtual Environments
Installation of pyDARNio in a virtual environment is recommended in most cases. 

## Installation Steps
**pip3 install**

`pip3 install pydarnio`

## Installing for Development 
`$ git clone https://github.com/superdarn/pyDARNio`

Change directories to pyDARNio

`$ git checkout develop`

To install: 

`$ pip3 install .`
    
## Troubleshooting

### Pip3 installation with Ubuntu 20.4/python 3.8.4

Issue: `pip3 install --user git+https://github.com/superdarn/pyDARNio@develop` not working

Solution:
1. Check git is installed `apt install git` (for ubuntu)
2. Check pip version `pip --version` - with newer distros of Linux/Virtual machines `pip` may point to python3 and you will not need pip3. 
3. Alternative virtual environment steps for getting python 3.8 working

```bash 
$ sudo apt-get update
$ sudo apt-get install python3-virtualenv python3-pip
$ cd ~/
$ mkdir venvs
$ virtualenv -p python3.8 ~/venvs/py38
$ echo "source $HOME/venvs/py38/bin/activate" >> ~/.bashrc
```
Then open a new terminal and you should see `(py38)` in the prompt. 

Credit to this solution is Ashton Reimer, more details on the [issue #37](https://github.com/SuperDARN/pyDARNio/issues/37)


> If you find any problems/solutions, please make a [github issue](https://github.com/superdarn/pyDARNio/issues/new) so the community can help you or add it to the documentation
