<!--Copyright (C) SuperDARN Canada, University of Saskatchewan 
author(s) Marina Schmidt-->
# Installing pyDARNio 
---

[![License: LGPL v3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) 
[![Python 3.7](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 
![GitHub release (latest by date)](https://img.shields.io/github/v/release/superdarn/pyDARNio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4009470.svg)](https://doi.org/10.5281/zenodo.4009470)



## Prerequisites

**python 3.7+**

| Ubuntu      | OpenSuse       | Fedora        | OSX           |
| ----------- | -------------- | ------------- | ------------- |
| libyaml-dev | python3-PyYAML | libyaml-devel | Xcode/pip     |

You can check your python version with  

`$ python --version` or 
`$ python3 --version`

!!! Important 
    It is recommended to install pyDARNio via `pip`; however, please cite via the [DOI for the release](https://doi.org/10.5281/zenodo.4009470) 


## Dependencies

pyDARNio's setup will download the following dependencies: 

- [Git](https://git-scm.com/) (For developers)
- [pip3](https://help.dreamhost.com/hc/en-us/articles/115000699011-Using-pip3-to-install-Python3-modules)
- [NumPy](https://numpy.org/)
- [pathlib2](https://docs.python.org/dev/library/pathlib.html)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [h5py](https://www.h5py.org/)

## Virtual Environments
It is recommended to install pyDARNio in one of the suggested virtual environments if you have multiple python/pip 3 version on your computer, or do not want to affect the main system's python libraries. 

The following virtual environments have been tested by pyDARNio developers:"

### pip Virtual Environment
Instructions can be found here [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

1. `$ python3 -m pip install --user virtualenv`  
2. `$ python3 -m virtualenv <environment name>`  
* `$ source <environment name>/bin/activate`
* `$ pip install pydarnio`

!!! Note
    If you have already installed `pydarnio` you can use `pip3 install --upgrade pydarnio`

!!! Note
    If you have multiple versions of python 3 on your machine, you can access a specific version by: `python<version number>`. 
    For example, if you want to install python 3.7 virtual environment: `python3.7 -m pip install --user virtualenv`.

### Anaconda Virtual Environment
Instructions can be found here [conda environment](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) and installing [anaconda](https://docs.anaconda.com/anaconda/install/)

1. `$ conda create -n yourenvname python=3.7 anaconda`
2. `$ conda activate yourenvname`
* `$ pip install pydarnio`

#### Adding the environment to PyCharm

To set the project interpreter to the anaconda environment:

1. File -> Settings -> Project Folder -> Project Interpreter
2. Click the project Interpreter drop down list and click on show all.
* If you don't see the environment you wish to use click the plus sign on the right side bar named "Add"
* Select "Conda Environment" on the left side menu.
* Click "Existing Environment" and give the interpreter field the path to your environment's python.exe and apply.

## Local Install
**pip3 install**

`pip3 install --user pydarnio`

## System Install 
`sudo pip3 install pydarnio`

## Installing for Development 
`$ git clone https://github.com/superdarn/pyDARNio`

Change directories to pyDARNio

`$ git checkout develop`

To install: 

`$ pip3 install . --user`

!!! Note
    If `pip --version` is pointing to python 3.7+ then you can use `pip install . --user` instead. 

!!! Warning
    Do not install pyDARNio with `python setup.py install` as this may install other libraries pyDARNio requires in a local directory causing issues down the road.
    
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
Then open a new terminal and you should see `(pyy38)` in the prompt. 

Credit to this solution is Ashton Reimer, more details on the [issue #37](https://github.com/SuperDARN/pyDARNio/issues/37)


> If you find any problems/solutions, please make a [github issue](https://github.com/superdarn/pyDARNio/issues/new) so the community can help you or add it to the documentation
