![pydarnio](imgs/pydarnio_logo.png)

PyDARNio is an open source python library for SuperDARN data reading, writing and converting.
Currently the library support reading for DMAP files:

- IQDAT
- RAWACF
- FITACF
- GRID
- MAP
- SND

As well as Borealis HDF5 files:

- antennas_iq
- bfiq
- rawacf

There is also utilities for converting Borealis files to SuperDARN DMAP file structures. 

## Source Code 

The library source code can be found on the [pyDARNio GitHub](https://github.com/SuperDARN/pyDARNio) repository. 

If you have any questions or concerns please submit an **Issue** on the SuperDARN pyDARNio repository. 

## Table of Contents 
  - [Installation](user/install.md)
  - [SuperDARN Data Access](user/superdarn_data.md)
  - [Citing](user/citing.md)
  - Tutorials 
    - IO 
        - [SuperDARN (DMAP) files](user/SDarn.md)
        - [Borealis files](user/BorealisIO.md)
    - Convert
      - [Borealis to SuperDARN (DMAP)](user/Borealis2SuperDARN.md)
    - [Radar and Hardware Information](user/hardware.md)
