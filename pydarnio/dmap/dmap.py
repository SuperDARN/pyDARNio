# Copyright 2018 SuperDARN Canada, University of Saskatchewan
# Authors: Marina Schmidt and Keith Kotyk
"""
This file contains functions for reading and writing of formats used by SuperDARN.

This module is a thin wrapper of the `darn-dmap` library API.
"""
# todo: investigate in darn-dmap library why numpy arrays of shape (1,) get interpreted as scalars and break the
#  write_[...] functions
