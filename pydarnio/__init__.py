"""
Copyright 2018 SuperDARN Canada, University Saskatchewan
Author(s): Marina Schmidt

Licensed under GNU v3.0

__init__.py
2018-11-05
Init file to setup the logging configuration and linking pyDARNio's
module, classes, and functions.
"""
# KEEP THIS FILE AS MINIMAL AS POSSIBLE!

__all__ = [
    "borealis_exceptions",
    "standard_warning_format",
    "only_message_warning_format",
    "read_iqdat",
    "read_rawacf",
    "read_fitacf",
    "read_grid",
    "read_map",
    "read_snd",
    "read_dmap",
    "write_iqdat",
    "write_rawacf",
    "write_fitacf",
    "write_grid",
    "write_map",
    "write_snd",
    "write_dmap",
    "borealis_formats",
    "BorealisRead",
    "BorealisWrite",
    "BorealisConvert",
    "BorealisRestructure",
]

# Exception formatting
from .exceptions import borealis_exceptions
from .exceptions.warning_formatting import standard_warning_format
from .exceptions.warning_formatting import only_message_warning_format

# DMap I/O
from dmap import (
    read_iqdat,
    read_rawacf,
    read_fitacf,
    read_grid,
    read_map,
    read_snd,
    read_dmap,
    write_iqdat,
    write_rawacf,
    write_fitacf,
    write_grid,
    write_map,
    write_snd,
    write_dmap
)

# Borealis I/O and converting Borealis to DMap
from .borealis import borealis_formats
from .borealis.borealis import BorealisRead, BorealisWrite
from .borealis.borealis_convert import BorealisConvert
from .borealis.borealis_restructure import BorealisRestructure
