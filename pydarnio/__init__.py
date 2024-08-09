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

# Importing pydarnio exception classes
from .exceptions import borealis_exceptions
from .exceptions.warning_formatting import standard_warning_format
from .exceptions.warning_formatting import only_message_warning_format

# Import the dmap I/O functions
from dmap import (
    read_iqdat,
    read_rawacf,
    read_fitacf,
    read_grid,
    read_map,
    read_snd,
    write_iqdat,
    write_rawacf,
    write_fitacf,
    write_grid,
    write_map,
    write_snd
)

# Importing pydarnio borealis classes
from .borealis import borealis_formats
from .borealis.borealis import BorealisRead
from .borealis.borealis import BorealisWrite
from .borealis.borealis_convert import BorealisConvert
from .borealis.borealis_restructure import BorealisRestructure
