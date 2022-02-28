# Copyright 2022 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains classes and functions for
restructuring of Borealis file types.

Classes
-------
BorealisRestructure: Restructures Borealis SuperDARN files types to/from
    site and array structures.

Exceptions
----------
TODO: Curate this list
BorealisFileTypeError
BorealisFieldMissingError
BorealisExtraFieldError
BorealisDataFormatTypeError
BorealisConversionTypesError
BorealisConvert2IqdatError
BorealisConvert2RawacfError

See Also
--------
BorealisRead
BorealisWrite
BorealisSiteRead
BorealisSiteWrite
BorealisArrayRead
BorealisArrayWrite

Notes
-----

For more information on Borealis data files and their structures,
see: https://borealis.readthedocs.io/en/master/

Future Work
-----------


"""
import h5py
import deepdish as dd
import logging
import numpy as np

from datetime import datetime
from typing import Union

from pydarnio import (borealis_exceptions, BorealisRead)

pyDARNio_log = logging.getLogger('pyDARNio')


class BorealisRestructure(object):
    """
    Class for restructuring Borealis filetypes.

    See Also
    --------
    BorealisRawacf
    BorealisBfiq
    BorealisAntennasIq
    BorealisRead
    BorealisSiteRead
    BorealisArrayRead
    BorealisWrite
    BorealisSiteWrite
    BorealisArrayWrite

    Attributes
    ----------
    infile_name: str
        The filename of the Borealis HDF5 file being read.
    outfile_name: str
        The filename of the Borealis HDF5 file being written to.
    borealis_filetype: str
        The type of Borealis file. Restructurable types include:
        'antennas_iq'
        'bfiq'
        'rawacf'
    outfile_structure: str
        The desired Borealis structure of outfile_name. Supported
        structures are 'site' and 'array'.
    """

    def __init__(self, infile_name: str, outfile_name: str,
                 borealis_filetype: str, outfile_structure: str):
        """
        Restructure HDF5 Borealis records to a given Borealis file structure.

        Parameters
        ----------
        infile_name: str
            file name containing Borealis hdf5 data.
        outfile_name: str
            file name to save the restructured file to.
        borealis_filetype: str
            The type of Borealis file.
        outfile_structure: str
            The write structure of the file provided. Possible types are
            'site' or 'array'. If the output structure is the same as the
            input structure, the file will be copied to a new file with
            name "outfile_name".

        Raises
        ------
        BorealisFileTypeError
        BorealisStructureError
        ConvertFileOverWriteError
        """
        self.infile_name = infile_name
        self.outfile_name = outfile_name

        if borealis_filetype not in ['antennas_iq', 'bfiq', 'rawacf']:
            raise borealis_exceptions.BorealisFileTypeError(
                self.infile_name, borealis_filetype)
        self.borealis_filetype = borealis_filetype

        if outfile_structure not in ['site', 'array']:
            raise borealis_exceptions.BorealisStructureError(
                "Unknown structure type: {}"
                "".format(outfile_structure))
        self.outfile_structure = outfile_structure

        if self.infile_name == self.outfile_name:
            raise borealis_exceptions.ConvertFileOverWriteError(
                    self.infile_name)

        self.record_names = self.get_record_names(infile_name)
        self.borealis_structure = self.determine_borealis_structure()

        # TODO: Call to some restructure method here.
        self.restructure()

    def __repr__(self):
        """ for representation of the class object"""

        return "{class_name}({infile}, {borealis_filetype}, {outfile})"\
               "".format(class_name=self.__class__.__name__,
                         infile=self.infile_name,
                         borealis_filetype=self.borealis_filetype,
                         outfile=self.outfile_name)

    def __str__(self):
        """ for printing of the class object"""

        return "Restructuring {infile} to {borealis_structure} "\
               "and writing to file {outfile}."\
               "".format(infile=self.infile_name,
                         borealis_structure=self.outfile_structure,
                         outfile=self.outfile_name)

    @staticmethod
    def get_record_names(borealis_hdf5_file: str):
        """
        Gets the top-level names of the records stored in the Borealis
        HDF5 file specified.

        Parameters
        ----------
        borealis_hdf5_file
            Borealis file to read. Either array- or site-structured.

        Returns
        -------
        record_names
            List of the top-level keys of the HDF5 file.

        Raises
        ------

        See Also
        --------

        """
        with h5py.File(borealis_hdf5_file, 'r') as f:
            key_view = f.keys()
            record_names = [str(key) for key in key_view]

        return record_names

    def determine_borealis_structure(self):
        """
        Determines the type of Borealis HDF5 file structure based on
        the names of the top-level records in the HDF5 file.

        Returns
        -------
        structure
            The Borealis HDF5 file structure of the input file. Either
            'site' or 'array'.
        """
        if 'borealis_git_hash' in self.record_names:
            structure = 'array'
        else:
            structure = 'site'
        return structure

    def restructure(self):
        """
        Top-level method for restructuring Borealis HDF5 files. Calls
        the appropriate restructuring method based on the direction of
        restructuring, i.e. site-to-array or array-to-site.
        """
        if self.borealis_structure == self.outfile_structure:
            print("File {infile} is already structured in {struct} style."
                  "".format(infile=self.infile_name, struct=self.outfile_structure))
            return

        if self.outfile_structure == 'site':
            self.array_to_site_restructure()
        else:
            self.site_to_array_restructure()

    def array_to_site_restructure(self):
        pass

    def site_to_array_restructure(self):
        pass
