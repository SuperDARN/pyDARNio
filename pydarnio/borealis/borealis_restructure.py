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
import os
import subprocess as sp
import warnings
from pathlib import Path
import h5py
import deepdish as dd
import logging
import numpy as np
from datetime import datetime
from typing import Union

from pydarnio import (borealis_exceptions, BorealisRead)
from pydarnio.borealis import borealis_formats
from pydarnio.borealis.borealis_utilities import BorealisUtilities

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
                 borealis_filetype: str, outfile_structure: str,
                 hdf5_compression: Union[str, None] = None):
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
        hdf5_compression: Union[str, None]
            String representing HDF5 compression type. Default None.

        Raises
        ------
        BorealisFileTypeError
        BorealisStructureError
        ConvertFileOverWriteError
        """
        self.infile_name = infile_name
        self.outfile_name = outfile_name
        self.compression = hdf5_compression

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

        # Initialize to None, will be updated in restructure()
        self._borealis_version = None
        self._format = None

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

    @property
    def software_version(self):
        """
        The version of the file, taken from the 'borealis_git_hash' in the
        first record, in the init.
        """
        return self._borealis_version

    @property
    def format(self):
        """
        The format class used for the file, from the borealis_formats module.
        """
        return self._format

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

        # Find the version number
        try:
            version = dd.io.load(self.infile_name,
                                 group='/borealis_git_hash').split('-')[0]
        except ValueError as err:
            raise borealis_exceptions.BorealisStructureError(
                ' {} Could not find the borealis_git_hash required to '
                'determine file version. Data file may be corrupted. {}'
                ''.format(self.infile_name, err)) from err

        if version not in borealis_formats.borealis_version_dict:
            raise borealis_exceptions.BorealisVersionError(self.infile_name,
                                                           version)
        else:
            self._borealis_version = version

        self._format = borealis_formats.borealis_version_dict[
                self.software_version][self.borealis_filetype]

        if self.format.is_restructurable():
            attribute_types = self.format.site_single_element_types()
            dataset_types = self.format.site_dtypes()
            try:
                # TODO: One by one, create a record from the arrays and save to file
                pass
            except Exception as err:
                raise borealis_exceptions.BorealisRestructureError(
                    'Records for {}: Error restructuring {} from array to site '
                    'style: {}'
                    ''.format(self.infile_name, self.format.__name__, err)) from err

        else:
            raise borealis_exceptions.BorealisRestructureError(
                'Records for {}: File format {} not recognized as '
                'restructureable from array to site style'
                ''.format(self.infile_name, self.format.__name__))

    def site_to_array_restructure(self):

        # Find the version number
        try:
            version = dd.io.load(self.infile_name,
                                 group='/{}/borealis_git_hash'
                                       ''.format(self.record_names[0])).split('-')[0]
        except (IndexError, ValueError) as err:
            # Shouldn't happen unless file somehow is corrupted
            raise borealis_exceptions.BorealisStructureError(
                ' {} Could not find the borealis_git_hash required to '
                'determine read version. Data file may be corrupted. {}'
                ''.format(self.infile_name, err)) from err

        if version not in borealis_formats.borealis_version_dict:
            raise borealis_exceptions.BorealisVersionError(self.infile_name, version)
        else:
            self._borealis_version = version

        self._format = borealis_formats.borealis_version_dict[
            self.software_version][self.borealis_filetype]

        if self.format.is_restructurable():
            attribute_types = self.format.array_single_element_types()
            dataset_types = self.format.array_array_dtypes()
            unshared_fields = self.format.unshared_fields()
            try:
                # TODO: Use the number of records and the first record to create the correct
                #   array shapes (may need to extend some for zero-padding later)
                # TODO: Iterate over the records and add them into the arrays

                new_data_dict = dict()
                num_records = len(self.record_names)
                for record_name in self.record_names:
                    record = dd.io.load('/{}'.format(record_name))

                    # some fields are linear in site style and need to be reshaped.
                    # TODO: Figure out if the record is too shallow, i.e. param should be {'1': record}
                    #   so reshape_site_arrays() functions properly
                    data_dict = self.format.reshape_site_arrays(record)

                    # write shared fields to dictionary
                    # TODO: Only do this once
                    first_key = list(data_dict.keys())[0]
                    for field in self.format.shared_fields():
                        new_data_dict[field] = data_dict[first_key][field]

                    # write array specific fields using the given functions.
                    # TODO: Figure out a way to do this iteratively (some fields may have entry for
                    #   each record (e.g. num_beams in bfiq files)
                    for field in self.format.array_specific_fields():
                        new_data_dict[field] = self.format.array_specific_fields_generate(
                        )[field](data_dict)

                    # write the unshared fields, initializing empty arrays to start.
                    temp_array_dict = dict()

                    # get array dims of the unshared fields arrays
                    # TODO: Figure out way to either do this on the fly, or deal with loading each
                    #   record into memory twice (once for this, once for data)
                    field_dimensions = {}
                    for field in self.format.unshared_fields():
                        d = [dimension_function(data_dict) for
                             dimension_function in
                             self.format.unshared_fields_dims_array()[field]]

                        dims = []
                        for dim in d:
                            if isinstance(dim, list):
                                for i in dim:
                                    dims.append(i)
                            else:
                                dims.append(dim)

                        field_dimensions[field] = dims

                    # all fields to become arrays
                    for field, dims in field_dimensions.items():
                        array_dims = [num_records] + dims
                        array_dims = tuple(array_dims)

                        if field in self.format.single_element_types():
                            datatype = self.format.single_element_types()[field]
                        else:  # field in array_dtypes
                            datatype = self.format.array_dtypes()[field]
                        if datatype == np.unicode_:
                            # unicode type needs to be explicitly set to have
                            # multiple chars (256)
                            datatype = '|U256'
                        empty_array = np.empty(array_dims, dtype=datatype)
                        # Some indices may not be filled due to dimensions that are maximum
                        # values (num_sequences, etc. can change between records), so they are
                        # initialized with a known value first.
                        # Initialize floating-point values to NaN, and integer values to -1.
                        if datatype is np.int64 or datatype is np.uint32:
                            empty_array[:] = -1
                        else:
                            empty_array[:] = np.NaN
                        temp_array_dict[field] = empty_array

                    # iterate through the records, filling the unshared and array only
                    # fields
                    # TODO: Figure out way to incorporate this loop with looping over other fields/dims
                    for rec_idx, k in enumerate(data_dict.keys()):
                        for field in self.format.unshared_fields():  # all unshared fields
                            empty_array = temp_array_dict[field]
                            if type(data_dict[first_key][field]) == np.ndarray:
                                # only fill the correct length, appended NaNs occur for
                                # dims with a determined max value
                                data_buffer = data_dict[k][field]
                                buffer_shape = data_buffer.shape
                                index_slice = [slice(0, i) for i in buffer_shape]
                                # insert record index at start of array's slice list
                                index_slice.insert(0, rec_idx)
                                index_slice = tuple(index_slice)
                                # place data buffer in the correct place
                                empty_array[index_slice] = data_buffer
                            else:  # not an array, num_records is the only dimension
                                empty_array[rec_idx] = data_dict[k][field]

                    new_data_dict.update(temp_array_dict)

                BorealisUtilities.check_arrays(
                    self.outfile_name, arrays,
                    attribute_types, dataset_types,
                    unshared_fields)
            except Exception as err:
                raise borealis_exceptions.BorealisRestructureError(
                    'Records for {}: Error restructuring {} from site to array '
                    'style: {}'
                    ''.format(self.infile_name, self.format.__name__, err)) from err

        else:
            raise borealis_exceptions.BorealisRestructureError(
                'Records for {}: File format {} not recognized as '
                'restructureable from site to array style'
                ''.format(self.infile_name, self.format.__name__))

    def _write_borealis_record(self, record: dict, record_name: str,
                               attribute_types: dict, dataset_types: dict):
        """
        Add a record to the output file in site style after checking the record.

        Several Borealis field checks are done to insure the integrity of the
        record.

        Parameters
        ----------
        record: dict
            Dictionary containing the site-structured record.
        record_name: str
            Group name of the record for the HDF5 hierarchy.
        attribute_types: dict
            Dictionary with the required types for the attributes in the file.
        dataset_types: dict
            Dictionary with the required dtypes for the numpy arrays in the
            file.

        Raises
        ------
        BorealisFieldMissingError - when a field is missing from the Borealis
                                file/stream type
        BorealisExtraFieldError - when an extra field is present in the
                                Borealis file/stream type
        BorealisDataFormatTypeError - when a field has the incorrect
                                field type for the Borealis file/stream type

        See Also
        --------
        BorealisUtilities
        """
        Path(self.outfile_name).touch()
        BorealisUtilities.check_records(self.outfile_name, record,
                                        attribute_types, dataset_types)

        # use external h5copy utility to move new record into 2hr file.

        warnings.filterwarnings("ignore")
        # Must use temporary file to append to a file; writing entire
        # dictionary at once also doesn't work so this is required.
        tmp_filename = self.outfile_name + '.tmp'
        Path(tmp_filename).touch()

        dd.io.save(tmp_filename, {str(record_name): record},
                   compression=self.compression)
        cp_cmd = 'h5copy -i {newfile} -o {full_file} -s {dtstr} -d {dtstr}'
        cmd = cp_cmd.format(newfile=tmp_filename, full_file=self.outfile_name,
                            dtstr='/'+str(record_name))
        sp.call(cmd.split())
        os.remove(tmp_filename)
