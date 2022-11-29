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
BorealisFileTypeError
BorealisStructureError
ConvertFileOverWriteError
BorealisVersionError
BorealisRestructureError

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
from collections import OrderedDict

from pydarnio import borealis_exceptions, borealis_formats
from .borealis_utilities import BorealisUtilities

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

        self.record_names = BorealisUtilities.get_record_names(infile_name)
        self.borealis_structure = BorealisUtilities.\
            get_borealis_structure(self.record_names)
        self._borealis_version = BorealisUtilities.get_borealis_version(
            self.infile_name, self.record_names, self.borealis_structure)
        self._format = borealis_formats.borealis_version_dict[
            self.software_version][self.borealis_filetype]

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

    def restructure(self):
        """
        Top-level method for restructuring Borealis HDF5 files. Calls
        the appropriate restructuring method based on the direction of
        restructuring, i.e. site-to-array or array-to-site.
        """
        if self.borealis_structure == self.outfile_structure:
            print("File {infile} is already structured in {struct} style."
                  "".format(infile=self.infile_name,
                            struct=self.outfile_structure))
            return
        if self.format.is_restructureable():
            if self.outfile_structure == 'site':
                self._array_to_site_restructure()
            else:
                self._site_to_array_restructure()
        else:
            raise borealis_exceptions.BorealisRestructureError(
                'Records for {}: File format {} not recognized as '
                'restructureable from array to site style'
                ''.format(self.infile_name, self.format.__name__))

    def _array_to_site_restructure(self):
        """
        Performs restructuring on an array-structured Borealis HDF5 file,
        converting it to site-structured. This method only loads in the HDF5
        groups and datasets that it needs as it needs them, and generates
        one site-structured record at a time.

        Raises
        -------
        BorealisStructureError
        BorealisVersionError
        BorealisRestructureError
        """
        attribute_types = self.format.site_single_element_types()
        dataset_types = self.format.array_dtypes()
        try:
            shared_fields_dict = dict()
            # shared fields are common across records, so this is done once
            for field in self.format.shared_fields():
                field_data = dd.io.load(self.infile_name, '/{}'.format(field))
                shared_fields_dict[field] = field_data

            unshared_single_elements = dict()
            # These are fields which have one element per record, so the
            # arrays are small enough to be loaded completely into memory
            for field in self.format.unshared_fields():
                if field in self.format.single_element_types():
                    unshared_single_elements[field] = dd.io.load(
                        self.infile_name, '/{}'.format(field))

            sqn_timestamps_array = dd.io.load(self.infile_name,
                                              '/sqn_timestamps')
            for record_num, seq_timestamp in enumerate(sqn_timestamps_array):
                # format dictionary key in the same way it is done
                # in datawrite on site
                seq_datetime = datetime.utcfromtimestamp(seq_timestamp[0])
                epoch = datetime.utcfromtimestamp(0)
                key = str(int((seq_datetime - epoch).total_seconds() * 1000))

                # Make this fresh every time, to reduce memory footprint
                record_dict = dict()

                # Copy over the shared fields
                for k, v in shared_fields_dict.items():
                    record_dict[k] = v

                # populate site specific fields using given functions
                # that take both the arrays data and the record number
                with h5py.File(self.infile_name, 'r') as f:
                    for field in self.format.site_specific_fields():
                        record_dict[field] = \
                            self.format.site_specific_fields_generate(
                                )[field](f, record_num)

                for field in self.format.unshared_fields():
                    if field in self.format.single_element_types():
                        datatype = self.format.single_element_types()[field]
                        # field is not an array, single element per record.
                        # unshared_field_dims_site should give empty list.
                        record_dict[field] = \
                            datatype(unshared_single_elements[field][
                                         record_num])
                    else:  # field in array_dtypes
                        # need to get the dims correct,
                        # not always equal to the max
                        field_flag = False
                        with h5py.File(self.infile_name, 'r') as f:
                            site_dims = [dimension_function(f, record_num)
                                         for dimension_function in
                                         self.format.unshared_fields_dims_site(
                                                )[field]]
                            dims = []
                            for dim in site_dims:
                                if isinstance(dim, list):
                                    for i in dim:
                                        dims.append(i)
                                else:
                                    dims.append(dim)
                            if -1 in dims:
                                field_flag = True

                            site_dims = dims
                            index_slice = [slice(0, i) for i in site_dims if i != -1]
                            index_slice.insert(0, record_num)
                            index_slice = tuple(index_slice)
                            # If there was an incorrect dimension (-1 in dims), then use deepdish to extract the field
                            if field_flag:
                                record_dict[field] = dd.io.load(self.infile_name, f'/{field}')[index_slice]
                            else:
                                record_dict[field] = f[field][index_slice]
                # Wrap in another dict to use the format method
                record_dict = OrderedDict({key: record_dict})
                record_dict = self.format.flatten_site_arrays(record_dict)

                # Write the single record to file
                self._write_borealis_record(record_dict, key, attribute_types,
                                            dataset_types)
        except Exception as err:
            raise borealis_exceptions.BorealisRestructureError(
                'Records for {}: Error restructuring {} from array to site '
                'style: {}'
                ''.format(self.infile_name, self.format.__name__, err)) \
                    from err

    def _site_to_array_restructure(self):
        """
        Performs restructuring on a site-structured Borealis HDF5 file,
        converting it to array-structured. This method only loads in one record
        at a time, adding its data to the arrays before moving onto the
        next record.

        Raises
        -------
        BorealisStructureError
        BorealisVersionError
        BorealisRestructureError
        """
        try:
            new_data_dict = dict()
            num_records = len(self.record_names)
            first_time = True
            # get array dims of the unshared fields arrays
            max_field_dims, max_num_sequences, max_num_beams = self.format.site_get_max_dims(
                self.infile_name, self.format.unshared_fields())

            # Functions that get called on each record, storing them here for readability
            array_specific_fields_funcs = self.format.array_specific_fields_iterative_generator()

            with h5py.File(self.infile_name, 'r') as f:
                for rec_idx, record_name in enumerate(self.record_names):

                    record = f[record_name]     # returns a view, doesn't do full loading into memory
                    rec_keys = list(record.keys())
                    rec_dict = {k: record[k][()] for k in rec_keys}

                    # Some things are stored as attributes, must be loaded in separately
                    rec_attrs = [k for k in record.attrs.keys() if k not in
                                 ['CLASS', 'TITLE', 'VERSION'] + self.format.bool_types()]
                    rec_dict.update({k: record.attrs[k] for k in rec_attrs})
                    # Bitwise fields also need to be handled separately
                    for field in self.format.bool_types():
                        rec_dict[field] = dd.io.load(self.infile_name, f'/{record_name}/{field}')

                    # some fields are linear in site style and need to be reshaped.
                    # Pass in record nested in a dictionary, as
                    # reshape_site_arrays is for dealing with key, val pairs of
                    # timestamp, record. Unpack the dictionary returned
                    data_dict = self.format.reshape_site_arrays({'tmp': rec_dict})['tmp']

                    # write shared fields to dictionary
                    if first_time:
                        for field in self.format.shared_fields():
                            value = data_dict[field]
                            if field not in self.format.string_fields():  # Regular old data
                                new_data_dict[field] = value
                            elif field in self.format.single_string_fields():
                                if isinstance(value, bytes):  # This is how single strings are interpreted by h5py
                                    new_data_dict[field] = self.format.single_element_types()[field](
                                        value.decode('utf-8'))
                                elif isinstance(value, h5py.Empty):  # Field is empty
                                    if value.dtype.char == 'S':
                                        new_data_dict[field] = self.format.single_element_types()[field]('')
                                    else:
                                        raise TypeError(f'Unknown datatype for empty field {field}: {value.dtype}')
                                else:
                                    raise TypeError(f'Field {field} has unrecognized data: {value}')
                            elif field in self.format.array_string_fields():
                                # h5py reads numpy string arrays as contiguous unsigned ints, so we need deepdish here
                                new_data_dict[field] = dd.io.load(self.infile_name, f'/{record_name}/{field}')
                            else:
                                raise TypeError(f'Field {field} unrecognized')

                        for field in self.format.array_specific_fields():
                            # Field is a constant value, i.e. doesn't depend on
                            # the data within the file, only the file type
                            if field not in array_specific_fields_funcs.keys():
                                new_data_dict[field] = self.format.array_specific_fields_generate()[field](
                                    {'tmp': data_dict})
                            else:
                                # Initialize array now with correct data type.
                                dtype = self.format.single_element_types()[field]
                                new_data_dict[field] = np.empty(num_records, dtype=dtype)
                                if dtype is np.int64 or dtype is np.uint32:
                                    new_data_dict[field][:] = -1
                                else:
                                    new_data_dict[field][:] = np.NaN

                    # Add data for this record to all fields that are
                    # array-specific and record-dependent
                    for field in array_specific_fields_funcs.keys():
                        new_data_dict[field][rec_idx] = array_specific_fields_funcs[field](rec_dict)

                    # write the unshared fields, initializing empty arrays first
                    if first_time:
                        # all fields to become arrays
                        for field, dims in max_field_dims.items():
                            array_dims = [num_records]
                            array_dims.extend([i for i in dims])
                            array_dims = tuple(array_dims)

                            if field in self.format.single_element_types():
                                datatype = self.format.single_element_types()[field]
                            else:  # field in array_dtypes
                                datatype = self.format.array_dtypes()[field]
                            if datatype == np.unicode_:
                                # unicode type needs to be explicitly set to
                                # have multiple chars (256)
                                datatype = '|U256'
                            empty_array = np.empty(array_dims, dtype=datatype)
                            # Some indices may not be filled due to dimensions
                            # that are maximum values (num_sequences, etc. can
                            # change between records), so they are initialized
                            # with a known value first. Initialize floating-
                            # point values to NaN, and integer values to -1.
                            if datatype is np.int64 or datatype is np.uint32:
                                empty_array[:] = -1
                            else:
                                empty_array[:] = np.NaN
                            new_data_dict[field] = empty_array
                        first_time = False

                    # Fill the unshared and array-only fields for this record
                    for field in self.format.unshared_fields():
                        empty_array = new_data_dict[field]
                        if type(data_dict[field]) == np.ndarray:
                            # only fill the correct length, appended NaNs occur
                            # for dims with a determined max value
                            data_buffer = data_dict[field]
                            buffer_shape = data_buffer.shape
                            index_slice = [slice(0, i) for i in buffer_shape]
                            # insert record index at start of array's slice list
                            index_slice.insert(0, rec_idx)
                            index_slice = tuple(index_slice)
                            # place data buffer in the correct place
                            empty_array[index_slice] = data_buffer
                        else:  # not an array, num_records is the only dimension
                            empty_array[rec_idx] = data_dict[field]

            attribute_types = self.format.array_single_element_types()
            dataset_types = self.format.array_array_dtypes()
            unshared_fields = self.format.unshared_fields()
            BorealisUtilities.check_arrays(self.infile_name, new_data_dict,
                                           attribute_types, dataset_types,
                                           unshared_fields)
            dd.io.save(self.outfile_name, new_data_dict,
                       compression=self.compression)

        except TypeError as err:
            raise borealis_exceptions.BorealisRestructureError(
                'Records for {}: Error restructuring {} from site to array '
                'style: {}'.format(self.infile_name, self.format.__name__, err)
            ) from err

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
        BorealisFieldMissingError
        BorealisExtraFieldError
        BorealisDataFormatTypeError

        See Also
        --------
        BorealisUtilities
        """
        Path(self.outfile_name).touch()
        BorealisUtilities.check_records(self.infile_name, record,
                                        attribute_types, dataset_types)

        # use external h5copy utility to move new record into 2hr file.

        warnings.filterwarnings("ignore")
        # Must use temporary file to append to a file; writing entire
        # dictionary at once also doesn't work so this is required.
        tmp_filename = self.outfile_name + '.tmp'
        Path(tmp_filename).touch()

        dd.io.save(tmp_filename, record[record_name],
                   compression=self.compression)
        f = dd.io.load(tmp_filename, '/')
        cp_cmd = 'h5copy -i {newfile} -o {full_file} -s / -d {dtstr}'
        cmd = cp_cmd.format(newfile=tmp_filename, full_file=self.outfile_name,
                            dtstr=record_name)
        sp.run(cmd.split())
        os.remove(tmp_filename)
