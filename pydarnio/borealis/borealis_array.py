# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
"""
This file contains classes for reading, and
writing of Borealis array file types. Array file
types are restructured Borealis files. Files are restructured
after being written by the radar to condense all information
from that time period into arrays. The other type of
Borealis file is the site file. Site file types
are Borealis site files, ie. stored in a record-by-record
fashion, before being converted to array types.

Classes
-------
BorealisArrayRead: Reads array-style Borealis SuperDARN file types (hdf5).
    These are the files commonly distributed and available.
BorealisArrayWrite: Writes array-style Borealis SuperDARN file types (hdf5).
    These are the files commonly distributed and available.

Exceptions
----------
BorealisFileTypeError
BorealisFieldMissingError
BorealisExtraFieldError
BorealisDataFormatTypeError
BorealisNumberOfRecordsError
BorealisVersionError
BorealisStructureError
BorealisRestructureError

See Also
--------
BorealisUtilities
BorealisSiteRead
BorealisSiteWrite

For more information on Borealis data files and how they convert to SDarn
files, see: https://borealis.readthedocs.io/en/latest/
"""
import h5py
import logging

from typing import List

from pydarnio import borealis_exceptions, borealis_formats

from .borealis_utilities import BorealisUtilities

pyDARNio_log = logging.getLogger('pyDARNio')


class BorealisArrayRead():
    """
    Class for reading Borealis 'array' filetypes. Array files have
    been restructured to no longer be read in record-by-record
    style.

    Attributes
    ----------
    filename: str
        The filename of the Borealis HDF5 file being read.
    borealis_filetype: str
        The type of Borealis file. Restructured types include:
        'bfiq'
        'antennas_iq'
        'rawacf'
    record_names: list(str)
    records: dict
    arrays: dict
    software_version : str
    format: subclass of borealis_formats.BaseFormat

    See Also
    --------
    BaseFormat
    BorealisRawacf
    BorealisBfiq
    BorealisAntennasIq
    BorealisRawrf
    """

    def __init__(self, filename: str, borealis_filetype: str):
        """
        Reads Borealis array file types into a dictionary.

        Parameters
        ----------
        filename: str
            file name containing Borealis hdf5 data.
        borealis_filetype: str
            The type of Borealis file. Types include:
            'bfiq'
            'antennas_iq'
            'rawacf'
            'rawrf'

        Raises
        ------
        OSError
            Unable to open file
        BorealisFileTypeError
            Filetype not recognized
        BorealisStructureError
            Cannot read the software version; file is incorrect structure
        BorealisVersionError
            Borealis software version format does not exist in pyDARNio
        """
        self.filename = filename

        if borealis_filetype not in ['bfiq', 'antennas_iq', 'rawacf', 'rawrf']:
            raise borealis_exceptions.\
                BorealisFileTypeError(self.filename, borealis_filetype)
        self.borealis_filetype = borealis_filetype

        # get the version of the file - split by the dash, first part should be
        # 'vX.X'
        try:
            with h5py.File(self.filename, 'r') as f:
                full_version = f.attrs['borealis_git_hash'].decode('utf-8').split('-')[0]
                version = '.'.join(full_version.split('.')[:2])      # vX.Y, ignore patch revision
        except KeyError as err:
            raise borealis_exceptions.BorealisStructureError(
                ' {} Could not find the borealis_git_hash required to '
                'determine read version (file may be site style) {}'
                ''.format(self.filename, err)) from err

        if version not in borealis_formats.borealis_version_dict:
            raise borealis_exceptions.BorealisVersionError(self.filename,
                                                           version)
        else:
            self._borealis_version = version

        self._format = borealis_formats.borealis_version_dict[
                self.software_version][self.borealis_filetype]

        # Records are private to avoid tampering.
        self._arrays = {}
        self.read_file()

    def __repr__(self):
        """ for representation of the class object"""
        # __class__.__name__ allows to grab the class name such that
        # when a class inherits this one, the class name will be the child
        # class and not the parent class
        return "{class_name}({filename}, {borealis_filetype})"\
            "".format(class_name=self.__class__.__name__,
                      filename=self.filename,
                      borealis_filetype=self.borealis_filetype)

    def __str__(self):
        """ for printing of the class object"""

        return "Reading from {filename}"\
            "".format(filename=self.filename)

    @property
    def record_names(self):
        """
        A sorted list of the set of record names in the HDF5 file read.
        These correspond to Borealis file record write times (in ms), and
        are equal to the group names in the site file types.
        """
        return sorted(list(self.records.keys()))

    @property
    def records(self):
        """
        The Borealis data in a dictionary of records, according to the
        site file format.

        Raises
        ------
        BorealisRestructureError
            Errors in restructuring to site style file.
        """
        if self.format.is_restructureable():
            try:
                records = self.format._array_to_site(self.arrays)
                BorealisUtilities.check_records(
                    self.filename, records,
                    self.format.site_single_element_types(),
                    self.format.site_array_dtypes())
            except Exception as err:
                raise borealis_exceptions.BorealisRestructureError(
                    'Arrays from {}: Error restructuring {} from array to site'
                    ' style: {}'.format(self.filename,
                                        self.format.__name__, err)) from err
        else:
            raise borealis_exceptions.BorealisRestructureError(
                'Arrays from {}: File format {} not recognized as '
                'restructureable from site to array style or vice versa.'
                ''.format(self.filename, self.format.__name__))

        return records

    @property
    def arrays(self):
        """
        The Borealis data in a dictionary of arrays, according to the
        restructured array file format.
        """
        return self._arrays

    @property
    def software_version(self):
        """
        The version of the file, taken from the 'borealis_git_hash' field, in
        the init.
        """
        return self._borealis_version

    @property
    def format(self):
        """
        The format class used for the file, from the borealis_formats module.
        """
        return self._format

    def read_file(self) -> dict:
        """
        Reads the specified Borealis file using the other functions for
        the proper file type.

        Reads the entire file.

        Returns
        -------
        arrays: dict
            Borealis data dictionary of arrays. Keys are data field names and
            unshared fields have a first dimension = number of records
            in the file.

        See Also
        --------
        BaseFormat
        """
        pyDARNio_log.info("Reading Borealis {} {} file: {}"
                          "".format(self.software_version,
                                    self.borealis_filetype, self.filename))

        attribute_types = self.format.array_single_element_types()
        dataset_types = self.format.array_array_dtypes()
        unshared_fields = self.format.unshared_fields()

        arrays = self.format.read_arrays(self.filename)
        BorealisUtilities.pulse_phase_offset_array_fix(arrays)  # To fix the problematic field stemming from deepdish
        BorealisUtilities.check_arrays(self.filename, arrays,
                                       attribute_types, dataset_types,
                                       unshared_fields)
        self._arrays = arrays

        return self._arrays


class BorealisArrayWrite():
    """
    Class for writing Borealis 'array' filetypes. Array files have
    been restructured to no longer be read in record-by-record
    style.

    Attributes
    ----------
    filename: str
        The filename of the Borealis HDF5 file being written.
    borealis_filetype: str
        The type of Borealis file. Restructured types include:
        'bfiq'
        'antennas_iq'
        'rawacf'
    record_names: list(str)
    records: dict
    arrays: dict
    software_version: str
    format: subclass of borealis_formats.BaseFormat
    compression: str
        The type of compression to write the file as. Default zlib.
        zlib is hdf5 default compression for fast reading. We want
        the fastest read possible for downstream users with this
        structure style.

    See Also
    --------
    BaseFormat
    BorealisRawacf
    BorealisBfiq
    BorealisAntennasIq
    BorealisRawrf
    """

    def __init__(self, filename: str, borealis_arrays: dict,
                 borealis_filetype: str, hdf5_compression: str = 'zlib'):
        """
        Write Borealis arrays to an array restructured file.

        Parameters
        ----------
        filename: str
            Name of the file the user wants to write to
        borealis_arrays: dict
            Borealis data dictionary. Keys are data field names and
            unshared fields have a first dimension = number of records
            in the file.
        borealis_filetype: str
            The type of Borealis file. Restructured types include:
            'bfiq'
            'antennas_iq'
            'rawacf'
        hdf5_compression: str
            String representing hdf5 compression type. Default zlib.

        Raises
        ------
        BorealisFileTypeError
            Filetype not recognized
        BorealisStructureError
            Cannot read the software version; file is incorrect structure
        BorealisVersionError
            Borealis software version format does not exist in pyDARNio
        """
        self.filename = filename
        self._arrays = borealis_arrays
        if borealis_filetype not in ['bfiq', 'antennas_iq', 'rawacf', 'rawrf']:
            raise borealis_exceptions.BorealisFileTypeError(
                self.filename, borealis_filetype)
        self.borealis_filetype = borealis_filetype

        # get the version of the file - split by the dash, first part should be
        # 'vX.X'
        try:
            version = self._arrays['borealis_git_hash'].split('-')[0]
            version = '.'.join(version.split('.')[:2])      # vX.Y, ignore patch revision
        except KeyError as err:
            raise borealis_exceptions.BorealisStructureError(
                ' {} Could not find the borealis_git_hash required to '
                'determine write version (data may be site style): {}'
                ''.format(self.filename, err)) from err

        if version not in borealis_formats.borealis_version_dict:
            raise borealis_exceptions.BorealisVersionError(self.filename,
                                                           version)
        else:
            self._borealis_version = version

        self._format = borealis_formats.borealis_version_dict[
                self.software_version][self.borealis_filetype]
        self.compression = hdf5_compression
        self.write_file()

    def __repr__(self):
        """For representation of the class object"""

        return "{class_name}({filename}, {current_record_name})"\
               "".format(class_name=self.__class__.__name__,
                         filename=self.filename,
                         current_record_name=self.current_record_name)

    def __str__(self):
        """For printing of the class object"""

        return "Writing to filename: {filename} at record name: "\
               "{current_record_name}"\
               "".format(filename=self.filename,
                         current_record_name=self.current_record_name)

    @property
    def record_names(self):
        """
        A sorted list of the set of record names in the HDF5 file read.
        These correspond to Borealis file record write times (in ms), and
        are equal to the group names in the site file types.
        """
        return sorted(list(self.records.keys()))

    @property
    def records(self):
        """
        The Borealis data in a dictionary of records, according to the
        site file format.

        Raises
        ------
        BorealisRestructureError
            Errors restructuring from arrays to site style file
        """
        if self.format.is_restructureable():
            try:
                records = self.format._array_to_site(self.arrays)
                BorealisUtilities.check_records(
                    self.filename, records,
                    self.format.site_single_element_types(),
                    self.format.site_array_dtypes())
            except Exception as err:
                raise borealis_exceptions.BorealisRestructureError(
                    'Arrays for {}: Error restructuring {} from array to site'
                    ' style: {}'.format(self.filename,
                                        self.format.__name__, err)) from err
        else:
            raise borealis_exceptions.BorealisRestructureError(
                'Arrays for {}: File format {} not recognized as '
                'restructureable from site to array style or vice versa.'
                ''.format(self.filename, self.format.__name__))

        return records

    @property
    def arrays(self):
        """
        The Borealis data in a dictionary of arrays, according to the
        restructured array file format.
        """
        return self._arrays

    @property
    def software_version(self):
        """
        The version of the file, taken from the 'borealis_git_hash' field, in
        the init.
        """
        return self._borealis_version

    @property
    def format(self):
        """
        The format class used for the file, from the borealis_formats module.
        """
        return self._format

    def write_file(self) -> str:
        """
        Write Borealis records to a file given filetype.

        Raises
        ------
        BorealisFieldMissingError - when a field is missing from the Borealis
                                file
        BorealisExtraFieldError - when an extra field is present in the
                                Borealis file
        BorealisDataFormatTypeError - when a field has the incorrect
                                field type for the Borealis file
        BorealisNumberOfRecordsError - when the number of records cannot
                                be discerned from the arrays

        See Also
        --------
        BaseFormat
        """

        pyDARNio_log.info("Writing Borealis {} {} file: {}"
                          "".format(self.software_version,
                                    self.borealis_filetype, self.filename))

        attribute_types = self.format.array_single_element_types()
        dataset_types = self.format.array_array_dtypes()
        unshared_fields = self.format.unshared_fields()
        BorealisUtilities.check_arrays(self.filename, self.arrays, attribute_types,
                                       dataset_types, unshared_fields)
        self.format.write_arrays(self.filename, self.arrays, self.compression)
        return self.filename
