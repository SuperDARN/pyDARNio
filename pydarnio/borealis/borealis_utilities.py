# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
"""
This file contains a utility class for Borealis file type
checks.

Classes
-------
BorealisUtilities: utilites class that contains static methods for
SuperDARN Borealis file type checking

Exceptions
----------
BorealisFieldMissingError
BorealisExtraFieldError
BorealisDataFormatTypeError
BorealisNumberOfRecordsError

See Also
--------
For more information on Borealis data files and how they convert to dmap,
see: https://borealis.readthedocs.io/en/latest/

Future Work
-----------
Look into combining SDARNUtilties and BorealisUtilities into a single util
class

"""
import logging
import h5py
import numpy as np
import sys

from functools import reduce
from typing import Union, List

from pydarnio import borealis_exceptions

pyDARNio_log = logging.getLogger('pyDARNio')


class BorealisUtilities():
    """
    Utility class containing static methods used by other classes.

    Contains static methods that do dictionary set calculations
    used for determining if there is any missing or extra
    fields in Borealis file types. Also does data format type checks
    for Borealis file types.

    Static Methods
    --------------
    set_diff(dict1, dict2)
        Returns a set of the difference between dict1 and dict2 keys or the
        elements of the dict1 and dict2 sets/lists
    dict_list2set(dict_list)
        Converts a list of dictionaries to a set containing their keys
    missing_field_check(filename, file_struct_list, parameter_names,
            [record_name])
        Checks if there is any missing fields in the record/file from
        a list of possible file fields
    extra_field_check(filename, file_struct_list, parameter_names,
            [record_name])
        Checks if there is any extra fields in the record/file from
        a list of possible file fields
    record_incorrect_types_check(filename, attributes_type_dict,
            datasets_type_dict, record, record_name)
        Checks if there is any incorrect types in the record from
        a list of possible file fields and their data type formats
    array_incorrect_types_check(filename, attributes_type_dict,
            datasets_type_dict, unshared_parameters, file_data)
        Checks if there are any incorrect types in the file data
        from a list of possible fields and their formats, and whether
        the data is a shared field in the array file or not.
    array_num_records_check(filename, unshared_parameters, file_data)
        Checks if there is a consistent number of records in the arrays
        from the array file.
    """

    @staticmethod
    def set_diff(dict1: Union[dict, set, list],
                 dict2: Union[dict, set, list]) -> set:
        """
        Determines the difference in the key set from the
        first dictionary to the second dictionary.
        ex) Let A = {a, b, c} and B = {d, a, b}
        Then A - B = {c}

        Parameters
        ----------
        dict1: dict or set
            dictionary or set to subtract from
        dict2: dict or set
            dictionary or set subtracting from dict1

        Returns
        -------
        dict_keys: set
            difference between dict1 and dict2 keys or the sets/lists
        """
        diff_keys = set(dict1) - set(dict2)
        return diff_keys

    @staticmethod
    def dict_list2set(dict_list: List[dict]) -> set:
        """
        Converts a list of dictionaries to list of sets

        Parameters
        ----------
        dict_list: list
            list of dictionaries

        Returns
        -------
        complete_set: set
            set containing all dictionary key from the list of dicts
        """
        # convert dictionaries to set to do some set magic
        sets = [set(dic) for dic in dict_list]
        # create a complete set list
        # * - expands the list out into multiple set arguments
        # then the union operator creates it into a full unique set
        # example:
        # s = [{'a','v'}, {'v','x'}] => set.union(*s) = {'a', 'v', 'x'}
        complete_set = set.union(*sets)
        return complete_set

    @staticmethod
    def missing_field_check(filename: str, file_struct_list: List[dict],
                            parameter_names: Union[List[str], dict, set],
                            **kwargs):
        """
        Checks if any fields are missing from the file/record compared to the
        file structure.

        Parameters
        ----------
        filename: str
            Name of the file being checked
        file_struct_list: List[dict]
            List of dictionaries for the possible file structure fields
        parameter_names: List[str], dict, set
            List of parameter names, or set, or dict, in the file or in
            the record.
        record_name: str
            The name of the record (first sequence start time), if in a
            record style file.

        Raises
        -------
        BorealisFieldMissingError
        BorealisStructureError

        Notes
        -----
        Checks sets and subsets. Any missing fields are a problem because
        Borealis field names are well-defined.
        """
        missing_fields = set()
        for file_struct in file_struct_list:
            diff_fields = BorealisUtilities.set_diff(file_struct,
                                                     parameter_names)
            if len(diff_fields) != 0:
                missing_fields = missing_fields.union(diff_fields)

        if len(missing_fields) == \
                reduce(lambda a, b: a+b, [len(x) for x in file_struct_list]):
            # all fields are missing
            if 'record_name' in kwargs.keys():
                raise borealis_exceptions.\
                        BorealisStructureError("All fields expected are "
                                               "missing in record {}: {}"
                                               "".format(kwargs['record_name'],
                                                         missing_fields))
            else:
                raise borealis_exceptions.\
                        BorealisStructureError("All fields expected are"
                                               " missing : {}"
                                               "".format(missing_fields))

        if len(missing_fields) > 0:
            if 'record_name' in kwargs.keys():
                raise borealis_exceptions.BorealisFieldMissingError(
                    filename, missing_fields,
                    record_name=kwargs['record_name'])
            else:
                raise borealis_exceptions.\
                        BorealisFieldMissingError(filename,
                                                  missing_fields)

    array_missing_field_check = record_missing_field_check =\
        missing_field_check

    @staticmethod
    def extra_field_check(filename: str, file_struct_list: List[dict],
                          parameter_names: Union[List[str], dict, set],
                          **kwargs):
        """
        Check if there is an extra field in the file/record.

        Parameters
        ----------
        filename: str
            Name of the file being checked
        file_struct_list: List[dict]
            List of dictionaries for the possible file structure fields
        parameter_names: List[str], dict, set
            List of parameter names, or set, or dict, in the file or in
            the record.
        record_name: str
            Record name for better error message information, if in a
            record style file.

        Raises
        ------
        BorealisExtraFieldError
        """
        file_struct = BorealisUtilities.dict_list2set(file_struct_list)
        extra_fields = BorealisUtilities.set_diff(parameter_names, file_struct)

        if len(extra_fields) > 0:
            if 'record_name' in kwargs.keys():
                raise borealis_exceptions.BorealisExtraFieldError(
                    filename, extra_fields,
                    record_name=kwargs['record_name'])
            else:
                raise borealis_exceptions.\
                        BorealisExtraFieldError(filename, extra_fields)

    array_extra_field_check = record_extra_field_check = extra_field_check

    @staticmethod
    def record_incorrect_types_check(filename: str, attributes_type_dict: dict,
                                     datasets_type_dict: dict, record: dict,
                                     record_name: str):
        """
        Checks if the record's data type formats are correct according
        to the file structure fields data type formats.

        Checks both single element types and numpy array dtypes separately.

        Parameters
        ----------
        filename: str
            Name of the file being checked
        attributes_type_dict: dict
            Dictionary with the required types for the attributes in the file.
        datasets_type_dict: dict
            Dictionary with the require dtypes for the numpy array types in the
            file.
        record: dict
            Record dictionary
        record_name: str
            Record name for a better error message information

        Raises
        ------
        BorealisDataFormatTypeError
        """
        incorrect_types_check = {param: str(attributes_type_dict[param])
                                 for param in attributes_type_dict.keys()
                                 if type(record[param]) !=
                                 attributes_type_dict[param] and
                                 record[param].shape is not None}

        incorrect_types_check.update({param: 'np.ndarray of ' +
                                      str(datasets_type_dict[param])
                                      for param in datasets_type_dict.keys()
                                      if record[param].dtype.type !=
                                      datasets_type_dict[param]})
        if len(incorrect_types_check) > 0:
            raise borealis_exceptions.BorealisDataFormatTypeError(
                filename, incorrect_types_check, record_name=record_name)

    @staticmethod
    def array_incorrect_types_check(filename: str, attributes_type_dict: dict,
                                    datasets_type_dict: dict, file_data: dict):
        """
        Checks if the file's data type formats are correct according
        to the file structure fields data type formats and whether or
        not the field is a shared field between records.

        Checks both single element types and numpy array dtypes separately.

        Parameters
        ----------
        filename: str
            Name of the file being checked
        attributes_type_dict: dict
            Dictionary with the required types for the attributes in the file.
        datasets_type_dict: dict
            Dictionary with the require dtypes for the numpy
            arrays in the file.
        file_data: dict
            dictionary containing all file information.

        Raises
        ------
        BorealisDataFormatTypeError
        BorealisStructureError
        """

        attributes_type_dict_keys = sorted(list(attributes_type_dict.keys()))
        np_array_types = [isinstance(file_data[param], np.ndarray) for param
                          in attributes_type_dict_keys]

        # There is a better way to do this TODO
        array_keys = [attributes_type_dict_keys[x] for x, y in
                      enumerate(np_array_types) if y]  # True
        array_fields = [(field, attributes_type_dict[field]) for field in
                        array_keys]

        if any(np_array_types):
            raise borealis_exceptions.\
                    BorealisStructureError('Unexpected array in the '
                                           'following fields where single'
                                           ' attributes expected:  {}'
                                           ''.format(array_fields))

        incorrect_types_check = {param: str(attributes_type_dict[param])
                                 for param in attributes_type_dict.keys()
                                 if type(file_data[param]) !=
                                 attributes_type_dict[param] and
                                 file_data[param].shape is not None}

        datasets_type_dict_keys = sorted(list(datasets_type_dict.keys()))
        np_array_types = [isinstance(file_data[param], np.ndarray) for param in
                          datasets_type_dict_keys]
        array_keys = [datasets_type_dict_keys[x] for x, y in
                      enumerate(np_array_types) if not y]
        non_array_fields = [(field, datasets_type_dict[field]) for field in
                            array_keys]

        if not all(np_array_types):
            raise borealis_exceptions.\
                    BorealisStructureError('Unexpected single attribute in the'
                                           ' following fields where arrays of '
                                           'given dtype are expected: {}'
                                           ''.format(non_array_fields))

        incorrect_types_check.update({param: 'np.ndarray of ' +
                                      str(datasets_type_dict[param])
                                      for param in datasets_type_dict.keys()
                                      if file_data[param].dtype.type !=
                                      datasets_type_dict[param] and
                                      file_data[param].dtype.type != np.str_})
        if 'pulse_phase_offset' in incorrect_types_check.keys():
            if file_data['pulse_phase_offset'].dtype.type == np.int64 and file_data['pulse_phase_offset'].shape == (2,):
                incorrect_types_check.pop('pulse_phase_offset')     # This field is problematic, hack to ignore

        if len(incorrect_types_check) > 0:
            raise borealis_exceptions.\
                    BorealisDataFormatTypeError(filename,
                                                incorrect_types_check)

    @staticmethod
    def array_num_records_check(filename: str, unshared_parameters: List[str],
                                file_data: dict):
        """
        Checks the number of records in all unshared parameter fields to
        ensure they are the same for data integrity.

        Parameters
        ----------
        filename: str
            Name of the file being checked
        unshared_parameters: List[str]
            List of parameter names that are not shared between all the records
            in the array restructured file, i.e. will have a dimension = to
            number of records.
        file_data: dict
            dictionary containing all file information.

        Raises
        ------
        BorealisNumberOfRecordsError
        """
        num_records = {parameter: file_data[parameter].shape for parameter
                       in unshared_parameters}

        dimensions = list(num_records.values())
        try:
            dim_compliance = {k: v[0] == dimensions[0][0] for k, v in num_records.items()}
            # if 'pulse_phase_offset' in dim_compliance.keys() and dim_compliance['pulse_phase_offset'] == False:
            #     dim_compliance.pop('pulse_phase_offset')    # This field is broken in older versions
            if not all(dim_compliance.values()):
                raise borealis_exceptions.\
                        BorealisNumberOfRecordsError(filename, num_records)
        except IndexError:  # some fields are not arrays!
            tb = sys.exc_info()[2]
            raise borealis_exceptions.\
                BorealisNumberOfRecordsError(filename,
                                             num_records).with_traceback(tb)

    @classmethod
    def check_arrays(cls, origin_string: str, arrays: dict,
                     attribute_types: dict, dataset_types: dict,
                     unshared_fields: List[str]):
        """
        Parameters
        ----------
        origin_string: str
            Name of file to be checked or other origin descriptor.
        arrays: dict
            Dictionary of arrays to be checked.
        attribute_types: dict
            Dictionary with the required types for the attributes in the file.
        dataset_types: dict
            Dictionary with the require dtypes for the numpy arrays in the
            file.
        unshared_fields: List[str]
            List of fields that are not shared between the records and
            therefore should be an array with first dimension = number of
            records

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
        array_missing_field_check(filename, format_fields, parameters)
                        - checks for missing fields. See this
                        method for information on why we use format_fields.
        array_extra_field_check(filename, format_fields, parameters)
                        - checks for extra fields in the record
        array_incorrect_types_check(filename, attribute_types_dict,
                        dataset_types_dict, unshared_fields, file_data)
                         - checks for incorrect data types for file fields
        array_num_records_check(filename, unshared_fields, file_data)
        """
        all_format_fields = [attribute_types, dataset_types]
        cls.array_missing_field_check(origin_string, all_format_fields, arrays)
        cls.array_extra_field_check(origin_string, all_format_fields, arrays)
        cls.array_incorrect_types_check(origin_string, attribute_types,
                                        dataset_types, arrays)
        cls.array_num_records_check(origin_string, unshared_fields, arrays)

    @classmethod
    def check_records(cls, origin_string: str, records: dict,
                      attribute_types: dict, dataset_types: dict):
        """
        Do checks on the restructured data the same as would be done as if
        the dictionary had just been read from file.

        Parameters
        ----------
        origin_string: str
            Name of file to be checked or other origin descriptor.
        records: dict
            Dictionary of records to be checked for errors.
        attribute_types: dict
            Dictionary with the required types for the attributes in the file.
        dataset_types: dict
            Dictionary with the require dtypes for the numpy arrays in the
            file.

        Raises
        ------
        OSError: file does not exist
        BorealisFieldMissingError - when a field is missing from the Borealis
                                file/stream type
        BorealisExtraFieldError - when an extra field is present in the
                                Borealis file/stream type
        BorealisDataFormatTypeError - when a field has the incorrect
                                field type for the Borealis file/stream type

        See Also
        --------
        record_missing_field_check(filename, format_fields, record,
                        record_name) - checks for missing fields. See this
                        method for information on why we use format_fields.
        record_extra_field_check(filename, format_fields, record, record_name)
                        - checks for extra fields in the record
        record_incorrect_types_check(filename, attribute_types_dict,
                        dataset_types_dict, record, record_name) - checks
                        for incorrect data types for file fields
        """
        all_format_fields = [attribute_types, dataset_types]

        all_records_good = True
        records_with_missing_fields = dict()
        records_with_extra_fields = dict()
        records_with_incorrect_types = dict()
        for record_name, record in records.items():
            try:
                cls.record_missing_field_check(origin_string, all_format_fields,
                                               record, record_name=record_name)
                cls.record_extra_field_check(origin_string, all_format_fields,
                                             record, record_name=record_name)
                cls.record_incorrect_types_check(origin_string, attribute_types,
                                                 dataset_types, record,
                                                 record_name)
            except borealis_exceptions.BorealisFieldMissingError as err:
                records_with_missing_fields[record_name] = err.fields
                all_records_good = False
            except borealis_exceptions.BorealisExtraFieldError as err:
                records_with_extra_fields[record_name] = err.fields
                all_records_good = False
            except borealis_exceptions.BorealisDataFormatTypeError as err:
                records_with_incorrect_types[record_name] = err.incorrect_types
                all_records_good = False
        if not all_records_good:
            raise borealis_exceptions.BorealisBadRecordsError(origin_string, records_with_missing_fields,
                                                              records_with_extra_fields, records_with_incorrect_types)

    @staticmethod
    def get_record_names(filename: str):
        """
        Gets the top-level names of the groups and attributes stored in the
        HDF5 file specified.

        Parameters
        ----------
        filename
            Borealis file to read. Either array- or site-structured.

        Returns
        -------
        record_names
            List of the top-level keys of the HDF5 file.
        """
        hdf5_default_attrs = \
            ['CLASS', 'DEEPDISH_IO_VERSION', 'PYTABLES_FORMAT_VERSION',
             'TITLE', 'VERSION']
        with h5py.File(filename, 'r') as f:
            key_view = f.keys()
            scalars = f.attrs  # Attributes for the HDF5 file, including scalar
            # fields
            record_names = [key for key in key_view]
            record_names.extend(
                [val for val in scalars if val not in hdf5_default_attrs])

        return record_names

    @staticmethod
    def get_borealis_structure(record_names):
        """
        Determines the type of Borealis HDF5 file structure based on
        the names of the top-level records in the HDF5 file.

        Parameters
        ----------
        record_names
            List of names of the top-level groups and attributes of the file.

        Returns
        -------
        structure
            The Borealis HDF5 file structure of the input file. Either
            'site' or 'array'.
        """
        if 'borealis_git_hash' in record_names:
            structure = 'array'
        else:
            structure = 'site'
        return structure

    @staticmethod
    def get_borealis_version(filename: str, record_names, structure: str):
        """
        Gets the Borealis version of the file from the borealis_git_hash.

        Parameters
        ----------
        filename
            Path to Borealis data file.
        record_names
            List of names of the top-level groups and attributes of the file.
        structure
            Structure of the file. Supported structures are 'site' and 'array'.

        Returns
        -------
        version
            The Borealis version of the file. Formatted as 'v0.5'
        """
        if structure == 'array':
            try:
                with h5py.File(filename, 'r') as f:
                    borealis_git_hash = f.attrs['borealis_git_hash'].decode('utf-8')
            except KeyError as err:
                raise borealis_exceptions.BorealisStructureError(
                    ' {} Could not find the borealis_git_hash required to '
                    'determine file version. Data file may be corrupted. {}'
                    ''.format(filename, err)) from err
        elif structure == 'site':
            try:
                with h5py.File(filename, 'r') as f:
                    records = sorted(list(f.keys()))
                    first_rec = f[records[0]]
                    borealis_git_hash = first_rec.attrs['borealis_git_hash']\
                                            .decode('utf-8')
            except KeyError as err:
                raise borealis_exceptions.BorealisStructureError(
                    ' {} Could not find the borealis_git_hash required to '
                    'determine file version. Data file may be corrupted. {}'
                    ''.format(filename, err)) from err
        else:
            raise borealis_exceptions.BorealisStructureError(
                ' {} Could not find the borealis_git_hash required to '
                'determine file version. Data file may be corrupted.'
                ''.format(filename))

        version = borealis_git_hash.split('-')[0]
        version = '.'.join(version.split('.')[:2])  # vX.Y, ignore patch revision

        return version

    @staticmethod
    def pulse_phase_offset_array_fix(data_dict: dict):
        """
        Fixes the dtype and shape of the pulse_phase_offset field. Modifies data_dict in place.
        """
        if 'pulse_phase_offset' not in data_dict:
            return
        ppo = data_dict['pulse_phase_offset']
        if ppo.shape == (2,):  # The field is broken, was empty when written to file and so was restructured improperly
            data_dict['pulse_phase_offset'] = np.zeros((data_dict['num_sequences'].shape[0], 0), dtype=np.float32)

    @staticmethod
    def pulse_phase_offset_site_fix(data_dict: dict):
        """
        Fixes the dtype and shape of the pulse_phase_offset field. Modifies data_dict in place.
        """
        for rec in data_dict.values():
            if 'pulse_phase_offset' not in rec:
                continue
            ppo = rec['pulse_phase_offset']
            if ppo.shape != rec['pulses'].shape:  # The field is broken, was empty when written to file and so was restructured improperly
                rec['pulse_phase_offset'] = np.zeros(rec['pulses'].shape, dtype=np.float32)