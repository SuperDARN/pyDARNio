# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
"""
This file contains several classes with the fields that pertain to
SuperDARN Borealis HDF5 files. All formats inherit from BaseFormat.

Classes
-------
BorealisRawacf
BorealisBfiq
BorealisAntennasIq
BorealisRawrf
as well as previous versions of these classes, currently including
BorealisRawacfv0_6
BorealisBfiqv0_6
BorealisAntennasIqv0_6
BorealisRawrfv0_6
BorealisRawacfv0_5
BorealisBfiqv0_5
BorealisAntennasIqv0_5
BorealisRawrfv0_5
BorealisRawacfv0_4
BorealisBfiqv0_4
BorealisAntennasIqv0_4
BorealisRawrfv0_4

Globals
-------
borealis_version_dict
    A lookup table for [version][filetype] that provides the appropriate class
    given the version and filetype strings.

Notes
-----
- Debug data files such as Borealis stage iq data (an intermediate
  product that can be generated during filtering and decimating, showing
  progression from rawrf to output ptrs iq files) will not be included here.
  This is a debug format only and should not be used for higher level
  data products.

See Also
--------
- BaseFormat
    It is critical to understand the BaseFormat methods and design concept
    to understand how each of the format class' methods work, and how they
    are used in restructuring from site to array structure (and vice versa)
    for each format.
- The Borealis documentation on formats is at
  https://borealis.readthedocs.io/en/latest/borealis_data.html
"""

import copy
import h5py
import numpy as np
from typing import List

from collections import OrderedDict

from .base_format import BaseFormat


class BorealisFieldsv0_4():
    """
    Class containing the mapping of Borealis data fields and types for each 
    Borealis file type for Borealis v0.4 and earlier.

    See Also
    --------
    BorealisFields (most up to date format)
    """

    @classmethod
    def files_with_fields(cls):
        """
        Get the mapping of Borealis data fields to each file type.

        Returns
        -------
        A dictionary containing data fields as keys and a list of file types
        as the values.
        """
        return {
            "borealis_git_hash": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "experiment_id": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "experiment_name": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "experiment_comment": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "slice_comment": ['antennas_iq', 'bfiq', 'rawacf'],
            "num_slices": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "station": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "num_sequences": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "range_sep": ['bfiq', 'rawacf'],
            "first_range_rtt": ['bfiq', 'rawacf'],
            "first_range": ['bfiq', 'rawacf'],
            "rx_sample_rate": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "scan_start_marker": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "int_time": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "tx_pulse_len": ['antennas_iq', 'bfiq', 'rawacf'],
            "tau_spacing": ['antennas_iq', 'bfiq', 'rawacf'],
            "main_antenna_count": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "intf_antenna_count": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "freq": ['antennas_iq', 'bfiq', 'rawacf'],
            "samples_data_type": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "data_normalization_factor": ['antennas_iq', 'bfiq', 'rawacf'],
            "num_beams": ['antennas_iq', 'bfiq', 'rawacf'],
            "pulses": ['antennas_iq', 'bfiq', 'rawacf'],
            "lags": ['bfiq', 'rawacf'],
            "blanked_samples": ['bfiq', 'rawacf'],
            "sqn_timestamps": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "beam_nums": ['antennas_iq', 'bfiq', 'rawacf'],
            "beam_azms": ['antennas_iq', 'bfiq', 'rawacf'],
            "noise_at_freq": ['antennas_iq', 'bfiq', 'rawacf'],
            "correlation_descriptors": ['rawacf'],
            "correlation_dimensions": ['rawacf'],
            "main_acfs": ['rawacf'],
            "intf_acfs": ['rawacf'],
            "xcfs": ['rawacf'],
            "num_samps": ['antennas_iq', 'bfiq', 'rawrf'],
            "num_ranges": ['bfiq'],
            "pulse_phase_offset": ['antennas_iq', 'bfiq'],
            "antenna_arrays_order": ['antennas_iq', 'bfiq'],
            "data": ['antennas_iq', 'bfiq', 'rawrf'],
            "data_descriptors": ['antennas_iq', 'bfiq', 'rawrf'],
            "data_dimensions": ['antennas_iq', 'bfiq', 'rawrf'],
            "rx_center_freq": ['rawrf'],
        }

    @classmethod
    def all_single_element_types(cls):
        """
        Get the mapping of Borealis data fields to its single element type.

        Returns
        -------
        A dictionary containing data fields as keys and the data field variable
        types as values.
        """
        return {
            # Identifies the version of Borealis that made this data. Necessary
            # for all versions.
            "borealis_git_hash": str,
            # Number used to identify experiment.
            "experiment_id": np.int64,
            # Name of the experiment file.
            "experiment_name": str,
            # Comment about the whole experiment
            "experiment_comment": str,
            # Additional text comment that describes the slice.
            "slice_comment": str,
            # Number of slices in the experiment at this integration time.
            "num_slices": np.int64,
            # Three letter radar identifier.
            "station": str,
            # Number of sampling periods in the integration time.
            "num_sequences": np.int64,
            # range gate separation (equivalent distance between samples), km.
            "range_sep": np.float32,
            # Round trip time of flight to first range in microseconds.
            "first_range_rtt": np.float32,
            # Distance to first range in km.
            "first_range": np.float32,
            # Sampling rate of the samples being written to file in Hz.
            "rx_sample_rate": np.float64,
            # Designates if the record is the first in a scan.
            "scan_start_marker": np.uint8,
            # Integration time in seconds.
            "int_time": np.float32,
            # Length of the pulse in microseconds.
            "tx_pulse_len": np.uint32,
            # The minimum spacing between pulses, spacing between pulses is
            # always a multiple of this in microseconds.
            "tau_spacing": np.uint32,
            # Number of main array antennas.
            "main_antenna_count": np.uint32,
            # Number of interferometer array antennas.
            "intf_antenna_count": np.uint32,
            # The frequency used for this experiment slice in kHz.
            "freq": np.uint32,
            # str denoting C data type of the samples included in the data
            # array, such as 'complex float'.
            "samples_data_type": str,
            # data normalization factor determined by the filter scaling in the
            # decimation scheme.
            "data_normalization_factor": np.float64,
            # number of beams calculated for the integration time.
            "num_beams": np.uint32,
            # Number of samples in the sampling period.
            "num_samps": np.uint32,
            # Number of ranges to calculate correlations for.
            "num_ranges": np.uint32,
            # The center frequency of this data in kHz
            "rx_center_freq": np.float64,
        }

    @classmethod
    def all_array_types(cls):
        """
        Get the mapping of Borealis array data fields to its type

        Returns
        -------
        A dictionary containing data fields as keys and the data field variable
        types as values.
        """
        return {
            # The pulse sequence in multiples of the tau_spacing.
            "pulses": np.uint32,
            # The lags created from combined pulses.
            "lags": np.uint32,
            # Samples that have been blanked during TR switching.
            "blanked_samples": np.uint32,
            # A list of GPS timestamps of the beginning of transmission for
            # each sampling period in the integration time. Seconds since
            # epoch.
            "sqn_timestamps": np.float64,
            # A list of beam numbers used in this slice.
            "beam_nums": np.uint32,
            # A list of the beams azimuths for each beam in degrees.
            "beam_azms": np.float64,
            # Noise at the receive frequency, should be an array
            # (one value per sequence) (TODO units??) (TODO document
            # FFT resolution bandwidth for this value, should be =
            # output_sample rate?)
            "noise_at_freq": np.float64,
            # Denotes what each acf/xcf dimension represents. = "num_beams",
            # "num_ranges", "num_lags" in site rawacf files.
            "correlation_descriptors": np.unicode_,
            # The dimensions in which to reshape the acf/xcf data.
            "correlation_dimensions": np.uint32,
            # Main array autocorrelations
            "main_acfs": np.complex64,
            # Interferometer array autocorrelations
            "intf_acfs": np.complex64,
            # Crosscorrelations between main and interferometer arrays
            "xcfs": np.complex64,
            # States what order the data is in. Describes the data layout.
            "antenna_arrays_order": np.unicode_,
            # Denotes what each data dimension represents. =
            # "num_antenna_arrays", "num_sequences", "num_beams", "num_samps"
            # for site bfiq.
            "data_descriptors": np.unicode_,
            # The dimensions in which to reshape the data.
            "data_dimensions": np.uint32,
            # For pulse encoding phase, in degrees offset.
            # Contains one phase offset per pulse in pulses.
            "pulse_phase_offset": np.float32,
            # A contiguous set of samples (complex float) at given sample rate
            "data": np.complex64
        }

    @classmethod
    def single_element_types(cls, file_type: str) -> dict:
        """
        Gets the single element types of a given file type.

        Parameters
        ----------
        file_type: str
            File type to get single element fields for. One of 'antennas_iq', 
            'bfiq', 'rawacf', or 'rawrf'

        Returns
        -------
        dict
            Dictionary of field: type for all fields contained in file_type.
        """
        relevant_fields = [k for k, v in cls.files_with_fields().items() if file_type in v]
        single_elements = cls.all_single_element_types()
        return {k: v for k, v in single_elements.items() if k in relevant_fields}

    @classmethod
    def array_types(cls, file_type: str) -> dict:
        """
        Gets the array types of a given file type.

        Parameters
        ----------
        file_type: str
            File type to get array fields for. One of 'antennas_iq', 'bfiq', 
            'rawacf', or 'rawrf'

        Returns
        -------
        dict
            Dictionary of field: type for all fields contained in file_type.
        """
        relevant_fields = [k for k, v in cls.files_with_fields().items() if file_type in v]
        array_elements = cls.all_array_types()
        return {k: v for k, v in array_elements.items() if k in relevant_fields}

    @classmethod
    def all_shared_fields(cls):
        """
        List of all fields that are shared between records in a site-structured file.

        Notes
        -----
        The dimension info for shared_fields is not necessary because the
        dimensions will be the same for site and restructured files.
        """
        return ['antenna_arrays_order',
                'blanked_samples',
                'borealis_git_hash',
                'data_normalization_factor',
                'experiment_comment',
                'experiment_id',
                'experiment_name',
                'first_range',
                'first_range_rtt',
                'freq',
                'intf_antenna_count',
                'lags',
                'main_antenna_count',
                'num_ranges',
                'num_samps',
                'pulse_phase_offset',
                'pulses',
                'range_sep',
                'rx_sample_rate',
                'samples_data_type',
                'slice_comment',
                'station',
                'tau_spacing',
                'tx_pulse_len']

    @classmethod
    def shared_fields(cls, file_type: str) -> list[str]:
        """
        Gets the shared fields of a given file type.

        Parameters
        ----------
        file_type: str
            File type to get shared fields for. One of 'antennas_iq', 'bfiq',
            'rawacf', or 'rawrf'

        Returns
        -------
        list[str]
            List of shared fields for the given data type.
        """
        relevant_fields = [k for k, v in cls.files_with_fields().items() if file_type in v]
        shared = cls.all_shared_fields()
        return [k for k in shared if k in relevant_fields]


class BorealisFieldsv0_5(BorealisFieldsv0_4):
    """
    Class containing the mapping of Borealis data fields and types for each 
    Borealis file type for Borealis v0.5.

    See Also
    --------
    BorealisFields (most up to date format)
    """

    @classmethod
    def files_with_fields(cls):
        """
        Get the mapping of Borealis data fields to each file type.
        Mapping is updated to reflect changes from previous version of Borealis.

        Returns
        -------
        A dictionary containing data fields as keys and a list of file types
        as the values.
        """
        field_file_mapping = super().files_with_fields()
        field_file_mapping.update({
            "slice_id": ['antennas_iq', 'bfiq', 'rawacf'],
            "slice_interfacing": ['antennas_iq', 'bfiq', 'rawacf'],
            "scheduling_mode": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "averaging_method": ['rawacf'],
            "num_blanked_samples": ['antennas_iq', 'bfiq', 'rawacf'],
        })
        field_file_mapping['blanked_samples'].extend(['antennas_iq', 'rawrf'])

        return field_file_mapping

    @classmethod
    def all_single_element_types(cls):
        """
        Get the mapping of Borealis data fields to its single element type.
        Mapping is updated to reflect changes from previous version of Borealis.

        Returns
        -------
        A dictionary containing data fields as keys and the data field variable
        types as values.
        """
        single_element_types = super().all_single_element_types()
        single_element_types.update({
            # the slice id of the file and dataset.
            "slice_id": np.uint32,
            # the interfacing of this slice to other slices.
            "slice_interfacing": str,
            # A string describing the type of scheduling time at the time of
            # this dataset.
            "scheduling_mode": str,
            # A string describing the averaging method, ex. mean, median
            "averaging_method": str,
            # number of blanked samples in the sequence.
            "num_blanked_samples": np.uint32
        })
        return single_element_types

    @classmethod
    def all_shared_fields(cls):
        """
        List of all fields that are shared between records in a site-structured file.

        Notes
        -----
        In Borealis v0.5, slice_id, scheduling_mode, and
        averaging_method were added and these will be shared fields. These
        fields will not change from record to record. Blanked samples may
        change from record to record if a new slice is added and interfaced
        within the sequence. Therefore, this bug was fixed by changing
        blanked_samples to an unshared field in Borealis v0.5.
        """
        shared = super().all_shared_fields() + \
            ['slice_id', 'scheduling_mode', 'averaging_method']
        shared.remove('blanked_samples')
        return shared


class BorealisFieldsv0_6(BorealisFieldsv0_5):
    """
    Class containing the mapping of Borealis data fields and types for each 
    Borealis file type for Borealis v0.6.

    See Also
    --------
    BorealisFields (most up to date format)
    """

    @classmethod
    def files_with_fields(cls):
        """
        Get the mapping of Borealis data fields to each file type.
        Mapping is updated to reflect changes from previous version of Borealis.

        Returns
        -------
        A dictionary containing data fields as keys and a list of file types
        as the values.
        """
        field_file_mapping = super().files_with_fields()
        field_file_mapping.update({
            "agc_status_word": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "lp_status_word": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "gps_locked": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
            "gps_to_system_time_diff": ['antennas_iq', 'bfiq', 'rawacf', 'rawrf'],
        })

        return field_file_mapping

    @classmethod
    def all_single_element_types(cls):
        """
        Get the mapping of Borealis data fields to its single element type.
        Mapping is updated to reflect changes from previous version of Borealis.

        Returns
        -------
        A dictionary containing data fields as keys and the data field variable
        types as values.
        """
        single_element_types = super().all_single_element_types()
        single_element_types.update({
            # the agc fault status of each transmitter, transmitter/USRP
            # mapped to bit position
            # A '1' indicates an agc fault at least once during the integration
            # period.
            "agc_status_word": np.uint32,
            # the low power status of each transmitter, transmitter/USRP
            # mapped to bit position
            # A '1' indicates a low power condition at least once during the
            # integration period.
            "lp_status_word": np.uint32,
            # Boolean indicating if the GPS was locked during the entire
            # integration period
            "gps_locked": np.uint8,
            # The max time diffe between GPS and system time during the
            # integration period. In seconds. Negative if GPS time ahead.
            "gps_to_system_time_diff": np.float64,
            # Updated to 16 bit number to avoid mismatch when converting
            # to DMAP format.
            "experiment_id": np.int16
        })
        return single_element_types


class BorealisFields(BorealisFieldsv0_6):
    """
    Class containing the mapping of Borealis data fields and types for each 
    Borealis file type for the current version of Borealis
    """

    @classmethod
    def files_with_fields(cls):
        """
        Get the mapping of Borealis data fields to each file type. 
        Mapping is updated to reflect changes from previous version of Borealis.

        Returns
        -------
        A dictionary containing data fields as keys and a list of file types
        as the values.
        """
        field_file_mapping = super().files_with_fields()
        field_file_mapping.pop('correlation_descriptors')
        field_file_mapping.pop('correlation_dimensions')
        field_file_mapping['data_descriptors'].append('rawacf')
        field_file_mapping['data_dimensions'].append('rawacf')
        field_file_mapping['first_range'].append('antennas_iq')
        field_file_mapping['first_range_rtt'].append('antennas_iq')
        field_file_mapping['lags'].append('antennas_iq')
        field_file_mapping['num_ranges'].append('antennas_iq')
        field_file_mapping['range_sep'].append('antennas_iq')

        return field_file_mapping

    @classmethod
    def all_single_element_types(cls):
        """
        Get the mapping of Borealis data fields to its single element type.
        Mapping is updated to reflect changes from previous version of Borealis.

        Returns
        -------
        A dictionary containing data fields as keys and the data field variable
        types as values.

        Notes
        -----
        In v0.7, gps_locked and scan_start_marker were changed to np.bool_ fields from np.uint8.
        """
        single_element_types = super().all_single_element_types()
        single_element_types.update({
            "gps_locked": np.bool_,
            "scan_start_marker": np.bool_,
        })
        return single_element_types

    @classmethod
    def all_array_types(cls):
        """
        Get the mapping of Borealis array data fields to its type

        Returns
        -------
        A dictionary containing data fields as keys and the data field variable
        types as values.

        Notes
        -----
        In v0.7, antenna_arrays_order and data_descriptors were changed to np.array(np.bytes_).
        """
        all_arrays = super().all_array_types()
        all_arrays.update({
            "antenna_arrays_order": np.bytes_,
            "data_descriptors": np.bytes_,
        })
        all_arrays.pop('correlation_dimensions')
        all_arrays.pop('correlation_descriptors')
        return all_arrays


class BorealisRawacfv0_4(BaseFormat):
    """
    Class containing Borealis Rawacf data fields and their types.

    See Also
    --------
    BaseFormat
    BorealisRawacf (most up to date format)

    Notes
    -----
    Rawacf data has been mixed, filtered, and decimated; beamformed and
    combined into antenna arrays; then autocorrelated and correlated between
    antenna arrays to produce matrices of num_ranges x num_lags.

    See BaseFormat for description of classmethods and some staticmethods
    and how they are used to verify format files and restructure Borealis
    files to array and site structure.

    Static Methods
    --------------
    find_num_ranges(OrderedDict): int
        Returns num ranges in the data for use in finding dimensions
    find_num_lags(OrderedDict): int
        Returns the num lags in the data for use in finding dimensions
    """
    fields = BorealisFieldsv0_4

    @staticmethod
    def find_num_ranges(records: OrderedDict) -> int:
        """
        Find the number of ranges given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_ranges
            The number of ranges being calculated in the acfs.

        Notes
        -----
        Num_ranges is unique to a slice so cannot change inside file.
        """
        first_key = list(records.keys())[0]
        num_ranges = records[first_key]['correlation_dimensions'][1]
        return num_ranges

    @staticmethod
    def find_num_lags(records: OrderedDict) -> int:
        """
        Find the number of lags given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_lags
            The number of lags being calculated in the acfs.

        Notes
        -----
        Num_lags is unique to a slice so cannot change inside file.
        """
        first_key = list(records.keys())[0]
        num_lags = records[first_key]['correlation_dimensions'][2]
        return num_lags

    @staticmethod
    def reshape_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the main_acfs,
            intf_acfs, and xcfs fields in all records reshaped to the correct
            dimensions.

        Notes
        -----
        BorealisRawacf has the correlation fields flattened in the
        site structured files, so this field is reshaped in here.
        """

        # dimensions provided in correlation_dimensions field as num_beams,
        # num_ranges, num_lags for the rawacf format.
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            record_dimensions = new_records[key]['correlation_dimensions']
            for field in ['main_acfs', 'intf_acfs', 'xcfs']:
                new_records[key][field] = new_records[key][field].\
                                        reshape(record_dimensions)

        return new_records

    @staticmethod
    def flatten_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the correlation
            fields in all records flattened as is the convention
            in site structured files.

        Notes
        -----
        BorealisRawacf has the main_acfs, intf_acfs, and xcfs fields flattened
        in the site structured files.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            for field in ['main_acfs', 'intf_acfs', 'xcfs']:
                new_records[key][field] = new_records[key][field].flatten()

        return new_records

    @classmethod
    def site_get_max_dims(cls, filename: str, unshared_parameters: List[str]):
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        filename: str
            Name of the site file being checked
        unshared_parameters: List[str]
            List of parameter names that are not shared between all the records
            in the site restructured file, i.e. may have different dimensions
            between records.
            
        Returns
        -------
        fields_max_dims: dict
            dictionary containing field names (str) as keys with maximum
            dimensions required to restructure to array file as values (tuples)
        """
        fields_max_dims, max_num_sequences, max_num_beams = super().site_get_max_dims(filename, unshared_parameters)

        # Now change the main_acfs, int_acfs and xcfs dicts to maximum required dims

        # Get the num_ranges and num_lags fields directly from one record of the file
        with h5py.File(filename, 'r') as site_file:
            # hacky way to get first key with KeyView object from .keys()
            record_name = [k for i, k in enumerate(site_file.keys()) if i == 0][0]
            _, num_ranges, num_lags = site_file[record_name]['correlation_dimensions']

        # Change the data dimensions to the multidimensional size instead of flattened size
        reshaped_correlation_dims = (max_num_beams, num_ranges, num_lags)
        fields_max_dims['main_acfs'] = reshaped_correlation_dims
        fields_max_dims['intf_acfs'] = reshaped_correlation_dims
        fields_max_dims['xcfs'] = reshaped_correlation_dims

        return fields_max_dims, max_num_sequences, max_num_beams

    @classmethod
    def is_restructureable(cls) -> bool:
        """
        See BaseFormat class for description and use of this method.
        """
        return True

    @classmethod
    def single_element_types(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        single_element_types
            All the single-element fields in records of the
            format, as a dictionary fieldname : type.
        """
        return cls.fields.single_element_types('rawacf')

    @classmethod
    def array_dtypes(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        array_dtypes
            All the array fields in records of the
            format, as a dictionary fieldname : array dtype.
        """
        return cls.fields.array_types('rawacf')

    @classmethod
    def shared_fields(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        The dimension info for shared_fields is not necessary because the
        dimensions will be the same for site and restructured files.
        """
        return cls.fields.shared_fields('rawacf')

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {  # functions take records dictionary
            'num_sequences': [],
            'int_time': [],
            'sqn_timestamps': [cls.find_max_sequences],
            'noise_at_freq': [cls.find_max_sequences],
            'main_acfs': [cls.find_max_beams, cls.find_num_ranges,
                          cls.find_num_lags],
            'intf_acfs': [cls.find_max_beams, cls.find_num_ranges,
                          cls.find_num_lags],
            'xcfs': [cls.find_max_beams, cls.find_num_ranges,
                     cls.find_num_lags],
            'scan_start_marker': [],
            'beam_nums': [cls.find_max_beams],
            'beam_azms': [cls.find_max_beams],
            'num_slices': []
        }

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {  # functions take arrays dictionary and record_num
            'num_sequences': [],
            'int_time': [],
            'sqn_timestamps': [lambda arrays, record_num:
                               arrays['num_sequences'][record_num]],
            'noise_at_freq': [lambda arrays, record_num:
                              arrays['num_sequences'][record_num]],
            'main_acfs': [lambda arrays, record_num:
                          arrays['num_beams'][record_num],
                          lambda arrays, record_num:
                          arrays['main_acfs'].shape[2],
                          lambda arrays, record_num:
                          arrays['main_acfs'].shape[3]],
            'intf_acfs': [lambda arrays, record_num:
                          arrays['num_beams'][record_num],
                          lambda arrays, record_num:
                          arrays['main_acfs'].shape[2],
                          lambda arrays, record_num:
                          arrays['main_acfs'].shape[3]],
            'xcfs': [lambda arrays, record_num:
                     arrays['num_beams'][record_num],
                     lambda arrays, record_num:
                     arrays['main_acfs'].shape[2],
                     lambda arrays, record_num:
                     arrays['main_acfs'].shape[3]],
            'scan_start_marker': [],
            'beam_nums': [lambda arrays, record_num:
                          arrays['num_beams'][record_num]],
            'beam_azms': [lambda arrays, record_num:
                          arrays['num_beams'][record_num]],
            'num_slices': []
            }

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_beams': lambda records: np.array(
                [len(record['beam_nums']) for key, record in records.items()],
                dtype=np.uint32),
            'correlation_descriptors': lambda records: np.array(
                ['num_records', 'max_num_beams', 'num_ranges', 'num_lags'])
            }

    @classmethod
    def array_specific_fields_iterative_generator(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_beams': lambda record: len(record['beam_nums'])
        }

    @classmethod
    def site_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'correlation_descriptors': lambda arrays, record_num: np.array(
                ['num_beams', 'num_ranges', 'num_lags']),
            'correlation_dimensions': lambda arrays, record_num: np.array(
                [arrays['num_beams'][record_num], arrays['main_acfs'].shape[2],
                 arrays['main_acfs'].shape[3]], dtype=np.uint32)
            }


class BorealisBfiqv0_4(BaseFormat):
    """
    Class containing Borealis Bfiq data fields and their types.

    See Also
    --------
    BaseFormat
    BorealisBfiq (most up to date format)

    Notes
    -----
    Bfiq data is beamformed i and q data. It has been mixed, filtered,
    decimated to the final output receive rate, and it has been beamformed
    and all channels have been combined into their arrays. No correlation
    or averaging has occurred.

    See BaseFormat for description of classmethods and some staticmethods and
    how they are used to verify format files and restructure Borealis files to
    array and site structure.

    Static Methods
    --------------
    find_num_antenna_arrays(OrderedDict): int
        Returns number of arrays in the data for use in finding dimensions
    find_num_samps(OrderedDict): int
        Returns the number of samples in the data for use in finding dimensions
    """
    fields = BorealisFieldsv0_4

    @staticmethod
    def find_num_antenna_arrays(records: OrderedDict) -> int:
        """
        Find the number of antenna arrays given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_arrays
            The number of arrays that have been beamformed and combined in
            the file. Typically 2; main and one interferometer.

        Notes
        -----
        Num_arrays is unique to a slice so cannot change inside file.
        """
        first_key = list(records.keys())[0]
        num_arrays = records[first_key]['data_dimensions'][0]
        return num_arrays

    @staticmethod
    def find_num_samps(records: OrderedDict) -> int:
        """
        Find the number of samples given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_samps
            The number of samples that have been recorded in a sequence.

        Notes
        -----
        The num_ranges/first_range and sampling rates that determine this
        value cannot change within a slice, therefore it is one value per file.
        """
        first_key = list(records.keys())[0]
        num_samps = records[first_key]['data_dimensions'][3]
        return num_samps

    @staticmethod
    def reshape_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the data
            field in all records reshaped to the correct dimensions.

        Notes
        -----
        BorealisBfiq has the data field flattened in the
        site structured files, so this field is reshaped here to the
        correct dimensions given in data_dimensions.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            record_dimensions = records[key]['data_dimensions']
            for field in ['data']:
                new_records[key][field] = new_records[key][field].\
                        reshape(record_dimensions)

        return new_records

    @staticmethod
    def flatten_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the data
            field in all records flattened as is the convention
            in site structured files.

        Notes
        -----
        BorealisBfiq has the data field flattened in the
        site structured files.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            for field in ['data']:
                new_records[key][field] = new_records[key][field].flatten()

        return new_records

    @classmethod
    def site_get_max_dims(cls, filename: str, unshared_parameters: List[str]):
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        filename: str
            Name of the site file being checked
        unshared_parameters: List[str]
            List of parameter names that are not shared between all the records
            in the site restructured file, i.e. may have different dimensions
            between records.

        Returns
        -------
        fields_max_dims: dict
            dictionary containing field names (str) as keys with maximum
            dimensions required to restructure to array file as values (tuples)
        """
        fields_max_dims, max_num_sequences, max_num_beams = super().site_get_max_dims(filename, unshared_parameters)

        # Get the num_ant_arrays and num_samps fields directly from one record of the file
        with h5py.File(filename, 'r') as site_file:
            # hacky way to get first key with KeyView object from .keys()
            record_name = [k for i, k in enumerate(site_file.keys()) if i == 0][0]
            num_ant_arrays, _, _, num_samps = site_file[record_name]['data_dimensions']

        # Change the data dimensions to the multidimensional size instead of flattened size
        reshaped_data_dims = (num_ant_arrays, max_num_sequences, max_num_beams, num_samps)
        fields_max_dims['data'] = reshaped_data_dims

        return fields_max_dims, max_num_sequences, max_num_beams

    @classmethod
    def is_restructureable(cls) -> bool:
        """
        See BaseFormat class for description and use of this method.
        """
        return True

    @classmethod
    def single_element_types(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        single_element_types
            All the single-element fields in records of the
            format, as a dictionary fieldname : type.
        """
        return cls.fields.single_element_types('bfiq')

    @classmethod
    def array_dtypes(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        array_dtypes
            All the array fields in records of the
            format, as a dictionary fieldname : array dtype.
        """
        return cls.fields.array_types('bfiq')

    @classmethod
    def shared_fields(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        The dimension info for shared_fields is not necessary because the
        dimensions will be the same for site and restructured files.
        """
        return cls.fields.shared_fields('bfiq')

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_sequences': [],
            'int_time': [],
            'sqn_timestamps': [cls.find_max_sequences],
            'noise_at_freq': [cls.find_max_sequences],
            'data': [cls.find_num_antenna_arrays,
                     cls.find_max_sequences, cls.find_max_beams,
                     cls.find_num_samps],
            'scan_start_marker': [],
            'beam_nums': [cls.find_max_beams],
            'beam_azms': [cls.find_max_beams],
            'num_slices': []
            }

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_sequences': [],
            'int_time': [],
            'sqn_timestamps': [lambda arrays, record_num:
                               arrays['num_sequences'][record_num]],
            'noise_at_freq': [lambda arrays, record_num:
                              arrays['num_sequences'][record_num]],
            'data': [lambda arrays, record_num: arrays['data'].shape[1],
                     lambda arrays, record_num:
                     arrays['num_sequences'][record_num],
                     lambda arrays, record_num:
                     arrays['num_beams'][record_num],
                     lambda arrays, record_num: arrays['data'].shape[4]],
            'scan_start_marker': [],
            'beam_nums': [lambda arrays, record_num:
                          arrays['num_beams'][record_num]],
            'beam_azms': [lambda arrays, record_num:
                          arrays['num_beams'][record_num]],
            'num_slices': []
            }

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_beams': lambda records: np.array(
                [len(record['beam_nums']) for key, record in records.items()],
                dtype=np.uint32),
            'data_descriptors': lambda records: np.array(
                ['num_records', 'num_antenna_arrays', 'max_num_sequences',
                 'max_num_beams', 'num_samps'])
            }

    @classmethod
    def array_specific_fields_iterative_generator(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_beams': lambda record: len(record['beam_nums'])
        }

    @classmethod
    def site_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'data_descriptors': lambda arrays, record_num: np.array(
                ['num_antenna_arrays', 'num_sequences', 'num_beams',
                 'num_samps']),
            'data_dimensions': lambda arrays, record_num: np.array(
                [arrays['data'].shape[1], arrays['num_sequences'][record_num],
                 arrays['num_beams'][record_num], arrays['data'].shape[4]],
                dtype=np.uint32)
            }


class BorealisAntennasIqv0_4(BaseFormat):
    """
    Class containing Borealis Antennas iq data fields and their types.

    See Also
    --------
    BaseFormat
    BorealisAntennasIq (most up to date format)

    Notes
    -----
    Antennas iq data is data with all channels separated. It has been mixed
    and filtered, but it has not been beamformed or combined into the
    entire antenna array data product.

    See BaseFormat for description of classmethods and some staticmethods and
    how they are used to verify format files and restructure Borealis files to
    array and site structure.

    Static Methods
    --------------
    find_num_antennas(OrderedDict): int
        Returns number of antennas in the data for use in finding dimensions
    find_num_samps(OrderedDict): int
        Returns the number of samples in the data for use in finding dimensions
    """
    fields = BorealisFieldsv0_4

    @staticmethod
    def find_num_antennas(records: OrderedDict) -> int:
        """
        Find the number of antennas given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_antennas
            The number of antennas that have been recorded and stored in the
            file.

        Notes
        -----
        Num_antennas is unique to a slice so cannot change inside file.
        """
        first_key = list(records.keys())[0]
        num_antennas = records[first_key]['data_dimensions'][0]
        return num_antennas

    @staticmethod
    def find_num_samps(records: OrderedDict) -> int:
        """
        Find the number of samples given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_samps
            The number of samples that have been recorded in a sequence.

        Notes
        -----
        The num_ranges/first_range and sampling rates that determine this
        value cannot change within a slice, therefore it is one value per file.
        """
        first_key = list(records.keys())[0]
        num_samps = records[first_key]['data_dimensions'][2]
        return num_samps

    @staticmethod
    def reshape_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the data
            field in all records reshaped to the correct dimensions.

        Notes
        -----
        BorealisAntennasIq has the data field flattened in the
        site structured files, so this field is reshaped here to the correct
        data_dimensions given in the file.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            record_dimensions = records[key]['data_dimensions']
            for field in ['data']:
                new_records[key][field] = new_records[key][field].\
                        reshape(record_dimensions)

        return new_records

    @staticmethod
    def flatten_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the data
            field in all records flattened as is the convention
            in site structured files.

        Notes
        -----
        BorealisAntennasIq has the data field flattened in the
        site structured files.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            for field in ['data']:
                new_records[key][field] = new_records[key][field].flatten()

        return new_records

    @classmethod
    def site_get_max_dims(cls, filename: str, unshared_parameters: List[str]):
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        filename: str
            Name of the site file being checked
        unshared_parameters: List[str]
            List of parameter names that are not shared between all the records
            in the site restructured file, i.e. may have different dimensions
            between records.
        Returns
        -------
        fields_max_dims: dict
            dictionary containing field names (str) as keys with maximum
            dimensions required to restructure to array file as values (tuples)
        Raises
        ------

        """
        fields_max_dims, max_num_sequences, max_num_beams = super().site_get_max_dims(filename, unshared_parameters)

        # Get the num_antennas and num_samps fields directly from one record of the file
        with h5py.File(filename, 'r') as site_file:
            # hacky way to get first key with KeyView object from .keys()
            record_name = [k for i, k in enumerate(site_file.keys()) if i == 0][0]
            num_antennas, _, num_samps = site_file[record_name]['data_dimensions']

        # Change the data dimensions to the multidimensional size instead of flattened size
        reshaped_data_dims = (num_antennas, max_num_sequences, num_samps)
        fields_max_dims['data'] = reshaped_data_dims

        return fields_max_dims, max_num_sequences, max_num_beams

    @classmethod
    def is_restructureable(cls) -> bool:
        """
        See BaseFormat class for description and use of this method.
        """
        return True

    @classmethod
    def single_element_types(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        single_element_types
            All the single-element fields in records of the
            format, as a dictionary fieldname : type.
        """
        return cls.fields.single_element_types('antennas_iq')

    @classmethod
    def array_dtypes(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        array_dtypes
            All the array fields in records of the
            format, as a dictionary fieldname : array dtype.
        """
        return cls.fields.array_types('antennas_iq')

    @classmethod
    def shared_fields(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        The dimension info for shared_fields is not necessary because the
        dimensions will be the same for site and restructured files.
        """
        return cls.fields.shared_fields('antennas_iq')

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_sequences': [],
            'int_time': [],
            'sqn_timestamps': [cls.find_max_sequences],
            'noise_at_freq': [cls.find_max_sequences],
            'data': [cls.find_num_antennas, cls.find_max_sequences,
                     cls.find_num_samps],
            'scan_start_marker': [],
            'beam_nums': [cls.find_max_beams],
            'beam_azms': [cls.find_max_beams],
            'num_slices': []
            }

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_sequences': [],
            'int_time': [],
            'sqn_timestamps': [lambda arrays, record_num:
                               arrays['num_sequences'][record_num]],
            'noise_at_freq': [lambda arrays, record_num:
                              arrays['num_sequences'][record_num]],
            'data': [lambda arrays, record_num: arrays['data'].shape[1],
                     lambda arrays, record_num:
                     arrays['num_sequences'][record_num],
                     lambda arrays, record_num: arrays['data'].shape[3]],
            'scan_start_marker': [],
            'beam_nums': [lambda arrays, record_num:
                          arrays['num_beams'][record_num]],
            'beam_azms': [lambda arrays, record_num:
                          arrays['num_beams'][record_num]],
            'num_slices': []
            }

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_beams': lambda records: np.array(
                [len(record['beam_nums']) for key, record in records.items()],
                dtype=np.uint32),
            'data_descriptors': lambda records: np.array(
                ['num_records', 'num_antennas', 'max_num_sequences',
                 'num_samps'])
            }

    @classmethod
    def array_specific_fields_iterative_generator(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'num_beams': lambda record: len(record['beam_nums'])
        }

    @classmethod
    def site_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        return {
            'data_descriptors': lambda arrays, record_num: np.array(
                ['num_antennas', 'num_sequences', 'num_samps']),
            'data_dimensions': lambda arrays, record_num: np.array(
                [arrays['data'].shape[1], arrays['num_sequences'][record_num],
                 arrays['data'].shape[3]], dtype=np.uint32)
            }


class BorealisRawrfv0_4(BaseFormat):
    """
    Class containing Borealis Rawrf data fields and their types.

    See Also
    --------
    BaseFormat
    BorealisRawrf (most up to date format)

    Notes
    -----
    This filetype only has site files and cannot be restructured.

    Rawrf data is data that has been produced at the original receive bandwidth
    and has not been mixed, filtered, or decimated.

    See BaseFormat for description of classmethods and some staticmethods
    and how they are used to verify format files and restructure Borealis
    files to array and site structure.
    """
    fields = BorealisFieldsv0_4

    @staticmethod
    def reshape_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the data
            field in all records reshaped to the correct dimensions.

        Notes
        -----
        BorealisRawrf has the data field flattened in the
        site structured files, so this field is reshaped in here.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            record_dimensions = records[key]['data_dimensions']
            for field in ['data']:
                new_records[key][field] = new_records[key][field].\
                                        reshape(record_dimensions)

        return new_records

    @staticmethod
    def flatten_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the data
            field in all records flattened as is the convention
            in site structured files.

        Notes
        -----
        BorealisRawrf has the data field flattened in the
        site structured files.
        """
        new_records = copy.deepcopy(records)
        for key in list(records.keys()):
            for field in ['data']:
                new_records[key][field] = new_records[key][field].flatten()

        return new_records

    @classmethod
    def site_get_max_dims(cls, filename: str, unshared_parameters: List[str]):
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        filename: str
            Name of the site file being checked
        unshared_parameters: List[str]
            List of parameter names that are not shared between all the records
            in the site restructured file, i.e. may have different dimensions
            between records.

        Returns
        -------
        fields_max_dims: dict
            dictionary containing field names (str) as keys with maximum
            dimensions required to restructure to array file as values (tuples)
        """
        fields_max_dims, max_num_sequences, max_num_beams = super().site_get_max_dims(filename, unshared_parameters)

        # Get the num_antennas and num_samps fields directly from one record of the file
        with h5py.File(filename, 'r') as site_file:
            # hacky way to get first key with KeyView object from .keys()
            record_name = [k for i, k in enumerate(site_file.keys()) if i == 0][0]
            num_antennas, _, num_samps = site_file[record_name]['data_dimensions']

        # Change the data dimensions to the multidimensional size instead of flattened size
        reshaped_data_dims = (num_antennas, max_num_sequences, num_samps)
        fields_max_dims['data'] = reshaped_data_dims

        return fields_max_dims, max_num_sequences, max_num_beams

    @classmethod
    def is_restructureable(cls) -> bool:
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        BorealisRawrf is a very uncommon format and therefore has
        not been implemented to be converted to arrays.
        """
        return False

    @classmethod
    def single_element_types(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        single_element_types
            All the single-element fields in records of the
            format, as a dictionary fieldname : type.
        """
        return cls.fields.single_element_types('rawrf')

    @classmethod
    def array_dtypes(cls):
        """
        See BaseFormat class for description and use of this method.

        Returns
        -------
        array_dtypes
            All the array fields in records of the
            format, as a dictionary fieldname : array dtype.
        """
        return cls.fields.array_types('rawrf')


class BorealisRawacfv0_5(BorealisRawacfv0_4):
    """
    Class containing Borealis Rawacf data fields and their types

    See Also
    --------
    BaseFormat
    BorealisRawacfv0_4
    https://borealis.readthedocs.io/en/latest/borealis_data.html

    Notes
    -----
    Rawacf data has been mixed, filtered, and decimated; beamformed and
    combined into antenna arrays; then autocorrelated and correlated between
    antenna arrays to produce matrices of num_ranges x num_lags.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.5, the following fields were added:
    slice_id, slice_interfacing, scheduling_mode, and averaging_method.
    As well, blanked_samples was changed from shared to unshared in the array
    restructuring, which necessitates an array-specific field,
    num_blanked_samples, to specify how much data to read in the
    blanked_samples array in the array style file.
    """
    fields = BorealisFieldsv0_5

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        In Borealis v0.5, blanked samples was changed to an unshared field.
        This was a bug in earlier versions. 'slice_interfacing' was a new
        field added in Borealis v0.5. It is an unshared field because
        new slices may be added and interfaced to this slice and therefore
        slice_interfacing may not be the same from record to record.
        """
        unshared_fields_dims = super().unshared_fields_dims_array()
        unshared_fields_dims.update({
            'blanked_samples': [cls.
                                find_max_field_len_func('blanked_samples')],
            'slice_interfacing': []
            })
        return unshared_fields_dims

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        unshared_fields_dims = super().unshared_fields_dims_site()
        unshared_fields_dims.update({
            'blanked_samples': [lambda arrays, record_num:
                                arrays['num_blanked_samples'][record_num]],
            'slice_interfacing': [lambda arrays, record_num:
                                  len(arrays['slice_interfacing'][record_num])]
            })
        return unshared_fields_dims

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        array_specific = super().array_specific_fields_generate()
        array_specific.update({
            'num_blanked_samples': lambda records: np.array(
                [len(record['blanked_samples']) for key, record in
                 records.items()], dtype=np.uint32)
            })
        return array_specific

    @classmethod
    def array_specific_fields_iterative_generator(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        array_specific = super().array_specific_fields_iterative_generator()
        array_specific.update({
            'num_blanked_samples': lambda record: len(record['blanked_samples'])
            })
        return array_specific


class BorealisBfiqv0_5(BorealisBfiqv0_4):
    """
    Class containing Borealis Bfiq data fields and their types

    See Also
    --------
    BaseFormat
    BorealisBfiqv0_4

    Notes
    -----
    Bfiq data is beamformed i and q data. It has been mixed, filtered,
    decimated to the final output receive rate, and it has been beamformed
    and all channels have been combined into their arrays. No correlation
    or averaging has occurred.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.5, the following fields were added:
    slice_id, slice_interfacing, and scheduling_mode.
    As well, blanked_samples was changed from shared to unshared in the array
    restructuring, which necessitates an array-specific field,
    num_blanked_samples, to specify how much data to read in the
    blanked_samples array in the array style file.
    """
    fields = BorealisFieldsv0_5

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        In Borealis v0.5, blanked samples was changed to an unshared field.
        This was a bug in earlier versions. 'slice_interfacing' was a new
        field added in Borealis v0.5. It is an unshared field because
        new slices may be added and interfaced to this slice and therefore
        slice_interfacing may not be the same from record to record.
        """
        unshared_fields_dims = super().unshared_fields_dims_array()
        unshared_fields_dims.update({
            'blanked_samples': [cls.
                                find_max_field_len_func('blanked_samples')],
            'slice_interfacing': []
            })
        return unshared_fields_dims

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        unshared_fields_dims = super().unshared_fields_dims_site()
        unshared_fields_dims.update({
            'blanked_samples': [lambda arrays, record_num:
                                arrays['num_blanked_samples'][record_num]],
            'slice_interfacing': [lambda arrays, record_num:
                                  len(arrays['slice_interfacing'][record_num])]
            })
        return unshared_fields_dims

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        array_specific = super().array_specific_fields_generate()
        array_specific.update({
            'num_blanked_samples': lambda records: np.array(
                [len(record['blanked_samples']) for key, record in
                 records.items()], dtype=np.uint32)
            })
        return array_specific

    @classmethod
    def array_specific_fields_iterative_generator(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        array_specific = super().array_specific_fields_iterative_generator()
        array_specific.update({
            'num_blanked_samples': lambda record: len(record['blanked_samples'])
        })
        return array_specific


class BorealisAntennasIqv0_5(BorealisAntennasIqv0_4):
    """
    Class containing Borealis Antennas iq data fields and their types

    See Also
    --------
    BaseFormat
    BorealisAntennasIqv0_4

    Notes
    -----
    Antennas iq data is data with all channels separated. It has been mixed
    and filtered, but it has not been beamformed or combined into the
    entire antenna array data product.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.5, the following fields were added to the Borealis-produced
    site structured files:
    slice_id, slice_interfacing, scheduling_mode, and blanked_samples.
    blanked_samples is unshared in the array restructuring, which necessitates
    an array-specific field, num_blanked_samples, to specify how much data to
    read in the blanked_samples array in the array style file.
    """
    fields = BorealisFieldsv0_5

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        In Borealis v0.5, blanked samples was added to the antennas_iq.
        This was a bug in earlier versions. 'slice_interfacing' was a new
        field as well. Both are unshared fields because
        new slices may be added and interfaced to this slice and therefore
        the field may not be the same from record to record.
        """
        unshared_fields_dims = super().unshared_fields_dims_array()
        unshared_fields_dims.update({
            'blanked_samples': [cls.
                                find_max_field_len_func('blanked_samples')],
            'slice_interfacing': []
            })
        return unshared_fields_dims

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        unshared_fields_dims = super().unshared_fields_dims_site()
        unshared_fields_dims.update({
            'blanked_samples': [lambda arrays, record_num:
                                arrays['num_blanked_samples'][record_num]],
            'slice_interfacing': []
            })
        return unshared_fields_dims

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        array_specific = super().array_specific_fields_generate()
        array_specific.update({
            'num_blanked_samples': lambda records: np.array(
                [len(record['blanked_samples']) for key, record in
                 records.items()], dtype=np.uint32)
            })
        return array_specific

    @classmethod
    def array_specific_fields_iterative_generator(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        array_specific = super().array_specific_fields_iterative_generator()
        array_specific.update({
            'num_blanked_samples': lambda record: len(record['blanked_samples'])
        })
        return array_specific


class BorealisRawrfv0_5(BorealisRawrfv0_4):
    """
    Class containing Borealis Rawrf data fields and their types

    See Also
    --------
    BaseFormat
    BorealisRawrfv0_4

    Notes
    -----
    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.5, the following fields were added to BorealisRawrf:
    scheduling_mode and blanked_samples.
    """
    fields = BorealisFieldsv0_5


class BorealisRawacfv0_6(BorealisRawacfv0_5):
    """
    Class containing Borealis Rawacf data fields and their types for Borealis 
    version 0.6.

    See Also
    --------
    BaseFormat
    BorealisRawacfv0_5
    https://borealis.readthedocs.io/en/latest/borealis_data.html

    Notes
    -----
    Rawacf data has been mixed, filtered, and decimated; beamformed and
    combined into antenna arrays; then autocorrelated and correlated between
    antenna arrays to produce matrices of num_ranges x num_lags.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.6, four fields were added to site files:
    gps_locked, gps_to_system_time_diff, agc_status_word, and
    lp_status_word. Array structured files contain the same fields,
    but with dims of [num_records].
    """
    fields = BorealisFieldsv0_6

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        In Borealis v0.6, gps_locked, gps_to_system_time_diff, agc_status_word,
        and lp_status_word were added to rawacf.
        All are unshared fields because their values may not be the same from
        record to record.
        """
        unshared_fields_dims = super().unshared_fields_dims_array()
        unshared_fields_dims.update({
            'agc_status_word': [],
            'lp_status_word': [],
            'gps_locked': [],
            'gps_to_system_time_diff': [],
            })
        return unshared_fields_dims

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        unshared_fields_dims = super().unshared_fields_dims_site()
        unshared_fields_dims.update({
            'agc_status_word': [],
            'lp_status_word': [],
            'gps_locked': [],
            'gps_to_system_time_diff': [],
            })
        return unshared_fields_dims


class BorealisBfiqv0_6(BorealisBfiqv0_5):
    """
    Class containing Borealis Bfiq data fields and their types for Borealis
    version 0.6.

    See Also
    --------
    BaseFormat
    BorealisBfiqv0_5

    Notes
    -----
    Bfiq data is beamformed i and q data. It has been mixed, filtered,
    decimated to the final output receive rate, and it has been beamformed
    and all channels have been combined into their arrays. No correlation
    or averaging has occurred.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.6, four fields were added to site files:
    gps_locked, gps_to_system_time_diff, agc_status_word, and
    lp_status_word. Array structured files contain the same fields,
    but with dims of [num_records].

    pulse_phase_offset was also added
    """
    fields = BorealisFieldsv0_6

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        In Borealis v0.6, gps_locked, gps_to_system_time_diff, agc_status_word,
        lp_status_word and pulse_phase_offset were added to bfiq.
        All are unshared fields because their values may not be the same from
        record to record.
        """
        unshared_fields_dims = super().unshared_fields_dims_array()
        unshared_fields_dims.update({
            'agc_status_word': [],
            'lp_status_word': [],
            'gps_locked': [],
            'gps_to_system_time_diff': [],
            'pulse_phase_offset': [cls.find_max_pulse_phase_offset]
            })
        return unshared_fields_dims

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        unshared_fields_dims = super().unshared_fields_dims_site()
        unshared_fields_dims.update({
            'agc_status_word': [],
            'lp_status_word': [],
            'gps_locked': [],
            'gps_to_system_time_diff': [],
            'pulse_phase_offset': [lambda arrays, record_num:
                                   -1 if arrays['pulse_phase_offset'].size < arrays['num_sequences'][record_num]
                                   else
                                   list((arrays['num_sequences'][record_num],)
                                        + arrays['pulse_phase_offset'][
                                              record_num].shape[1:])],
            })
        return unshared_fields_dims


class BorealisAntennasIqv0_6(BorealisAntennasIqv0_5):
    """
    Class containing Borealis Antennas iq data fields and their types for
    Borealis version 0.6.

    See Also
    --------
    BaseFormat
    BorealisAntennasIqv0_5

    Notes
    -----
    Antennas iq data is data with all channels separated. It has been mixed
    and filtered, but it has not been beamformed or combined into the
    entire antenna array data product.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.6, the following fields were added to the Borealis-produced
    site structured files:
    gps_locked, gps_to_system_time_diff, agc_status_word, and lp_status_word.
    Array structured files contain the same fields,
    but with dims of [num_records].

    pulse_phase_offset was also added to the site-structured files.
    """
    fields = BorealisFieldsv0_6

    @classmethod
    def unshared_fields_dims_array(cls):
        """
        See BaseFormat class for description and use of this method.

        Notes
        -----
        In Borealis v0.6, gps_locked, gps_to_system_time_diff, agc_status_word,
        lp_status_word and pulse_phase_offset were added to the antennas_iq.
        All are unshared fields because their values may not be the same from
        record to record.
        """
        unshared_fields_dims = super().unshared_fields_dims_array()
        unshared_fields_dims.update({
            'agc_status_word': [],
            'lp_status_word': [],
            'gps_locked': [],
            'gps_to_system_time_diff': [],
            'pulse_phase_offset': [cls.find_max_pulse_phase_offset]
            })
        return unshared_fields_dims

    @classmethod
    def unshared_fields_dims_site(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        unshared_fields_dims = super().unshared_fields_dims_site()
        unshared_fields_dims.update({
            'agc_status_word': [],
            'lp_status_word': [],
            'gps_locked': [],
            'gps_to_system_time_diff': [],
            'pulse_phase_offset': [lambda arrays, record_num:
                                   -1 if arrays['pulse_phase_offset'].size < arrays['num_sequences'][record_num]
                                   else
                                   list((arrays['num_sequences'][record_num],)
                                        + arrays['pulse_phase_offset'][
                                              record_num].shape[1:])],
            })
        return unshared_fields_dims


class BorealisRawrfv0_6(BorealisRawrfv0_5):
    """
    Class containing Borealis Rawrf data fields and their types for Borealis 
    version 0.6.

    See Also
    --------
    BaseFormat
    BorealisRawrfv0_5

    Notes
    -----
    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.6, the following fields were added to BorealisRawrf:
    gps_locked, gps_to_system_time_diff, agc_status_word, and
    lp_status_word.
    """
    fields = BorealisFieldsv0_6

# The following are the currently used classes, with additions according
# to Borealis updates.

class BorealisRawacf(BorealisRawacfv0_6):
    """
    Class containing Borealis Rawacf data fields and their types for the
    current version of Borealis (v0.7).

    See Also
    --------
    BaseFormat
    BorealisRawacfv0_6
    https://borealis.readthedocs.io/en/latest/borealis_data.html

    Notes
    -----
    Rawacf data has been mixed, filtered, and decimated; beamformed and
    combined into antenna arrays; then autocorrelated and correlated between
    antenna arrays to produce matrices of num_ranges x num_lags.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.7, the fields correlation_descriptors and correlation-dimensions
    were replaced by data_descriptors and data_dimensions, respectively.
    """
    fields = BorealisFields

    @staticmethod
    def find_num_ranges(records: OrderedDict) -> int:
        """
        Find the number of ranges given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_ranges
            The number of ranges being calculated in the acfs.

        Notes
        -----
        Num_ranges is unique to a slice so cannot change inside file.
        """
        first_key = list(records.keys())[0]
        num_ranges = records[first_key]['data_dimensions'][1]
        return num_ranges

    @staticmethod
    def find_num_lags(records: OrderedDict) -> int:
        """
        Find the number of lags given the records dictionary, for
        restructuring to arrays.

        Parameters
        ----------
        records
            The records dictionary from a site-style file.

        Returns
        -------
        num_lags
            The number of lags being calculated in the acfs.

        Notes
        -----
        Num_lags is unique to a slice so cannot change inside file.
        """
        first_key = list(records.keys())[0]
        num_lags = records[first_key]['data_dimensions'][2]
        return num_lags

    @classmethod
    def site_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        fields_generate = super().site_specific_fields_generate()
        data_dimensions = fields_generate.pop('correlation_dimensions')
        fields_generate['data_dimensions'] = data_dimensions
        fields_generate.pop('correlation_descriptors')
        fields_generate['data_descriptors'] = lambda arrays, record_num: np.bytes_(
            ['num_beams', 'num_ranges', 'num_lags'])

        return fields_generate

    @classmethod
    def array_specific_fields_generate(cls):
        """
        See BaseFormat class for description and use of this method.
        """
        fields_generate = super().array_specific_fields_generate()
        fields_generate.pop('correlation_descriptors')
        fields_generate['data_descriptors'] = lambda records: np.bytes_(
            ['num_records', 'max_num_beams', 'num_ranges', 'num_lags'])
        return fields_generate

    @staticmethod
    def reshape_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data, with the main_acfs,
            intf_acfs, and xcfs fields in all records reshaped to the correct
            dimensions.

        Notes
        -----
        BorealisRawacf has the correlation fields non-flattened, so nothing needs to be done.
        """
        # dimensions provided in data_dimensions field as num_beams,
        # num_ranges, num_lags for the rawacf format.
        new_records = copy.deepcopy(records)
        return new_records

    @staticmethod
    def flatten_site_arrays(records: OrderedDict) -> OrderedDict:
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        records
            An OrderedDict of the site style data, organized
            by record. Records are stored with timestamps
            as the keys and the data for that timestamp
            stored as a dictionary.

        Returns
        -------
        records
            An OrderedDict of the site style data.

        Notes
        -----
        BorealisRawacf has the main_acfs, intf_acfs, and xcfs fields non-flattened
        in the site structured files, so nothing needs to be done.
        """
        new_records = copy.deepcopy(records)
        return new_records

    @classmethod
    def site_get_max_dims(cls, filename: str, unshared_parameters: List[str]):
        """
        See BaseFormat class for description and use of this method.

        Parameters
        ----------
        filename: str
            Name of the site file being checked
        unshared_parameters: List[str]
            List of parameter names that are not shared between all the records
            in the site restructured file, i.e. may have different dimensions
            between records.

        Returns
        -------
        fields_max_dims: dict
            dictionary containing field names (str) as keys with maximum
            dimensions required to restructure to array file as values (tuples)
        """
        fields_max_dims, max_num_sequences, max_num_beams = BaseFormat.site_get_max_dims(filename, unshared_parameters)

        # Now change the main_acfs, int_acfs and xcfs dicts to maximum required dims

        # Get the num_ranges and num_lags fields directly from one record of the file
        with h5py.File(filename, 'r') as site_file:
            # hacky way to get first key with KeyView object from .keys()
            record_name = [k for i, k in enumerate(site_file.keys()) if i == 0][0]
            _, num_ranges, num_lags = site_file[record_name]['data_dimensions']

        # Change the data dimensions to the multidimensional size instead of flattened size
        reshaped_correlation_dims = (max_num_beams, num_ranges, num_lags)
        fields_max_dims['main_acfs'] = reshaped_correlation_dims
        fields_max_dims['intf_acfs'] = reshaped_correlation_dims
        fields_max_dims['xcfs'] = reshaped_correlation_dims

        return fields_max_dims, max_num_sequences, max_num_beams


class BorealisBfiq(BorealisBfiqv0_6):
    """
    Class containing Borealis Bfiq data fields and their types for the
    current version of Borealis (v0.7).

    See Also
    --------
    BaseFormat
    BorealisBfiqv0_6

    Notes
    -----
    Bfiq data is beamformed i and q data. It has been mixed, filtered,
    decimated to the final output receive rate, and it has been beamformed
    and all channels have been combined into their arrays. No correlation
    or averaging has occurred.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    There were no changes to the bfiq file format in v0.7.
    """
    fields = BorealisFields


class BorealisAntennasIq(BorealisAntennasIqv0_6):
    """
    Class containing Borealis Antennas iq data fields and their types for
    Borealis current version (v0.7).

    See Also
    --------
    BaseFormat
    BorealisAntennasIqv0_6

    Notes
    -----
    Antennas iq data is data with all channels separated. It has been mixed
    and filtered, but it has not been beamformed or combined into the
    entire antenna array data product.

    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    In v0.7, the following fields were added to the Borealis-produced
    site structured files for ease of postprocessing:
    first_range
    first_range_rtt
    lags
    num_ranges
    range_sep
    """
    fields = BorealisFields


class BorealisRawrf(BorealisRawrfv0_6):
    """
    Class containing Borealis Rawrf data fields and their types for current
    Borealis version (v0.7).

    See Also
    --------
    BaseFormat
    BorealisRawrfv0_6

    Notes
    -----
    See BaseFormat for description of classmethods and how they
    are used to verify format files and restructure Borealis files to
    array and site structure.

    There were no changes to the rawrf file format in v0.7.
    """
    fields = BorealisFields

# borealis versions
borealis_version_dict = {
    'v0.2': {
        'bfiq': BorealisBfiqv0_4,
        'rawacf': BorealisRawacfv0_4,
        'antennas_iq': BorealisAntennasIqv0_4,
        'rawrf': BorealisRawrfv0_4
        },
    'v0.3': {
        'bfiq': BorealisBfiqv0_4,
        'rawacf': BorealisRawacfv0_4,
        'antennas_iq': BorealisAntennasIqv0_4,
        'rawrf': BorealisRawrfv0_4
        },
    'v0.4': {
        'bfiq': BorealisBfiqv0_4,
        'rawacf': BorealisRawacfv0_4,
        'antennas_iq': BorealisAntennasIqv0_4,
        'rawrf': BorealisRawrfv0_4
        },
    'v0.5': {
        'bfiq': BorealisBfiqv0_5,
        'rawacf': BorealisRawacfv0_5,
        'antennas_iq': BorealisAntennasIqv0_5,
        'rawrf': BorealisRawrfv0_5
        },
    'v0.6': {
        'bfiq': BorealisBfiqv0_6,
        'rawacf': BorealisRawacfv0_6,
        'antennas_iq': BorealisAntennasIqv0_6,
        'rawrf': BorealisRawrfv0_6
        },
    'v0.7': {
        'bfiq': BorealisBfiq,
        'rawacf': BorealisRawacf,
        'antennas_iq': BorealisAntennasIq,
        'rawrf': BorealisRawrf
    }
}
