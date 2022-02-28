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

        # TODO: Call to some restructure method here.

    def __repr__(self):
        """ for representation of the class object"""

        return "{class_name}({records}{borealis_filetype}{sdarn_filename})"\
               "".format(class_name=self.__class__.__name__,
                         records=self.records,
                         borealis_filetype=self.borealis_filetype,
                         sdarn_filename=self.sdarn_filename)

    def __str__(self):
        """ for printing of the class object"""

        return "Converting {total_records} {borealis_filetype} records into "\
               "DMap SDARN records and writing to file {sdarn_filename}."\
               "".format(total_records=len(self.borealis_records.keys()),
                         borealis_filetype=self.borealis_filetype,
                         sdarn_filename=self.sdarn_filename)

    @property
    def sdarn_dmap_records(self):
        """
        The converted SDARN DMap records to write to file.
        """
        return self._sdarn_dmap_records

    @property
    def sdarn_dict(self):
        """
        The converted SDARN records as a dictionary, before being converted
        to DMap.
        """
        return self._sdarn_dict

    @property
    def sdarn_filetype(self):
        """
        The dmap filetype converted to. 'rawacf' and 'iqdat' are allowed.
        """
        return self._sdarn_filetype

    @property
    def borealis_slice_id(self):
        """
        The slice id of the file being converted. Used as channel identifier
        in SDARN DMap records.
        """
        return self._borealis_slice_id

    @property
    def scaling_factor(self):
        """
        The scaling factor the data has been multiplied by before converting
        to integers for the corresponding dmap format.
        """
        return self._scaling_factor

    def _write_to_sdarn(self) -> str:
        """
        Write the Borealis records as SDARN DMap records to a file using
        pyDARNio.

        Returns
        -------
        sdarn_filename, the name of the SDARN file written.
        """

        self._convert_records_to_dmap()
        sdarn_writer = SDarnWrite(self._sdarn_dmap_records,
                                  self.sdarn_filename)
        if self.sdarn_filetype == 'iqdat':
            sdarn_writer.write_iqdat(self.sdarn_filename)
        elif self.sdarn_filetype == 'rawacf':
            sdarn_writer.write_rawacf(self.sdarn_filename)
        return self.sdarn_filename

    def _convert_records_to_dmap(self):
        """
        Convert the Borealis records to the a DMap filetype according to
        the origin filetype.

        Raises
        ------
        BorealisConversionTypesError
        """
        if self.sdarn_filetype == 'iqdat':
            if self._is_convertible_to_iqdat():
                self._convert_bfiq_to_iqdat()
        elif self.sdarn_filetype == 'rawacf':
            if self._is_convertible_to_rawacf():
                self._convert_rawacf_to_rawacf()
        else:  # nothing else is currently supported
            raise borealis_exceptions.BorealisConversionTypesError(
                self.sdarn_filename, self.borealis_filetype,
                self.__allowed_conversions)

    def _is_convertible_to_iqdat(self) -> bool:
        """
        Checks if the file is convertible to iqdat.

        The file is convertible if:
            - the origin filetype is bfiq
            - the blanked_samples array = pulses array for all records
            - the pulse_phase_offset array contains all zeroes for all records

        Raises
        ------
        BorealisConversionTypesError
        BorealisConvert2IqdatError

        Returns
        -------
        True if convertible to the IQDAT format
        """
        if self.borealis_filetype != 'bfiq':
            raise borealis_exceptions.BorealisConversionTypesError(
                self.sdarn_filename, self.borealis_filetype,
                self.__allowed_conversions)
        else:  # There are some specific things to check
            for record_key, record in self.borealis_records.items():
                sample_spacing = int(record['tau_spacing'] /
                                     record['tx_pulse_len'])
                normal_blanked_1 = record['pulses'] * sample_spacing
                normal_blanked_2 = normal_blanked_1 + 1
                blanked = np.concatenate((normal_blanked_1, normal_blanked_2))
                blanked = np.sort(blanked)
                if not np.array_equal(record['blanked_samples'], blanked):
                    raise borealis_exceptions.\
                            BorealisConvert2IqdatError(
                                'Increased complexity: Borealis bfiq file'
                                ' record {} blanked_samples {} does not equate'
                                ' to pulses array converted to sample number '
                                '{} * {}'.format(record_key,
                                                 record['blanked_samples'],
                                                 record['pulses'],
                                                 int(record['tau_spacing'] /
                                                     record['tx_pulse_len'])))
                if not all([x == 0 for x in record['pulse_phase_offset']]):
                    raise borealis_exceptions.\
                            BorealisConvert2IqdatError(
                                'Increased complexity: Borealis bfiq file '
                                'record {} pulse_phase_offset {} contains '
                                'non-zero values.'.format(
                                    record_key, record['pulse_phase_offset']))
        return True

    def _is_convertible_to_rawacf(self) -> bool:
        """
        Checks if the file is convertible to rawacf.

        The file is convertible if:
            - the origin filetype is rawacf
            - the blanked_samples array = pulses array for all records
            - the pulse_phase_offset array contains all zeroes for all records

        TODO: should this fail for multiple beams in the same
        integration time. IE, is it ok for dmap files to have multiple
        records with same origin time and timestamps due to a different
        beam azimuth.

        Raises
        ------
        BorealisConversionTypesError
        BorealisConvert2RawacfError

        Returns
        -------
        True if convertible to the RAWACF format
        """
        if self.borealis_filetype != 'rawacf':
            raise borealis_exceptions.\
                    BorealisConversionTypesError(self.sdarn_filename,
                                                 self.borealis_filetype,
                                                 self.__allowed_conversions)
        else:  # There are some specific things to check
            for record_key, record in self.borealis_records.items():
                sample_spacing = int(record['tau_spacing'] /
                                     record['tx_pulse_len'])
                normal_blanked_1 = record['pulses'] * sample_spacing
                normal_blanked_2 = normal_blanked_1 + 1
                blanked = np.concatenate((normal_blanked_1, normal_blanked_2))
                blanked = np.sort(blanked)
                if not np.array_equal(record['blanked_samples'], blanked):
                    raise borealis_exceptions.\
                            BorealisConvert2RawacfError(
                                'Increased complexity: Borealis rawacf file'
                                ' record {} blanked_samples {} does not equate'
                                ' to pulses array converted to sample number '
                                '{} * {}'.format(record_key,
                                                 record['blanked_samples'],
                                                 record['pulses'],
                                                 int(record['tau_spacing'] /
                                                     record['tx_pulse_len'])))

        return True

    def _convert_bfiq_to_iqdat(self):
        """
        Conversion for bfiq to iqdat SDARN DMap records.

        See Also
        --------
        __convert_bfiq_record
        https://superdarn.github.io/rst/superdarn/src.doc/rfc/0027.html
        https://borealis.readthedocs.io/en/master/
        BorealisBfiq
        Iqdat

        Raises
        ------
        BorealisConvert2IqdatError

        Notes
        -----
        SuperDARN RFC 0027 specifies that the dimensions of the data in
        iqdat should be by number of sequences, number of arrays, number
        of samples, 2 (i+q). There is some history where the dimensions were
        instead sequences, samples, arrays, 2(i+q). We have chosen to
        use the former, as it is consistent with the rest of SuperDARN Canada
        radars at this time and is as specified in the document. This means
        that you may need to use make_raw with the -d option in RST if you
        wish to process the resulting iqdat into rawacf.

        Returns
        -------
        dmap_recs, the records converted to DMap format
        """
        try:
            recs = []
            for record in self.borealis_records.items():
                sdarn_record_dict = \
                        self.__convert_bfiq_record(self.borealis_slice_id,
                                                   record,
                                                   self.borealis_filename,
                                                   self.scaling_factor)
                recs.append(sdarn_record_dict)
            self._sdarn_dict = recs
            self._sdarn_dmap_records = dict2dmap(recs)
        except Exception as e:
            raise borealis_exceptions.BorealisConvert2IqdatError(e) from e

    @staticmethod
    def __convert_bfiq_record(borealis_slice_id: int,
                              borealis_bfiq_record: tuple,
                              origin_string: str,
                              scaling_factor: int = 1) -> dict:
        """
        Converts a single record dict of Borealis bfiq data to a SDARN DMap
        record dict.

        Parameters
        ----------
        borealis_slice_id : int
            slice id integer of the borealis data, for conversion. Used
            as SDARN DMap channel identifier.
        borealis_bfiq_record : tuple(str, dict)
            Key is bfiq record timestamp, value is dictionary of bfiq
            record data.
        origin_string : str
            String representing origin of the Borealis data, typically
            Borealis filename.
        scaling_factor : int
            A scaling factor to adjust the integer values by, as the precision
            of bfiq floating points are much greater than the int16 can
            accommodate. This value is provided to multiply the data
            by before converting to int, to allow the noise floor to be
            seen, for instance.

        Notes
        -----
        The scaling_factor can cause the data to scale outside the limits of
        int16, at which point the data will be equal to the int16 max or min.
        """

        # key value pair from Borealis bfiq record.
        (record_key, record_dict) = borealis_bfiq_record

        # data_descriptors (dimensions) are num_antenna_arrays,
        # num_sequences, num_beams, num_samps
        # scale by normalization and then scale to integer max as per
        # dmap style
        data = record_dict['data'].reshape(record_dict['data_dimensions']).\
            astype(np.complex64) / record_dict['data_normalization_factor'] *\
            np.iinfo(np.int16).max * scaling_factor

        # Borealis git tag version numbers. If not a tagged version,
        # then use 255.255
        if record_dict['borealis_git_hash'][0] == 'v' and \
                record_dict['borealis_git_hash'][2] == '.':

            borealis_major_revision = record_dict['borealis_git_hash'][1]
            borealis_minor_revision = record_dict['borealis_git_hash'][3]
        else:
            borealis_major_revision = 255
            borealis_minor_revision = 255

        # base offset for setting the toff field in SDARN DMap iqdat file.
        offset = 2 * record_dict['antenna_arrays_order'].shape[0] * \
            record_dict['num_samps']

        for beam_index, beam in enumerate(record_dict['beam_nums']):
            # grab this beam's data
            # shape is now num_antenna_arrays x num_sequences
            # x num_samps
            this_data = data[:, :, beam_index, :]
            # iqdat shape is num_sequences x num_antennas_arrays x
            # num_samps x 2 (real, imag), flattened
            reshaped_data = []
            for i in range(record_dict['num_sequences']):
                # get the samples for each array 1 after the other
                arrays = [this_data[x, i, :]
                          for x in range(this_data.shape[0])]
                # append
                reshaped_data.append(np.ravel(arrays))

            # (num_sequences x num_antenna_arrays x num_samps,
            # flattened)
            flattened_data = np.array(reshaped_data).flatten()

            int_data = np.empty(flattened_data.size * 2, dtype=np.float64)
            int_data[0::2] = flattened_data.real
            int_data[1::2] = flattened_data.imag

            np.minimum(int_data, 32767, int_data)
            np.maximum(int_data, -32768, int_data)

            int_data = np.array(int_data, dtype=np.int16)

            # flattening done in convert_to_dmap_datastructures
            sdarn_record_dict = {
                'radar.revision.major': np.int8(borealis_major_revision),
                'radar.revision.minor': np.int8(borealis_minor_revision),
                'origin.code': np.int8(100),  # indicating Borealis
                'origin.time':
                    datetime.\
                    utcfromtimestamp(record_dict['sqn_timestamps'][0]).\
                    strftime("%c"),
                'origin.command': 'Borealis ' + \
                                  record_dict['borealis_git_hash'] + \
                                  ' ' + record_dict['experiment_name'],
                'cp': np.int16(record_dict['experiment_id']),
                'stid': np.int16(code_to_stid[record_dict['station']]),
                'time.yr': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    year),
                'time.mo': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    month),
                'time.dy': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    day),
                'time.hr': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    hour),
                'time.mt': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    minute),
                'time.sc': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    second),
                'time.us': np.int32(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    microsecond),
                'txpow': np.int16(-1),
                'nave': np.int16(record_dict['num_sequences']),
                'atten': np.int16(0),
                'lagfr': np.int16(record_dict['first_range_rtt']),
                # smsep is in us; conversion from seconds
                'smsep': np.int16(1e6 / record_dict['rx_sample_rate']),
                'ercod': np.int16(0),
                'stat.agc': np.int16(record_dict['agc_status_word']),
                'stat.lopwr': np.int16(record_dict['lp_status_word']),
                # TODO: currently not implemented
                'noise.search': np.float32(record_dict['noise_at_freq'][0]),
                # TODO: currently not implemented
                'noise.mean': np.float32(0),
                'channel': np.int16(borealis_slice_id),
                'bmnum': np.int16(beam),
                'bmazm': np.float32(record_dict['beam_azms'][beam_index]),
                'scan': np.int16(record_dict['scan_start_marker']),
                # no digital receiver offset or rxrise required in
                # Borealis
                'offset': np.int16(0),
                'rxrise': np.int16(0),
                'intt.sc': np.int16(np.floor(record_dict['int_time'])),
                'intt.us': np.int32(np.fmod(record_dict['int_time'], 1.0) * \
                                    1e6),
                'txpl': np.int16(record_dict['tx_pulse_len']),
                'mpinc': np.int16(record_dict['tau_spacing']),
                'mppul': np.int16(len(record_dict['pulses'])),
                # an alternate lag-zero will be given, so subtract 1.
                'mplgs': np.int16(record_dict['lags'].shape[0] - 1),
                'nrang': np.int16(record_dict['num_ranges']),
                'frang': np.int16(round(record_dict['first_range'])),
                'rsep': np.int16(round(record_dict['range_sep'])),
                'xcf': np.int16('intf' in record_dict['antenna_arrays_order']),
                'tfreq': np.int16(record_dict['freq']),
                # mxpwr filler; cannot specify this information
                'mxpwr': np.int32(-1),
                # lvmax RST default
                'lvmax': np.int32(20000),
                'iqdata.revision.major': np.int32(1),
                'iqdata.revision.minor': np.int32(0),
                'combf': 'Converted from Borealis file: ' + origin_string +\
                         ' record ' + str(record_key) + \
                         ' with scaling factor = ' + str(scaling_factor) + \
                         ' ; Number of beams in record: ' + \
                         str(len(record_dict['beam_nums'])) + ' ; ' + \
                         record_dict['experiment_comment'] + ' ; ' + \
                         record_dict['slice_comment'],
                'seqnum': np.int32(record_dict['num_sequences']),
                'chnnum': np.int32(record_dict['antenna_arrays_order'].
                                   shape[0]),
                'smpnum': np.int32(record_dict['num_samps']),
                # NOTE: The following is a hack. This is currently how
                # iqdat files are being processed . RST make_raw does
                # not use first range information at all, only skip
                # number.
                # However ROS provides the number of ranges to the
                # first range as the skip number. Skip number is
                # documented as number to identify bad ranges due
                # to digital receiver rise time. Borealis skpnum should
                # in theory =0 as the first sample from Borealis
                # decimated (prebfiq) data is centred on the first
                # pulse.
                'skpnum': np.int32(record_dict['first_range'] / \
                                   record_dict['range_sep']),
                'ptab': record_dict['pulses'].astype(np.int16),
                'ltab': record_dict['lags'].astype(np.int16),
                # timestamps in ms, convert to seconds and us.
                'tsc': np.array([np.floor(x/1e3) for x in
                                 record_dict['sqn_timestamps']],
                                dtype=np.int32),
                'tus': np.array([np.fmod(x, 1000.0) * 1e3 for x in
                                 record_dict['sqn_timestamps']],
                                dtype=np.int32),
                'tatten': np.array([0] * record_dict['num_sequences'],
                                   dtype=np.int16),
                'tnoise': record_dict['noise_at_freq'].astype(np.float32),
                'toff': np.array([i * offset for i in
                                  range(record_dict['num_sequences'])],
                                 dtype=np.int32),
                'tsze': np.array([offset] * record_dict['num_sequences'],
                                 dtype=np.int32),
                'data': int_data
            }
        return sdarn_record_dict

    def _convert_rawacf_to_rawacf(self):
        """
        Conversion for Borealis hdf5 rawacf to SDARN DMap rawacf files.

        See Also
        --------
        __convert_rawacf_record
        https://superdarn.github.io/rst/superdarn/src.doc/rfc/0008.html
        https://borealis.readthedocs.io/en/master/
        BorealisRawacf
        Rawacf

        Raises
        ------
        BorealisConvert2RawacfError

        Returns
        -------
        dmap_recs, the records converted to DMap format
        """
        try:
            recs = []
            for record in self.borealis_records.items():
                sdarn_record_dict = \
                        self.__convert_rawacf_record(self.borealis_slice_id,
                                                     record,
                                                     self.borealis_filename,
                                                     self.scaling_factor)
                recs.append(sdarn_record_dict)
            self._sdarn_dict = recs
            self._sdarn_dmap_records = dict2dmap(recs)
        except Exception as e:
            raise borealis_exceptions.BorealisConvert2RawacfError(e) from e

    @staticmethod
    def __convert_rawacf_record(borealis_slice_id: int,
                                borealis_rawacf_record: tuple,
                                origin_string: str,
                                scaling_factor: int = 1) -> dict:
        """
        Converts a single record dict of Borealis rawacf data to a SDARN DMap
        record dict.

        Parameters
        ----------
        borealis_slice_id : int
            slice id integer of the borealis data, for conversion. Used
            as SDARN DMap channel identifier.
        borealis_rawacf_record : tuple(str, dict)
            Key is rawacf record timestamp, value is dictionary of rawacf
            record data.
        origin_string : str
            String representing origin of the Borealis data, typically
            Borealis filename.
        scaling_factor : int
            A scaling factor to adjust the integer values by, as the precision
            of floating points are much greater than the int16 can
            accommodate. This value is provided to multiply the data
            by before converting to int, to allow the noise floor to be
            seen, for instance.
        """

        # key value pair from Borealis record dictionary
        (record_key, record_dict) = borealis_rawacf_record

        shaped_data = {}
        # correlation_descriptors are num_beams, num_ranges, num_lags
        # scale by the scale squared to make up for the multiply
        # in correlation (integer max squared)
        shaped_data['main_acfs'] = record_dict['main_acfs'].reshape(
            record_dict['correlation_dimensions']).astype(
            np.complex64) *\
            ((np.iinfo(np.int16).max**2 * scaling_factor) /
             (record_dict['data_normalization_factor']**2))

        if 'intf_acfs' in record_dict.keys():
            shaped_data['intf_acfs'] = record_dict['intf_acfs'].reshape(
                record_dict['correlation_dimensions']).astype(np.complex64) *\
                ((np.iinfo(np.int16).max**2 * scaling_factor) /
                 (record_dict['data_normalization_factor']**2))
        if 'xcfs' in record_dict.keys():
            shaped_data['xcfs'] = record_dict['xcfs'].reshape(
                record_dict['correlation_dimensions']).astype(np.complex64) *\
                ((np.iinfo(np.int16).max**2 * scaling_factor) /
                 (record_dict['data_normalization_factor']**2))

        # Borealis git tag version numbers. If not a tagged version,
        # then use 255.255
        if record_dict['borealis_git_hash'][0] == 'v' and \
                record_dict['borealis_git_hash'][2] == '.':
            borealis_major_revision = record_dict['borealis_git_hash'][1]
            borealis_minor_revision = record_dict['borealis_git_hash'][3]
        else:
            borealis_major_revision = 255
            borealis_minor_revision = 255

        for beam_index, beam in enumerate(record_dict['beam_nums']):
            # this beam, all ranges lag 0
            lag_zero = shaped_data['main_acfs'][beam_index, :, 0]
            lag_zero[-10:] = shaped_data['main_acfs'][beam_index, -10:, -1]
            lag_zero_power = (lag_zero.real**2 + lag_zero.imag**2)**0.5

            correlation_dict = {}
            for key in shaped_data:
                # num_ranges x num_lags (complex)
                this_correlation = shaped_data[key][beam_index, :, :-1]
                # set the lag0 to the alternate lag0 for the end of the
                # array (when interference of first pulse would occur)
                this_correlation[-10:, 0] = \
                    shaped_data[key][beam_index, -10:, -1]
                # shape num_beams x num_ranges x num_la gs, now
                # num_ranges x num_lags-1 b/c alternate lag-0 combined
                # with lag-0 (only used for last ranges)

                # (num_ranges x num_lags, flattened)
                flattened_data = np.array(this_correlation).flatten()

                int_data = np.empty(flattened_data.size * 2, dtype=np.float32)
                int_data[0::2] = flattened_data.real
                int_data[1::2] = flattened_data.imag
                # num_ranges x num_lags x 2; num_lags is one less than
                # in Borealis file because Borealis keeps alternate
                # lag0
                new_data = int_data.reshape(
                    record_dict['correlation_dimensions'][1],
                    record_dict['correlation_dimensions'][2]-1,
                    2)
                # NOTE: Flattening happening in
                # convert_to_dmap_datastructures
                # place the SDARN-style array in the dict
                correlation_dict[key] = new_data

            sdarn_record_dict = {
                'radar.revision.major': np.int8(borealis_major_revision),
                'radar.revision.minor': np.int8(borealis_minor_revision),
                'origin.code': np.int8(100),  # indicating Borealis
                'origin.time':
                    datetime.\
                    utcfromtimestamp(record_dict['sqn_timestamps'][0]).\
                    strftime("%c"),
                'origin.command': 'Borealis ' +\
                                  record_dict['borealis_git_hash'] +\
                                  ' ' + record_dict['experiment_name'],
                'cp': np.int16(record_dict['experiment_id']),
                'stid': np.int16(code_to_stid[record_dict['station']]),
                'time.yr': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    year),
                'time.mo': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    month),
                'time.dy': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    day),
                'time.hr': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    hour),
                'time.mt': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    minute),
                'time.sc': np.int16(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    second),
                'time.us': np.int32(datetime.
                                    utcfromtimestamp(
                                        record_dict['sqn_timestamps'][0]).
                                    microsecond),
                'txpow': np.int16(-1),
                # see Borealis documentation
                'nave': np.int16(record_dict['num_sequences']),
                'atten': np.int16(0),
                'lagfr': np.int16(record_dict['first_range_rtt']),
                'smsep': np.int16(1e6/record_dict['rx_sample_rate']),
                'ercod': np.int16(0),
                'stat.agc': np.int16(record_dict['agc_status_word']),
                'stat.lopwr': np.int16(record_dict['lp_status_word']),
                # TODO: currently not implemented
                'noise.search': np.float32(record_dict['noise_at_freq'][0]),
                # TODO: currently not implemented
                'noise.mean': np.float32(0),
                'channel': np.int16(borealis_slice_id),
                'bmnum': np.int16(beam),
                'bmazm': np.float32(record_dict['beam_azms'][beam_index]),
                'scan': np.int16(record_dict['scan_start_marker']),
                # no digital receiver offset or rxrise required in
                # Borealis
                'offset': np.int16(0),
                'rxrise': np.int16(0),
                'intt.sc': np.int16(np.floor(record_dict['int_time'])),
                'intt.us': np.int32(np.fmod(record_dict['int_time'], 1.0) * \
                                    1e6),
                'txpl': np.int16(record_dict['tx_pulse_len']),
                'mpinc': np.int16(record_dict['tau_spacing']),
                'mppul': np.int16(len(record_dict['pulses'])),
                # an alternate lag-zero will be given.
                'mplgs': np.int16(record_dict['lags'].shape[0] - 1),
                'nrang': np.int16(record_dict['correlation_dimensions'][1]),
                'frang': np.int16(round(record_dict['first_range'])),
                'rsep': np.int16(round(record_dict['range_sep'])),
                # False if list is empty.
                'xcf': np.int16(bool('xcfs' in record_dict.keys())),
                'tfreq': np.int16(record_dict['freq']),
                'mxpwr': np.int32(-1),
                'lvmax': np.int32(20000),
                'rawacf.revision.major': np.int32(1),
                'rawacf.revision.minor': np.int32(0),
                'combf': 'Converted from Borealis file: ' + origin_string + \
                         ' record ' + str(record_key) + \
                         ' with scaling factor = ' + str(scaling_factor) + \
                         ' ; Number of beams in record: ' + \
                         str(len(record_dict['beam_nums'])) + ' ; ' + \
                         record_dict['experiment_comment'] + ' ; ' + \
                         record_dict['slice_comment'],
                'thr': np.float32(0),
                'ptab': record_dict['pulses'].astype(np.int16),
                'ltab': record_dict['lags'].astype(np.int16),
                'pwr0': lag_zero_power.astype(np.float32),
                # list from 0 to num_ranges
                'slist': np.array(list(
                            range(0, record_dict['correlation_dimensions'][1]))
                            ).astype(np.int16),
                'acfd': correlation_dict['main_acfs'],
                'xcfd': correlation_dict['xcfs']
            }

        return sdarn_record_dict
