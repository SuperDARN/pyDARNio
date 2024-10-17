# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
"""
This file contains classes and functions for
converting of Borealis file types.

Classes
-------
BorealisConvert: Writes Borealis SuperDARN files types to
    SuperDARN SDARN files with DMap record structure

Exceptions
----------
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
BorealisConvert makes use of SDARNWrite to write to SuperDARN file types

For more information on Borealis data files and how they convert to dmap,
see: https://borealis.readthedocs.io/en/master/

Future Work
-----------
Update noise values in SDarn fields when these can be calculated.

"""
import logging
import numpy as np

from datetime import datetime
from typing import Union

from pydarnio import (borealis_exceptions, BorealisRead, SDarnWrite, dict2dmap)
from pydarnio.borealis.borealis_utilities import code_to_stid
from pydarnio.borealis.v1_onwards import BorealisV1Convert

pyDARNio_log = logging.getLogger('pyDARNio')


class BorealisConvert(BorealisRead):
    """
    Class for converting Borealis filetypes to SDARN DMap filetypes.

    See Also
    --------
    BorealisRawacf
    BorealisBfiq
    Rawacf
    Iqdat
    BorealisRead
    BorealisSiteRead
    BorealisArrayRead
    SDarnWrite

    Attributes
    ----------
    allowed_conversions: dict
        Mapping of allowed Borealis types to their corresponding
        SDARN DMap converted types.
    filename: str; borealis_filename: str
        The filename of the Borealis HDF5 file being read.
    borealis_filetype: str
        The type of Borealis file. Types include:
        'bfiq'
        'rawacf'
    reader: Union[BorealisSiteWrite, BorealisArrayWrite]
        the wrapped BorealisSiteRead or BorealisArrayRead instance
    borealis_file_structure: Union[str, None]
        The structure of the file read, 'site' or 'array'. Default None.
    records: OrderedDict{dict}; borealis_records: OrderedDict{dict}
        The Borealis data in a dictionary of records, according to the
        site file format. The conversion functions use
        record format only as that is what is used by SDARN DMap
        files.
    arrays: dict
        The Borealis data in a dictionary of arrays, according to the
        restructured array file format.
    record_names: list[str]
        A sorted list of the set of record names in the HDF5 file read.
        These correspond to Borealis file record write times (in ms since
        epoch), and are equal to the group names in the site file types.
    sdarn_filename: str
        The filename of the SDARN DMap file to be written.
    sdarn_dmap_records: list[dict]
        The converted DMap records to write to file.
    sdarn_dict: dict
        The dictionary of SDARN records before the conversion to DMap format.
    sdarn_filetype: str
        The dmap filetype converted to. 'rawacf' and 'iqdat' are allowed.
    borealis_slice_id: int
        The slice id of the file being converted. Used as channel identifier
        in SDARN DMap records.
    scaling_factor: int
        The scaling factor the data has been multiplied by before converting
        to integers for the corresponding dmap format.
    """

    __allowed_conversions = {'rawacf': 'rawacf', 'bfiq': 'iqdat'}

    def __init__(self, borealis_filename: str, borealis_filetype: str,
                 sdarn_filename: str, borealis_slice_id: int = None,
                 borealis_file_structure: Union[str, None] = None,
                 scaling_factor: int = 1):
        """
        Convert HDF5 Borealis records to a given SDARN file with DMap format.

        Parameters
        ----------
        borealis_filename: str
            file name containing Borealis hdf5 data. Name contains slice id
            and allows the file read.
        borealis_filetype: str
            The type of Borealis file. Types allowed to convert:
            'bfiq' -> 'iqdat'
            'rawacf' -> 'rawacf'
        sdarn_filename: str
            The filename of the SDARN DMap file to be written.
        borealis_slice_id: int
            The slice id of the file being converted. Only necessary for
            files produced by Borealis versions before v0.5.
        borealis_file_structure: Union[str, None]
            The write structure of the file provided. Possible types are
            'site', 'array', or None. If None (default), array read will be
            attempted first followed by site.
        scaling_factor : int
            A scaling factor to adjust the integer values by, as the precision
            of bfiq and rawacf floating points are much greater than the int16
            can accommodate. This value is provided to multiply the data
            by before converting to int, to allow the noise floor to be
            seen, for instance.

        Raises
        ------
        BorealisStructureError
        BorealisConversionTypesError
        ConvertFileOverWriteError
        """

        super(BorealisConvert, self).__init__(borealis_filename,
                                              borealis_filetype,
                                              borealis_file_structure)

        # Attributes written in the parent init (BorealisRead):
        #   filename : str
        #   borealis_filetype : str
        #   reader: Union[BorealisSiteRead, BorealisArrayRead]
        #   borealis_file_structure : str
        #   records : dict (of data)
        #   arrays : dict (of data)

        self.borealis_records = self.records
        self.sdarn_filename = sdarn_filename
        self.borealis_filename = self.filename

        if self.version[0] < 1:
            try:
                first_key = list(self.records.keys())[0]
                self._borealis_slice_id = self.records[first_key]['slice_id']
            except KeyError as kerr:
                if borealis_slice_id is not None:
                    self._borealis_slice_id = int(borealis_slice_id)
                else:
                    raise borealis_exceptions.BorealisStructureError(
                        'The slice_id could not be found in the file: Borealis '
                        'files produced before Borealis v0.5 must provide the '
                        'slice_id value to the BorealisConvert class.') from kerr

        self._sdarn_dmap_records = {}
        self._sdarn_dict = {}
        self._scaling_factor = scaling_factor
        try:
            self._sdarn_filetype = self.__allowed_conversions[
                    self.borealis_filetype]
        except KeyError:
            raise borealis_exceptions.BorealisConversionTypesError(
                self.sdarn_filename, self.borealis_filetype,
                self.__allowed_conversions)

        if self.borealis_filename == self.sdarn_filename:
            raise borealis_exceptions.ConvertFileOverWriteError(
                    self.borealis_filename)
        self._write_to_sdarn()

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
            if self.version[0] < 1:
                if self._is_convertible_to_iqdat():
                    self._convert_bfiq_to_iqdat()
            else:
                self._sdarn_dmap_records = BorealisV1Convert.bfiq_to_dmap(self.borealis_filename)
        elif self.sdarn_filetype == 'rawacf':
            if self.version[0] < 1:
                if self._is_convertible_to_rawacf():
                    self._convert_rawacf_to_rawacf()
            else:
                self._sdarn_dmap_records = BorealisV1Convert.rawacf_to_dmap(self.borealis_filename)
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

                # Borealis git tag version numbers. If not a tagged version,
                # then use 255.255
                if record['borealis_git_hash'][0] == 'v':  # tagged version, non-tagged versions have hexadecimal
                    version = record['borealis_git_hash'].split('-')[0].split('.')
                    borealis_major_revision = version[0][1:]  # strip off the 'v'
                    borealis_minor_revision = version[1]
                else:
                    borealis_major_revision = 255
                    borealis_minor_revision = 255

                if borealis_major_revision == 0 and borealis_minor_revision <= 5:
                    # Bfiq generated with borealis v0.5 or older
                    blanked = record['pulses'] * sample_spacing
                else:
                    # Bfiq generated with borealis v0.6 or newer, or untagged version
                    normal_blanked_1 = record['pulses'] * sample_spacing
                    if np.array_equal(normal_blanked_1, record['blanked_samples']):
                        # If file generated with untagged borealis version, it could be like v0.5
                        # and still have valid blanked samples, so we check that
                        blanked = normal_blanked_1
                    else:
                        normal_blanked_2 = normal_blanked_1 + 1
                        blanked = np.concatenate((normal_blanked_1, normal_blanked_2))
                        blanked = np.sort(blanked)

                if not np.array_equal(record['blanked_samples'], blanked):
                    raise borealis_exceptions.\
                            BorealisConvert2IqdatError(
                                'Increased complexity: Borealis bfiq file'
                                ' record {} blanked_samples {} is not correct'
                                ' for pulses array converted to sample number '
                                '{} * {}.'.format(record_key,
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
            - the blanked_samples array = pulses array for all records with Borealis v0.5 or older;
            - the blanked_samples array is the same as the pulses array * sample number, but also
              blanks the sample after each pulse, for Borealis v0.6 or newer
            - the pulse_phase_offset array contains all zeroes for all records

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

                # Borealis git tag version numbers. If not a tagged version,
                # then use 255.255
                if record['borealis_git_hash'][0] == 'v':  # tagged version, non-tagged versions have hexadecimal
                    version = record['borealis_git_hash'].split('-')[0].split('.')
                    borealis_major_revision = version[0][1:]  # strip off the 'v'
                    borealis_minor_revision = version[1]
                else:
                    borealis_major_revision = 255
                    borealis_minor_revision = 255

                if borealis_major_revision == 0 and borealis_minor_revision <= 5:
                    # Rawacf generated with borealis v0.5 or older
                    blanked = record['pulses'] * sample_spacing
                else:
                    # Rawacf generated with borealis v0.6 or newer, or untagged version
                    normal_blanked_1 = record['pulses'] * sample_spacing
                    if np.array_equal(normal_blanked_1, record['blanked_samples']):
                        # If file generated with untagged borealis version, it could be like v0.5
                        # and still have valid blanked samples, so we check that
                        blanked = normal_blanked_1
                    else:
                        normal_blanked_2 = normal_blanked_1 + 1
                        blanked = np.concatenate((normal_blanked_1, normal_blanked_2))
                        blanked = np.sort(blanked)

                if not np.array_equal(record['blanked_samples'], blanked):
                    raise borealis_exceptions.\
                            BorealisConvert2RawacfError(
                                'Increased complexity: Borealis rawacf file'
                                ' record {} blanked_samples {} is not correct'
                                ' for pulses array converted to sample number '
                                '{} * {}.'.format(record_key,
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
                record_dict_list = \
                        self.__convert_bfiq_record(self.borealis_slice_id,
                                                   record,
                                                   self.borealis_filename,
                                                   self.scaling_factor)
                recs.extend(record_dict_list)
            self._sdarn_dict = recs
            self._sdarn_dmap_records = dict2dmap(recs)
        except Exception as e:
            raise borealis_exceptions.BorealisConvert2IqdatError(e) from e

    @staticmethod
    def __convert_bfiq_record(borealis_slice_id: int,
                              borealis_bfiq_record: tuple,
                              origin_string: str,
                              scaling_factor: int = 1) -> list:
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
        if record_dict['borealis_git_hash'][0] == 'v':  # tagged version, non-tagged versions have hexadecimal
            version = record_dict['borealis_git_hash'].split('-')[0].split('.')
            borealis_major_revision = version[0][1:]   # strip off the 'v'
            borealis_minor_revision = version[1]
        else:
            borealis_major_revision = 255
            borealis_minor_revision = 255

        # base offset for setting the toff field in SDARN DMap iqdat file.
        offset = 2 * record_dict['antenna_arrays_order'].shape[0] * \
            record_dict['num_samps']

        record_dict_list = []
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

            # AGC Status Word only introduced in Borealis v0.6 onwards,
            # so it can be set to zero if not present
            if 'agc_status_word' not in record_dict.keys():
                agc_sw = 0
            else:
                agc_sw = record_dict['agc_status_word']

            # Low Power Status Word only introduced in Borealis v0.6 onwards,
            # so it can be set to zero if not present
            if 'lp_status_word' not in record_dict.keys():
                lp_sw = 0
            else:
                lp_sw = record_dict['lp_status_word']

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
                'stat.agc': np.int16(agc_sw),
                'stat.lopwr': np.int16(lp_sw),
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
            record_dict_list.append(sdarn_record_dict)
        return record_dict_list

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
                record_dict_list = \
                        self.__convert_rawacf_record(self.borealis_slice_id,
                                                     record,
                                                     self.borealis_filename,
                                                     self.scaling_factor)
                recs.extend(record_dict_list)
            self._sdarn_dict = recs
            self._sdarn_dmap_records = dict2dmap(recs)
        except Exception as e:
            raise borealis_exceptions.BorealisConvert2RawacfError(e) from e

    @staticmethod
    def __convert_rawacf_record(borealis_slice_id: int,
                                borealis_rawacf_record: tuple,
                                origin_string: str,
                                scaling_factor: int = 1) -> list:
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
        data_dimensions = record_dict.get('correlation_dimensions', None)     # Until Borealis v0.7
        if data_dimensions is None:
            data_dimensions = record_dict.get('data_dimensions')        # Borealis v0.7 onwards

        # correlation_descriptors are num_beams, num_ranges, num_lags
        # scale by the scale squared to make up for the multiply
        # in correlation (integer max squared)
        shaped_data['main_acfs'] = record_dict['main_acfs'].reshape(
            data_dimensions).astype(
            np.complex64) *\
            ((np.iinfo(np.int16).max**2 * scaling_factor) /
             (record_dict['data_normalization_factor']**2))

        if 'intf_acfs' in record_dict.keys():
            shaped_data['intf_acfs'] = record_dict['intf_acfs'].reshape(
                data_dimensions).astype(np.complex64) *\
                ((np.iinfo(np.int16).max**2 * scaling_factor) /
                 (record_dict['data_normalization_factor']**2))
        if 'xcfs' in record_dict.keys():
            shaped_data['xcfs'] = record_dict['xcfs'].reshape(
                data_dimensions).astype(np.complex64) *\
                ((np.iinfo(np.int16).max**2 * scaling_factor) /
                 (record_dict['data_normalization_factor']**2))

        # Borealis git tag version numbers. If not a tagged version,
        # then use 255.255
        if record_dict['borealis_git_hash'][0] == 'v':  # tagged version, non-tagged versions are hexadecimal
            version = record_dict['borealis_git_hash'].split('-')[0].split('.')
            borealis_major_revision = version[0][1:]    # strip off the 'v'
            borealis_minor_revision = version[1]
        else:
            borealis_major_revision = 255
            borealis_minor_revision = 255

        record_dict_list = []
        for beam_index, beam in enumerate(record_dict['beam_nums']):
            # this beam, all ranges lag 0
            lag_zero = shaped_data['main_acfs'][beam_index, :, 0]

            # Historically, the following line was un-commented. It's purpose was to replace the lag0 data for far
            # ranges which were contaminated by the second pulse in a sequence. However, it only worked for a small
            # subset of conditions; specifically, it failed for experiments where the second pulse occurred earlier
            # or the number of range gates was larger than normal. This replacement is now handled in borealis, and
            # is more flexible to deal with the variety of different conditions possible from experiments.
            # lag_zero[-10:] = shaped_data['main_acfs'][beam_index, -10:, -1]

            lag_zero_power = (lag_zero.real**2 + lag_zero.imag**2)**0.5

            correlation_dict = {}
            for key in shaped_data:
                # num_ranges x num_lags (complex)
                this_correlation = shaped_data[key][beam_index, :, :-1]

                ##### Similar to above, this line has been commented out to avoid far-range lag0 replacement
                ##### as it is now handled in borealis.
                # this_correlation[-10:, 0] = \
                #     shaped_data[key][beam_index, -10:, -1]

                # (num_ranges x num_lags, flattened)
                flattened_data = np.array(this_correlation).flatten()

                int_data = np.empty(flattened_data.size * 2, dtype=np.float32)
                int_data[0::2] = flattened_data.real
                int_data[1::2] = flattened_data.imag
                # num_ranges x num_lags x 2; num_lags is one less than
                # in Borealis file because Borealis keeps alternate
                # lag0
                new_data = int_data.reshape(
                    data_dimensions[1],
                    data_dimensions[2]-1,
                    2)
                # NOTE: Flattening happening in
                # convert_to_dmap_datastructures
                # place the SDARN-style array in the dict
                correlation_dict[key] = new_data

            # AGC Status Word only introduced in Borealis v0.6 onwards,
            # so it can be set to zero if not present
            if 'agc_status_word' not in record_dict.keys():
                agc_sw = 0
            else:
                agc_sw = record_dict['agc_status_word']

            # Low Power Status Word only introduced in Borealis v0.6 onwards,
            # so it can be set to zero if not present
            if 'lp_status_word' not in record_dict.keys():
                lp_sw = 0
            else:
                lp_sw = record_dict['lp_status_word']

            # TX Antenna Mag only introduced in Borealis v0.7 onwards, so txpow defaults to -1 if not present.
            # If present, txpow is a bitfield mapping of whether each antenna was transmitting. Antenna 15 is the
            # MSB, and Antenna 0 the LSB. Since txpow is a signed int in DMAP, -1 means all antennas transmitting.
            if 'tx_antenna_phases' not in record_dict.keys():
                txpow = -1      # This is the same as if all antennas were transmitting.
            else:
                txpow = np.uint16()
                for i in range(len(record_dict['tx_antenna_phases'])):
                    if np.abs(record_dict['tx_antenna_phases'][i]) > 0:
                        txpow += 1 << i

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
                'txpow': np.int16(txpow),
                # see Borealis documentation
                'nave': np.int16(record_dict['num_sequences']),
                'atten': np.int16(0),
                'lagfr': np.int16(record_dict['first_range_rtt']),
                'smsep': np.int16(1e6/record_dict['rx_sample_rate']),
                'ercod': np.int16(0),
                'stat.agc': np.int16(agc_sw),
                'stat.lopwr': np.int16(lp_sw),
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
                'nrang': np.int16(data_dimensions[1]),
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
                            range(0, data_dimensions[1]))
                            ).astype(np.int16),
                'acfd': correlation_dict['main_acfs'],
                'xcfd': correlation_dict['xcfs']
            }
            record_dict_list.append(sdarn_record_dict)

        return record_dict_list
