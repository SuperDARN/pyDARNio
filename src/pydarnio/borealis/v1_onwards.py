# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
"""
This file contains classes and functions for
converting of Borealis v1.0+ file types.

Classes
-------
BorealisRead
BorealisWrite

Exceptions
----------
BorealisFileTypeError
BorealisFieldMissingError
BorealisExtraFieldError
BorealisDataFormatTypeError
BorealisNumberOfRecordsError
BorealisConversionTypesError
BorealisConvert2IqdatError
BorealisConvert2RawacfError
ConvertFileOverWriteError

See Also
--------
BorealisConvert

For more information on Borealis data files and how they convert to SDARN
filetypes, see: https://borealis.readthedocs.io/en/latest/
"""

import logging
from datetime import datetime

import h5py
import numpy as np

from pydarnio import borealis_exceptions
from pydarnio.borealis.borealis_utilities import code_to_stid

pyDARNio_log = logging.getLogger('pyDARNio')


class BorealisV1Read:
    """
    Class for reading Borealis v1.0+ filetypes.

    See Also
    --------
    BaseFormat
    BorealisRawacf
    BorealisBfiq
    BorealisAntennasIq
    BorealisRawrf
    """

    @staticmethod
    def read_records(filename: str):
        """
        Reads in records and metadata from the file.
        """
        records = list()
        metadata = dict()
        with h5py.File(filename, 'r') as f:
            metadata_group = f['metadata']
            for k in list(metadata_group.keys()):
                metadata[k] = metadata_group[k][()]
            rec_names = sorted(list(f.keys()))
            rec_names.remove('metadata')
            for name in rec_names:
                rec = f[name]
                rec_dict = dict()
                for k in list(rec.keys()):
                    if k not in metadata.keys():
                        rec_dict[k] = rec[k][()]
                records.append(rec_dict)

        return records, metadata

    @staticmethod
    def read_as_xarray(filename: str):
        try:
            import xarray as xr
            datasets = list()
            with h5py.File(filename, 'r') as f:
                keys = sorted(list(f.keys()))
                keys.remove("metadata")
                for key in keys:
                    ds = xr.open_dataset(filename, group=f"/{key}", phony_dims="access")
                    datasets.append(ds)
            return datasets
        except ImportError:
            raise ImportError("Unable to import xarray. Ensure that you have xarray installed with the `h5netcdf` "
                              "engine (this can be installed by installing pydarnio with the `xarray` option, e.g. "
                              "`pip install pydarnio[xarray]`)")

    @staticmethod
    def read_arrays(filename: str):
        """
        Reads in records and metadata from the file, combining like-fields of records into
        single large arrays.
        """
        aveperiod_indices = []
        beam_indices = []
        bfiq_indices = []
        metadata = dict()
        with h5py.File(filename, 'r') as f:
            metadata_group = f['metadata']
            for k in list(metadata_group.keys()):
                metadata[k] = metadata_group[k][()]
            rec_names = sorted(list(f.keys()))
            rec_names.remove('metadata')
            field_names = sorted(list(f[rec_names[0]].keys()))
            for metadata_name in metadata.keys():
                field_names.remove(metadata_name)

            fields_lists = {k: list() for k in field_names}
            for i, name in enumerate(rec_names):
                rec = f[name]
                for k in field_names:
                    data = rec[k][()]
                    if k == 'sqn_timestamps':
                        aveperiod_indices.extend([i] * len(data))
                    if k == 'beam_nums':
                        beam_indices.extend([i] * len(data))
                    if k == 'bfiq_data':
                        bfiq_indices.extend([i] * data.shape[1] * data.shape[2])
                        data = data.reshape((data.shape[0], -1, data.shape[3]))
                    if k in ['beam_nums', 'beam_azms', 'rx_main_phases', 'rx_intf_phases', 'tx_antenna_phases']:
                        data = data.flatten()
                    if k in ['intf_acfs', 'main_acfs', 'xcfs']:
                        data = data.reshape((-1,) + data.shape[1:])
                    fields_lists[k].append(data)

        rec_dict = dict()
        for k, v in fields_lists.items():
            if k in ['antennas_iq_data', 'bfiq_data', 'rawrf_data']:
                rec_dict[k] = np.concatenate(v, axis=1)
            elif k in ['beam_nums', 'beam_azms', 'pulse_phase_offset', 'sqn_timestamps',
                       'intf_acfs', 'main_acfs', 'xcfs']:
                rec_dict[k] = np.concatenate(v, axis=0)
            else:
                rec_dict[k] = np.stack(v, axis=0)
        rec_dict.update(metadata)
        rec_dict['aveperiod_indices'] = np.array(aveperiod_indices, dtype=np.uint32)
        rec_dict['aveperiod'] = np.arange(len(rec_names), dtype=np.uint32)
        rec_dict['beam_indices'] = np.array(beam_indices, dtype=np.uint32)
        if 'bfiq_data' in rec_dict.keys():
            rec_dict['bfiq_indices'] = np.array(bfiq_indices, dtype=np.uint32)

        return rec_dict

    @staticmethod
    def read_arrays_as_xarray(filename: str):
        """
        Reads in the file as array-formatted fields, using xarray.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("Unable to import xarray. Ensure that you have xarray installed with the `h5netcdf` "
                              "engine (this can be installed by installing pydarnio with the `xarray` option, e.g. "
                              "`pip install pydarnio[xarray]`)")

        arrays = BorealisV1Read.read_arrays(filename)

        coord_fields = {
            "aveperiod": ("aveperiod", ""),
            "aveperiod_indices": ("aveperiod_idx", "index into `aveperiod`"),
            "beam_indices": ("beam", "index into `aveperiod`"),
            "bfiq_indices": ("sequence_times_beam", "index into `aveperiod`"),
            "blanked_samples": ("blanks", ""),
            "beam_nums": ("beam_num", ""),
            "coordinates": ("coord", ""),
            "antenna_arrays": ("array", ""),
            "antennas": ("antenna", ""),
            "cfs_freqs": ("freq", "Hz"),
            "lags": ("lag", ""),
            "lag_numbers": ("lag number", "`tau_spacing`"),
            "lag_pulse_descriptors": ("pulse_descriptor", ""),
            "pulses": ("pulse", "`tau_spacing`"),
            "range_gates": ("range_gate", ""),
            "rx_antennas": ("rx_antenna", ""),
            "rx_main_antennas": ("main_antenna", ""),
            "rx_intf_antennas": ("intf_antenna", ""),
            "sample_time": ("sample_time", "μs"),
            "sqn_timestamps": ("sequence", ""),
            "tx_antennas": ("tx_antenna", ""),
        }
        data_fields = {
            "agc_status_word": ["aveperiod"],
            "antenna_locations": ["antennas", "coordinates"],
            "antennas_iq_data": ["rx_antennas", "sqn_timestamps", "sample_time"],
            "averaging_method": [],
            "beam_azms": ["beam_nums"],
            "bfiq_data": ["antenna_arrays", "bfiq_indices", "sample_time"],
            "borealis_git_hash": [],
            "cfs_masks": ["aveperiod", "cfs_freqs"],
            "cfs_noise": ["aveperiod", "cfs_freqs"],
            "cfs_range": [],
            "data_normalization_factor": [],
            "experiment_comment": [],
            "experiment_id": [],
            "experiment_name": [],
            "first_range": [],
            "first_range_rtt": [],
            "freq": ["aveperiod"],
            "gps_locked": ["aveperiod"],
            "gps_to_system_time_diff": ["aveperiod"],
            "int_time": ["aveperiod"],
            "intf_acfs": ["beam_nums", "range_gates", "lag_numbers"],
            "lag_pulses": ["lags", "lag_pulse_descriptors"],
            "lp_status_word": ["aveperiod"],
            "main_acfs": ["beam_nums", "range_gates", "lag_numbers"],
            "num_sequences": ["aveperiod"],
            "num_slices": [],
            "pulse_phase_offset": ["sqn_timestamps", "pulses"],
            "range_sep": [],
            "rawrf_data": ["rx_antennas", "sqn_timestamps", "sample_time"],
            "rx_center_freq": [],
            "rx_sample_rate": [],
            "rx_main_phases": ["beam_nums", "rx_main_antennas"],
            "rx_intf_phases": ["beam_nums", "rx_intf_antennas"],
            "samples_data_type": [],
            "scan_start_marker": ["aveperiod"],
            "scheduling_mode": [],
            "slice_comment": [],
            "slice_id": [],
            "slice_interfacing": [],
            "station": [],
            "station_location": ["coordinates"],
            "tau_spacing": [],
            "tx_antenna_phases": ["aveperiod", "tx_antennas"],
            "tx_pulse_len": [],
            "xcfs": ["beam_nums", "range_gates", "lag_numbers"]
        }

        data_arrays = dict()
        for k, dims in data_fields.items():
            if k not in arrays.keys():
                continue
            coords = list()
            for d in dims:
                data = arrays[d]
                if d == 'sqn_timestamps':
                    data = np.array([datetime.utcfromtimestamp(x) for x in data])
                coords.append((coord_fields[d][0], data, {"units": coord_fields[d][1]}))
            if len(coords) == 0:
                coords = None
            data_arrays[k] = xr.DataArray(arrays[k], coords)

        return xr.Dataset(data_arrays)


class BorealisV1Convert:
    @staticmethod
    def bfiq_to_dmap(filename: str):
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
            records, metadata = BorealisV1Read.read_records(filename)
            for record in records:
                record_dict_list = BorealisV1Convert.convert_bfiq_record(
                    record,
                    metadata,
                    filename,
                )
                recs.extend(record_dict_list)
        except Exception as e:
            raise borealis_exceptions.BorealisConvert2IqdatError(e) from e
        return recs

    @staticmethod
    def convert_bfiq_record(bfiq_dict: dict, metadata_dict: dict, origin_string: str) -> list:
        """
        Converts a single record dict of Borealis bfiq data to a SDARN DMap
        record dict.

        Parameters
        ----------
        bfiq_dict: dict
            dictionary of bfiq record data.
        metadata_dict: dict
            dictionary of file-level metadata.
        origin_string: str
            String representing origin of the Borealis data, typically
            Borealis filename.
        """
        record_dict = {**bfiq_dict, **metadata_dict}
        # data_descriptors (dimensions) are num_antenna_arrays,
        # num_sequences, num_beams, num_samps
        # scale by normalization and then scale to integer max as per
        # dmap style
        data = (
            record_dict['bfiq_data'] /
            record_dict['data_normalization_factor'] *
            np.iinfo(np.int16).max
        )

        githash = record_dict['borealis_git_hash'][:].decode('utf-8')
        version = githash.lstrip('v').split('-')[0].split('.')
        borealis_major_revision = int(version[0])
        borealis_minor_revision = int(version[1])

        # base offset for setting the toff field in SDARN DMap iqdat file.
        offset = 2 * record_dict['antenna_arrays'].shape[0] * data.shape[-1]

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

            start_time = datetime.utcfromtimestamp(record_dict['sqn_timestamps'][0])

            # flattening done in convert_to_dmap_datastructures
            sdarn_record_dict = {
                'radar.revision.major': np.int8(borealis_major_revision),
                'radar.revision.minor': np.int8(borealis_minor_revision),
                'origin.code': np.int8(100),  # indicating Borealis
                'origin.time': datetime.utcfromtimestamp(record_dict['sqn_timestamps'][0]).strftime("%c"),
                'origin.command': 'Borealis ' + githash + ' ' + record_dict['experiment_name'][:].decode('utf-8'),
                'cp': np.int16(record_dict['experiment_id']),
                'stid': np.int16(code_to_stid[record_dict['station'][:].decode('utf-8')]),
                'time.yr': np.int16(start_time.year),
                'time.mo': np.int16(start_time.month),
                'time.dy': np.int16(start_time.day),
                'time.hr': np.int16(start_time.hour),
                'time.mt': np.int16(start_time.minute),
                'time.sc': np.int16(start_time.second),
                'time.us': np.int32(start_time.microsecond),
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
                'noise.search': np.float32(0.0),
                # TODO: currently not implemented
                'noise.mean': np.float32(0),
                'channel': np.int16(record_dict['slice_id']),
                'bmnum': np.int16(beam),
                'bmazm': np.float32(record_dict['beam_azms'][beam_index]),
                'scan': np.int16(record_dict['scan_start_marker']),
                # no digital receiver offset or rxrise required in
                # Borealis
                'offset': np.int16(0),
                'rxrise': np.int16(0),
                'intt.sc': np.int16(np.floor(record_dict['int_time'])),
                'intt.us': np.int32(np.fmod(record_dict['int_time'], 1.0) * 1e6),
                'txpl': np.int16(record_dict['tx_pulse_len']),
                'mpinc': np.int16(record_dict['tau_spacing']),
                'mppul': np.int16(len(record_dict['pulses'])),
                # an alternate lag-zero will be given, so subtract 1.
                'mplgs': np.int16(record_dict['lag_numbers'].shape[0] - 1),
                'nrang': np.int16(len(record_dict['range_gates'])),
                'frang': np.int16(round(record_dict['first_range'])),
                'rsep': np.int16(round(record_dict['range_sep'])),
                'xcf': np.int16('intf' in record_dict['antenna_arrays']),
                'tfreq': np.int16(record_dict['freq']),
                # mxpwr filler; cannot specify this information
                'mxpwr': np.int32(-1),
                # lvmax RST default
                'lvmax': np.int32(20000),
                'iqdata.revision.major': np.int32(1),
                'iqdata.revision.minor': np.int32(0),
                'combf': 'Converted from Borealis file: ' + origin_string +
                         ' ; Number of beams in record: ' + str(len(record_dict['beam_nums'])) +
                         ' ; ' + record_dict['experiment_comment'][:].decode('utf-8') +
                         ' ; ' + record_dict['slice_comment'][:].decode('utf-8'),
                'seqnum': np.int32(record_dict['num_sequences']),
                'chnnum': np.int32(record_dict['antenna_arrays'].shape[0]),
                'smpnum': np.int32(len(record_dict['sample_time'])),
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
                'skpnum': np.int32(record_dict['first_range'] / record_dict['range_sep']),
                'ptab': record_dict['pulses'].astype(np.int16),
                'ltab': record_dict['lag_pulses'].astype(np.int16),
                # timestamps in ms, convert to seconds and us.
                'tsc': np.array([np.floor(x/1e3) for x in record_dict['sqn_timestamps']], dtype=np.int32),
                'tus': np.array([np.fmod(x, 1000.0) * 1e3 for x in record_dict['sqn_timestamps']], dtype=np.int32),
                'tatten': np.array([0] * record_dict['num_sequences'], dtype=np.int16),
                'tnoise': np.zeros(record_dict['num_sequences'], dtype=np.float32),
                'toff': np.array([i * offset for i in range(record_dict['num_sequences'])], dtype=np.int32),
                'tsze': np.array([offset] * record_dict['num_sequences'], dtype=np.int32),
                'data': int_data
            }
            record_dict_list.append(sdarn_record_dict)
        return record_dict_list

    @staticmethod
    def rawacf_to_dmap(filename: str):
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
            records, metadata = BorealisV1Read.read_records(filename)
            for record in records:
                record_dict_list = BorealisV1Convert.convert_rawacf_record(
                    record,
                    metadata,
                    filename,
                )
                recs.extend(record_dict_list)
        except Exception as e:
            raise borealis_exceptions.BorealisConvert2RawacfError(e) from e
        return recs

    @staticmethod
    def convert_rawacf_record(rawacf_dict: dict, metadata_dict: dict, origin_string: str) -> list:
        """
        Converts a single record dict of Borealis rawacf data to a SDARN DMap
        record dict.

        Parameters
        ----------
        rawacf_dict : dict
            dictionary of rawacf record data.
        metadata_dict: dict
            dictionary of file-level metadata.
        origin_string : str
            String representing origin of the Borealis data, typically
            Borealis filename.
        """
        record_dict = {**rawacf_dict, **metadata_dict}
        shaped_data = {}
        data_dimensions = record_dict['main_acfs'].shape

        # correlation_descriptors are num_beams, num_ranges, num_lags
        # scale by the scale squared to make up for the multiply
        # in correlation (integer max squared)
        shaped_data['main_acfs'] = record_dict['main_acfs'] * (np.iinfo(np.int16).max**2 / record_dict['data_normalization_factor']**2)

        if 'intf_acfs' in record_dict.keys():
            shaped_data['intf_acfs'] = record_dict['intf_acfs'] * (np.iinfo(np.int16).max**2 / record_dict['data_normalization_factor']**2)
        if 'xcfs' in record_dict.keys():
            shaped_data['xcfs'] = record_dict['xcfs'] * (np.iinfo(np.int16).max**2 / record_dict['data_normalization_factor']**2)

        githash = record_dict['borealis_git_hash'][:].decode('utf-8')
        version = githash.lstrip('v').split('-')[0].split('.')
        borealis_major_revision = int(version[0])
        borealis_minor_revision = int(version[1])

        record_dict_list = []
        for beam_index, beam in enumerate(record_dict['beam_nums']):
            # this beam, all ranges lag 0
            lag_zero = shaped_data['main_acfs'][beam_index, :, 0]
            lag_zero_power = np.abs(lag_zero)

            correlation_dict = {}
            for key in shaped_data:
                # num_ranges x num_lags (complex)
                this_correlation = shaped_data[key][beam_index, :, :-1]

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

            # TX Antenna Mag only introduced in Borealis v0.7 onwards, so txpow defaults to -1 if not present.
            # If present, txpow is a bitfield mapping of whether each antenna was transmitting. Antenna 15 is the
            # MSB, and Antenna 0 the LSB. Since txpow is a signed int in DMAP, -1 means all antennas transmitting.
            txpow = np.uint16()
            if 'tx_antenna_phases' not in record_dict.keys():
                raise ValueError(f'"tx_antenna_phases" not in record: {record_dict.keys()}')
            for i in range(len(record_dict['tx_antenna_phases'])):
                if np.abs(record_dict['tx_antenna_phases'][i]) > 0:
                    txpow += 1 << i

            start_time = datetime.utcfromtimestamp(record_dict['sqn_timestamps'][0])
            sdarn_record_dict = {
                'radar.revision.major': np.int8(borealis_major_revision),
                'radar.revision.minor': np.int8(borealis_minor_revision),
                'origin.code': np.int8(100),  # indicating Borealis
                'origin.time': datetime.utcfromtimestamp(record_dict['sqn_timestamps'][0]).strftime("%c"),
                'origin.command': 'Borealis ' + githash + ' ' + record_dict['experiment_name'][:].decode('utf-8'),
                'cp': np.int16(record_dict['experiment_id']),
                'stid': np.int16(code_to_stid[record_dict['station'][:].decode('utf-8')]),
                'time.yr': np.int16(start_time.year),
                'time.mo': np.int16(start_time.month),
                'time.dy': np.int16(start_time.day),
                'time.hr': np.int16(start_time.hour),
                'time.mt': np.int16(start_time.minute),
                'time.sc': np.int16(start_time.second),
                'time.us': np.int32(start_time.microsecond),
                'txpow': np.int16(txpow),
                # see Borealis documentation
                'nave': np.int16(record_dict['num_sequences']),
                'atten': np.int16(0),
                'lagfr': np.int16(record_dict['first_range_rtt']),
                'smsep': np.int16(1e6/record_dict['rx_sample_rate']),
                'ercod': np.int16(0),
                'stat.agc': np.int16(record_dict['agc_status_word']),
                'stat.lopwr': np.int16(record_dict['lp_status_word']),
                # TODO: currently not implemented
                'noise.search': np.float32(0.0),
                # TODO: currently not implemented
                'noise.mean': np.float32(0),
                'channel': np.int16(record_dict['slice_id']),
                'bmnum': np.int16(beam),
                'bmazm': np.float32(record_dict['beam_azms'][beam_index]),
                'scan': np.int16(record_dict['scan_start_marker']),
                # no digital receiver offset or rxrise required in
                # Borealis
                'offset': np.int16(0),
                'rxrise': np.int16(0),
                'intt.sc': np.int16(np.floor(record_dict['int_time'])),
                'intt.us': np.int32(np.fmod(record_dict['int_time'], 1.0) * 1e6),
                'txpl': np.int16(record_dict['tx_pulse_len']),
                'mpinc': np.int16(record_dict['tau_spacing']),
                'mppul': np.int16(len(record_dict['pulses'])),
                # an alternate lag-zero will be given.
                'mplgs': np.int16(record_dict['lag_numbers'].shape[0] - 1),
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
                'combf': 'Converted from Borealis file: ' + origin_string +
                         ' ; Number of beams in record: ' + str(len(record_dict['beam_nums'])) +
                         ' ; ' + record_dict['experiment_comment'][:].decode('utf-8') +
                         ' ; ' + record_dict['slice_comment'][:].decode('utf-8'),
                'thr': np.float32(0),
                'ptab': record_dict['pulses'].astype(np.int16),
                'ltab': record_dict['lag_pulses'].astype(np.int16),
                'pwr0': lag_zero_power.astype(np.float32),
                # list from 0 to num_ranges
                'slist': np.array(list(range(0, data_dimensions[1]))).astype(np.int16),
                'acfd': correlation_dict['main_acfs'],
                'xcfd': correlation_dict['xcfs']
            }
            record_dict_list.append(sdarn_record_dict)

        return record_dict_list
