# Copyright 2020 
# Author: Angeline G. Burrell, U.S. Naval Research Laboratory (NRL)

"""
This module contains methods to extract time series and scans from data dicts
"""

import datetime as dt
import logging
import numpy as np
from typing import Dict, List

pyDARNio_log = logging.getLogger('pyDARNio')

def datetime_from_records(records: List[dict], yr_key='time.yr',
                          mo_key='time.mo', dy_key='time.dy', hr_key='time.hr',
                          mt_key='time.mt', sc_key='time.sc',
                          us_key='time.us'):
    """ Create a datetime object from dictionary values

    Parameters
    ----------
    records : List[dict]
        list of records contained in dictionaries
    yr_key : str
        Year key (default='time.yr')
    mo_key : str
        Month key (default='time.mo')
    dy_key : str
        Day of month key (default='time.dy')
    hr_key : str or NoneType
        Hour of day key or None if not available (default='time.hr')
    mt_key : str or NoneType
        Minutes of hour or None if not available (default='time.mt')
    sc_key : str or NoneType
        Seconds of minute or None if not available (default='time.sc')
    us_key : str or NoneType
        Microseconds of seconds or None if not available (default='time.us')

    Returns
    -------
    dtimes : List[dt.datetime]
        a list of datetime objects, containing the datetimes for each
        record in the input list

    Notes
    -----
    If None is used for any time key, the smaller increment keys are ignored.
    Default key values are the appropriate time values for dmap records.

    Raises
    ------
    ValueError
        When year, month, or day keys are specified
    KeyError
        When a time key is missing from the data record

    """

    # Get the time keys that will be used
    time_keys = list()
    time_fmts = ["%Y", "%m", "%d", "%H", "%M", "%S", "%f"]
    for tkey in [yr_key, mo_key, dy_key, hr_key, mt_key, sc_key, us_key]:
        if tkey is None:
            break
        time_keys.append(tkey)

    # Test the input
    if len(time_keys) < 3:
        raise ValueError('need a minimum of year, month, and day as input')

    # Build the datetime format string
    dtime_fmt = " ".join(time_fmts[:len(time_keys)])

    # Build the time string and datetime objects for each dict
    dtimes = list()
    for rec_dict in records:
        # Get the datetime after building the time string
        dtime_str = " ".join([str(rec_dict[tkey]) for tkey in time_keys])
        dtimes.append(dt.datetime.strptime(dtime_str, dtime_fmt))

    return dtimes

def time_series_from_records(records: List[dict], target_keys: List[str],
                             time_kwargs={}, time_key='rec_time'):
    """ Extract lists of data values as time series

    Parameters
    ----------
    records : List[dict]
        list of records contained in dictionaries
    target_keys : List[str]
        List of keys to extract as a time series from the records
    time_kwargs : Dict
        Dictionary of time keyword arguements used as optional inputs for
        datetime_from_records (default={})
    time_key : str
        Output dictionary key name for the datetime values (default='rec_time')

    Returns
    -------
    time_series : Dict[np.array]
        Dict of lists that contain the time-series data from the target keys
        plus time, which has the key specified in the input parameter `time_key`

    Raises
    ------
    ValueError
        If target data values have sizes that cannot be reconciled into a
        time series (e.g., two multi-dimensional data sets that have different
        lenths)

    Notes
    -----
    Will skip records that do not contain all target keys

    """

    # Get the datetime of each record
    rec_time = datetime_from_records(records, **time_kwargs)

    if len(target_keys) == 0:
        return {time_key: np.asarray(rec_time)}

    # Initialize the output
    time_series = {tkey: list() for tkey in target_keys}
    time_series[time_key] = list()

    # For each record, get the desire data values
    for irec, rec_data in enumerate(records):
        # Get the shape of the records at this time, also test that all
        # data keys are present.  If not, do not save data from this record
        try:
            rec_lens = np.unique([1 if np.asarray(rec_data[tkey]).shape == ()
                                  else np.asarray(rec_data[tkey]).shape[0]
                                  for tkey in target_keys])
        except KeyError as kerr:
            pyDARNio_log.info("missing key(s) at record {:d}: {:}".format(
                irec, kerr))
            continue

        if len(rec_lens) > 2:
            raise ValueError('target data types have different sizes')
        max_rec = rec_lens.max()

        # Save the output data in lists the length of the maximum record length
        for tkey in target_keys:
            if max_rec == 1:
                time_series[tkey].append(rec_data[tkey])
            else:
                if rec_data[tkey].shape == ():
                    time_series[tkey].extend([rec_data[tkey]
                                              for i in range(max_rec)])
                else:
                    time_series[tkey].extend(list(rec_data[tkey]))

        # Save the time data for this record
        if max_rec == 1:
            time_series[time_key].append(rec_time[irec])
        else:
            time_series[time_key].extend([rec_time[irec]
                                          for i in range(max_rec)])

    # Recast the time series lists as numpy arrays
    for tkey in time_series.keys():
        time_series[tkey] = np.asarray(time_series[tkey])

    return time_series
            
            

def scans_from_records(records: List[dict]):
    """ Set up scans for easy locating

    Parameters
    ----------
    records : List[dict]
        list of records contained in dictionaries

    Returns
    -------
    scan_stime : List[dt.datetime]
        list of datetimes corresponding to the start of each scan
    scan_inds : List[list]
        list of lists containing the indices for each scan

    Raises
    ------
    ValueError
        If records do not have a scan flag, which is used to identify the scans

    """

    # Get the times and scan flags from the records
    scan_flags = time_series_from_records(records, ['scan'])
    if 'scan' not in scan_flags.keys():
        raise ValueError('input records do not have scan flags')

    # Makes a list of lists that contain the record indices for each scan
    istart = -1
    scan_inds = list()
    scan_stime = list()
    for i, sflg in enumerate(scan_flags['scan']):
        if abs(sflg) == 1:
            # This is a new scan, increment the scan index and save the
            # start time.
            istart += 1
            scan_inds.append([i])
            scan_stime.append(scan_flags['rec_time'][i])
        if sflg == 0:
            # This record belongs to the current scan
            scan_inds[istart].append(i)

    # Return the scan start times and scan record indices
    return scan_stime, scan_inds
