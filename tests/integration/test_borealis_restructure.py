# Copyright 2022 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
# Modified by Kevin Krieger
"""
This test suite is to test the integration for the following classes:
    BorealisRestructure
Support for the following Borealis file types:
    antennas_iq
    bfiq
    rawacf
And supports conversion of the following Borealis types:
    site -> array
    array -> site
"""
import gc
import logging
import numpy as np
import os
import unittest

from collections import OrderedDict

import pydarnio import BorealisRestructure

pydarnio_logger = logging.getLogger('pydarnio')

# Site Test files
borealis_site_bfiq_file = "../test_files/"
borealis_site_rawacf_file =\
        "../test_files/"
borealis_site_antennas_iq_file =\
        "../test_files/"

# Array Test files
borealis_array_bfiq_file = "../test_files/"
borealis_array_rawacf_file = "../test_files/"
borealis_array_antennas_iq_file =\
        "../test_files/"

# Problem files TODO
borealis_site_extra_field_file = ""
borealis_site_missing_field_file = ""
borealis_site_incorrect_data_format_file = ""

borealis_array_extra_field_file = ""
borealis_array_missing_field_file = ""
borealis_array_incorrect_data_format_file = ""
borealis_array_num_records_error_file = ""
borealis_empty_file = "../test_files/empty.rawacf"


class IntegrationBorealisRestructure(unittest.TestCase):
    """
    Testing class for integrations of BorealisRestructure
    """

    def setUp(self):
        self.source_rawacf_site_file = borealis_site_rawacf_file
        self.write_rawacf_site_file = 'test_rawacf.rawacf.hdf5.site'
        self.source_rawacf_array_file = borealis_array_rawacf_file
        self.write_rawacf_array_file = 'test_rawacf.rawacf.hdf5'

        self.source_bfiq_site_file = borealis_site_bfiq_file
        self.write_bfiq_site_file = 'test_bfiq.bfiq.hdf5.site'
        self.source_bfiq_array_file = borealis_array_bfiq_file
        self.write_bfiq_array_file = 'test_bfiq.bfiq.hdf5.array'

        self.source_antennas_iq_site_file = borealis_site_antennas_iq_file
        self.write_antennas_iq_site_file =\
            'test_antennas_iq.antennas_iq.hdf5.site'
        self.source_antennas_iq_array_file = borealis_array_antennas_iq_file
        self.write_antennas_iq_array_file =\
            'test_antennas_iq.antennas_iq.hdf5.array'

    # RESTRUCTURING TESTS
    def check_dictionaries_are_same(self, dict1, dict2):

        self.assertEqual(sorted(list(dict1.keys())),
                         sorted(list(dict2.keys())))
        for key1, value1 in dict1.items():
            if isinstance(value1, dict) or isinstance(value1, OrderedDict):
                self.check_dictionaries_are_same(value1, dict2[key1])
            elif isinstance(value1, np.ndarray):
                try:
                    if value1.dtype.type == np.unicode_:
                        self.assertTrue((value1 == dict2[key1]).all())
                    else:
                        # NaN==NaN will return False, so only index
                        # where not NaN.
                        not_nan_array = np.logical_not(np.isnan(value1))
                        other_not_nan_array = \
                            np.logical_not(np.isnan(dict2[key1]))
                        self.assertTrue(
                            (not_nan_array == other_not_nan_array).all())
                        self.assertTrue(
                            (value1[not_nan_array] ==
                                dict2[key1][not_nan_array]).all())
                except (AssertionError, TypeError, AttributeError):
                    print(key1, value1.dtype)
                    raise
            elif key1 == 'experiment_comment':
                continue  # combf has filename inside, can differ
            else:
                try:
                    self.assertEqual(value1, dict2[key1])
                except AssertionError:
                    print(key1, value1, dict2[key1])
                    raise

        return True

    def test_read_write_site_rawacf(self):
        """
        Test reading and then writing site rawacf data to a file.

        Checks:
            - records that pass the read from a file can then be written
            - records written and then read are the same as original
        """
        dm = pydarnio.BorealisRead(self.source_rawacf_site_file, 'rawacf',
                                   'site')
        records = dm.records
        _ = pydarnio.BorealisWrite(self.write_rawacf_site_file,
                                   records, 'rawacf', 'site')
        self.assertTrue(os.path.isfile(self.write_rawacf_site_file))
        dm2 = pydarnio.BorealisRead(self.write_rawacf_site_file, 'rawacf',
                                    'site')
        new_records = dm2.records
        dictionaries_are_same = self.check_dictionaries_are_same(records,
                                                                 new_records)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_rawacf_site_file)
        del _, dm, dm2, records, new_records

    def test_read_write_array_rawacf(self):
        """
        Test reading and then writing array structured rawacf data to a file.

        Checks:
            - arrays that pass the read from a file can then be written
            - arrays written and then read are the same as original
        """
        dm = pydarnio.BorealisRead(self.source_rawacf_array_file, 'rawacf',
                                   'array')
        arrays = dm.arrays
        _ = pydarnio.BorealisWrite(self.write_rawacf_array_file,
                                   arrays, 'rawacf',
                                   'array')
        self.assertTrue(os.path.isfile(self.write_rawacf_array_file))
        dm2 = pydarnio.BorealisRead(self.write_rawacf_array_file, 'rawacf',
                                    'array')
        new_arrays = dm2.arrays
        dictionaries_are_same = self.check_dictionaries_are_same(arrays,
                                                                 new_arrays)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_rawacf_array_file)
        del _, dm, dm2, arrays, new_arrays

    def test_read_site_write_array_rawacf(self):
        """
        Test reading, restructuring, writing, restructuring rawacf.

        Checks:
            - records that pass the read from a file can then be written
                as arrays
            - records restructured, written as arrays, read as arrays,
                restructured back to records are the same as original records
        """
        dm = pydarnio.BorealisRead(self.source_rawacf_site_file, 'rawacf',
                                   'site')
        records = dm.records

        arrays = dm.arrays  # restructuring happens here
        _ = pydarnio.BorealisWrite(self.write_rawacf_array_file,
                                   arrays, 'rawacf',
                                   'array')
        del dm, arrays
        self.assertTrue(os.path.isfile(self.write_rawacf_array_file))
        dm2 = pydarnio.BorealisRead(self.write_rawacf_array_file, 'rawacf',
                                    'array')

        new_records = dm2.records  # restructuring happens here
        dictionaries_are_same = self.check_dictionaries_are_same(records,
                                                                 new_records)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_rawacf_array_file)
        del _, dm2, records, new_records

    def test_read_array_write_site_rawacf(self):
        """
        Test reading, restructuring, writing, restructuring rawacf.

        Checks:
            - arrays that pass the read from a file can then be written
                as records
            - arrays restructured, written as records, read as records,
                restructured back to arrays are the same as original arrays
        """
        dm = pydarnio.BorealisRead(self.source_rawacf_array_file, 'rawacf',
                                   'array')
        arrays = dm.arrays

        records = dm.records  # restructuring happens here
        _ = pydarnio.BorealisWrite(self.write_rawacf_site_file, records,
                                   'rawacf', 'site')
        del dm, records
        self.assertTrue(os.path.isfile(self.write_rawacf_site_file))
        dm2 = pydarnio.BorealisRead(self.write_rawacf_site_file, 'rawacf',
                                    'site')

        new_arrays = dm2.arrays  # restructuring happens here
        dictionaries_are_same = self.check_dictionaries_are_same(arrays,
                                                                 new_arrays)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_rawacf_site_file)
        del _, dm2, arrays, new_arrays

    def test_read_write_site_bfiq(self):
        """
        Test reading and then writing site bfiq data to a file.

        Checks:
            - records that pass the read from a file can then be written
            - records written and then read are the same as original
        """
        dm = pydarnio.BorealisRead(self.source_bfiq_site_file, 'bfiq',
                                   'site')
        records = dm.records
        _ = pydarnio.BorealisWrite(self.write_bfiq_site_file,
                                   records, 'bfiq',
                                   'site')
        self.assertTrue(os.path.isfile(self.write_bfiq_site_file))
        dm2 = pydarnio.BorealisRead(self.write_bfiq_site_file, 'bfiq',
                                    'site')
        new_records = dm2.records
        dictionaries_are_same = self.check_dictionaries_are_same(records,
                                                                 new_records)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_bfiq_site_file)
        del _, dm, dm2, records, new_records

    def test_read_write_array_bfiq(self):
        """
        Test reading and then writing array structured bfiq data to a file.

        Checks:
            - arrays that pass the read from a file can then be written
            - arrays written and then read are the same as original
        """
        dm = pydarnio.BorealisRead(self.source_bfiq_array_file, 'bfiq',
                                   'array')
        arrays = dm.arrays
        _ = pydarnio.BorealisWrite(self.write_bfiq_array_file,
                                   arrays, 'bfiq',
                                   'array')
        self.assertTrue(os.path.isfile(self.write_bfiq_array_file))
        dm2 = pydarnio.BorealisRead(self.write_bfiq_array_file, 'bfiq',
                                    'array')
        new_arrays = dm2.arrays
        dictionaries_are_same = self.check_dictionaries_are_same(arrays,
                                                                 new_arrays)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_bfiq_array_file)
        del _, dm, dm2, arrays, new_arrays

    def test_read_site_write_array_bfiq(self):
        """
        Test reading, restructuring, writing, restructuring bfiq.

        Checks:
            - records that pass the read from a file can then be written
                as arrays
            - records restructured, written as arrays, read as arrays,
                restructured back to records are the same as original records
        """
        dm = pydarnio.BorealisRead(self.source_bfiq_site_file, 'bfiq',
                                   'site')
        records = dm.records

        arrays = dm.arrays  # restructuring happens here
        _ = pydarnio.BorealisWrite(self.write_bfiq_array_file,
                                   arrays, 'bfiq',
                                   'array')
        del dm, arrays
        self.assertTrue(os.path.isfile(self.write_bfiq_array_file))
        dm2 = pydarnio.BorealisRead(self.write_bfiq_array_file, 'bfiq',
                                    'array')

        new_records = dm2.records  # restructuring happens here
        dictionaries_are_same = self.check_dictionaries_are_same(records,
                                                                 new_records)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_bfiq_array_file)
        del _, dm2, records, new_records

    def test_read_array_write_site_bfiq(self):
        """
        Test reading, restructuring, writing, restructuring bfiq.

        Checks:
            - arrays that pass the read from a file can then be written
                as records
            - arrays restructured, written as records, read as records,
                restructured back to arrays are the same as original arrays
        """
        dm = pydarnio.BorealisRead(self.source_bfiq_array_file, 'bfiq',
                                   'array')
        arrays = dm.arrays

        records = dm.records  # restructuring happens here
        _ = pydarnio.BorealisWrite(self.write_bfiq_site_file,
                                   records, 'bfiq',
                                   'site')
        del dm, records
        self.assertTrue(os.path.isfile(self.write_bfiq_site_file))
        dm2 = pydarnio.BorealisRead(self.write_bfiq_site_file, 'bfiq',
                                    'site')

        new_arrays = dm2.arrays  # restructuring happens here
        dictionaries_are_same = self.check_dictionaries_are_same(arrays,
                                                                 new_arrays)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_bfiq_site_file)
        del _, dm2, arrays, new_arrays

    def test_read_write_site_antennas_iq(self):
        """
        Test reading and then writing site antennas_iq data to a file.

        Checks:
            - records that pass the read from a file can then be written
            - records written and then read are the same as original
        """
        dm = pydarnio.BorealisRead(self.source_antennas_iq_site_file,
                                   'antennas_iq',
                                   'site')
        records = dm.records
        _ = pydarnio.BorealisWrite(self.write_antennas_iq_site_file,
                                   records, 'antennas_iq',
                                   'site')
        self.assertTrue(os.path.isfile(self.write_antennas_iq_site_file))
        dm2 = pydarnio.BorealisRead(self.write_antennas_iq_site_file,
                                    'antennas_iq',
                                    'site')
        new_records = dm2.records
        dictionaries_are_same = self.check_dictionaries_are_same(records,
                                                                 new_records)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_antennas_iq_site_file)
        del _, dm, dm2, records, new_records

    def test_read_write_array_antennas_iq(self):
        """
        Test reading and then writing array structured
        antennas_iq data to a file.

        Checks:
            - arrays that pass the read from a file can then be written
            - arrays written and then read are the same as original
        """
        dm = pydarnio.BorealisRead(self.source_antennas_iq_array_file,
                                   'antennas_iq',
                                   'array')
        arrays = dm.arrays
        _ = pydarnio.BorealisWrite(self.write_antennas_iq_array_file,
                                   arrays, 'antennas_iq',
                                   'array')
        self.assertTrue(os.path.isfile(self.write_antennas_iq_array_file))
        dm2 = pydarnio.BorealisRead(self.write_antennas_iq_array_file,
                                    'antennas_iq',
                                    'array')
        new_arrays = dm2.arrays
        dictionaries_are_same = self.check_dictionaries_are_same(arrays,
                                                                 new_arrays)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_antennas_iq_array_file)
        del _, dm, dm2, arrays, new_arrays

    def test_read_site_write_array_antennas_iq(self):
        """
        Test reading, restructuring, writing, restructuring antennas_iq.

        Checks:
            - records that pass the read from a file can then be written
                as arrays
            - records restructured, written as arrays, read as arrays,
                restructured back to records are the same as original records
        """
        dm = pydarnio.BorealisRead(self.source_antennas_iq_site_file,
                                   'antennas_iq',
                                   'site')
        records = dm.records

        arrays = dm.arrays  # restructuring happens here
        del dm
        gc.collect()
        writer = pydarnio.BorealisWrite(self.write_antennas_iq_array_file,
                                        arrays, 'antennas_iq',
                                        'array')
        del arrays, writer
        gc.collect()
        self.assertTrue(os.path.isfile(self.write_antennas_iq_array_file))
        dm2 = pydarnio.BorealisRead(self.write_antennas_iq_array_file,
                                    'antennas_iq',
                                    'array')

        new_records = dm2.records  # restructuring happens here
        dictionaries_are_same = self.check_dictionaries_are_same(records,
                                                                 new_records)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_antennas_iq_array_file)
        del dm2, records, new_records

    def test_read_array_write_site_antennas_iq(self):
        """
        Test reading, restructuring, writing, restructuring antennas_iq.

        Checks:
            - arrays that pass the read from a file can then be written
                as records
            - arrays restructured, written as records, read as records,
                restructured back to arrays are the same as original arrays
        """
        dm = pydarnio.BorealisRead(self.source_antennas_iq_array_file,
                                   'antennas_iq',
                                   'array')
        arrays = dm.arrays

        writer = pydarnio.BorealisWrite(self.write_antennas_iq_site_file,
                                        dm.records, 'antennas_iq',
                                        'site')
        del dm, writer
        gc.collect()
        self.assertTrue(os.path.isfile(self.write_antennas_iq_site_file))
        dm2 = pydarnio.BorealisRead(self.write_antennas_iq_site_file,
                                    'antennas_iq',
                                    'site')

        new_arrays = dm2.arrays  # restructuring happens here
        dictionaries_are_same = self.check_dictionaries_are_same(arrays,
                                                                 new_arrays)
        self.assertTrue(dictionaries_are_same)

        os.remove(self.write_antennas_iq_site_file)
        del dm2, arrays, new_arrays


# TODO ADD FAILURE TESTS FOR CONVERT (converting to wrong filetype, etc.)
if __name__ == '__main__':
    """
    Runs the above class in a unittest system.
    """
    pydarnio_logger.info("Starting Borealis unit testing")

    unittest.main()
