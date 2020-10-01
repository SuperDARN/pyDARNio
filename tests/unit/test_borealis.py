# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller, Angeline Burrell
"""
This test suite is to test the implementation for the following classes:
    BorealisRead
    BorealisWrite
    BorealisConvert
Support for the following Borealis file types:
    rawrf
    antennas_iq
    bfiq
    rawacf
And supports conversion of the following Borealis -> SDARN DMap types:
    bfiq -> iqdat
    rawacf -> rawacf
"""

from collections import OrderedDict
import copy
import logging
import numpy as np
import os
import unittest

import pyDARNio

import file_utils
from borealis_rawacf_data_sets import (borealis_array_rawacf_data,
                                       borealis_site_rawacf_data)
from borealis_bfiq_data_sets import (borealis_array_bfiq_data,
                                     borealis_site_bfiq_data)

pyDARNio_logger = logging.getLogger('pyDARNio')


class TestBorealisReadSitev04(file_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v04 Site data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pyDARNio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "site"
        self.version = 4

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisReadSitev05(file_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v05 Site data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pyDARNio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "site"
        self.version = 5

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisReadArrayv04(file_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v04 array data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pyDARNio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "array"
        self.version = 4

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisReadArrayv05(file_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v05 Array data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pyDARNio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "array"
        self.version = 5

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


@unittest.skip("Not Re-written Yet")
class TestBorealisWrite(unittest.TestCase):
    """
    Tests BorealisWrite class
    """

    def setUp(self):
        self.rawacf_site_data = copy.deepcopy(
            borealis_site_rawacf_data)
        self.rawacf_site_missing_field = copy.deepcopy(
            borealis_site_rawacf_data)
        self.rawacf_site_extra_field = copy.deepcopy(
            borealis_site_rawacf_data)
        self.rawacf_site_incorrect_fmt = copy.deepcopy(
            borealis_site_rawacf_data)
        self.bfiq_site_data = copy.deepcopy(
            borealis_site_bfiq_data)
        self.bfiq_site_missing_field = copy.deepcopy(
            borealis_site_bfiq_data)
        self.bfiq_site_extra_field = copy.deepcopy(
            borealis_site_bfiq_data)
        self.bfiq_site_incorrect_fmt = copy.deepcopy(
            borealis_site_bfiq_data)

        self.rawacf_array_data = copy.deepcopy(
            borealis_array_rawacf_data)
        self.rawacf_array_missing_field = copy.deepcopy(
            borealis_array_rawacf_data)
        self.rawacf_array_extra_field = copy.deepcopy(
            borealis_array_rawacf_data)
        self.rawacf_array_incorrect_fmt = copy.deepcopy(
            borealis_array_rawacf_data)
        self.bfiq_array_data = copy.deepcopy(
            borealis_array_bfiq_data)
        self.bfiq_array_missing_field = copy.deepcopy(
            borealis_array_bfiq_data)
        self.bfiq_array_extra_field = copy.deepcopy(
            borealis_array_bfiq_data)
        self.bfiq_array_incorrect_fmt = copy.deepcopy(
            borealis_array_bfiq_data)

    # Read/write tests to check input vs output
    def check_dictionaries_are_same(self, dict1, dict2):

        self.assertEqual(sorted(list(dict1.keys())),
                         sorted(list(dict2.keys())))
        for key1, value1 in dict1.items():
            if isinstance(value1, dict) or isinstance(value1, OrderedDict):
                self.check_dictionaries_are_same(value1, dict2[key1])
            elif isinstance(value1, np.ndarray):
                self.assertTrue((value1 == dict2[key1]).all())
            else:
                self.assertEqual(value1, dict2[key1])

        return True

    def test_writing_site_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf file

        Expected behaviour
        ------------------
        Rawacf file is produced
        """
        test_file = "./test_rawacf.rawacf.hdf5"
        pyDARNio.BorealisWrite(test_file, self.rawacf_site_data,
                             'rawacf', 'site')
        # only testing the file is created since it should only be created
        # at the last step after all checks have passed
        # Testing the integrity of the insides of the file will be part of
        # integration testing since we need BorealisSiteRead for that.
        self.assertTrue(os.path.isfile(test_file))
        reader = pyDARNio.BorealisRead(test_file, 'rawacf', 'site')
        records = reader.records
        dictionaries_are_same =\
            self.check_dictionaries_are_same(records, self.rawacf_site_data)
        self.assertTrue(dictionaries_are_same)
        os.remove(test_file)

    def test_missing_field_site_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisFieldMissingError - because the rawacf data is
        missing field num_sequences
        """

        keys = sorted(list(self.rawacf_site_missing_field.keys()))
        del self.rawacf_site_missing_field[keys[0]]['num_sequences']

        try:
            pyDARNio.BorealisWrite("test_rawacf.rawacf.hdf5",
                                 self.rawacf_site_missing_field,
                                 'rawacf', 'site')
        except pyDARNio.borealis_exceptions.BorealisFieldMissingError as err:
            self.assertEqual(err.fields, {'num_sequences'})
            self.assertEqual(err.record_name, keys[0])

    def test_extra_field_site_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisExtraFieldError because the rawacf data
        has an extra field dummy
        """
        keys = sorted(list(self.rawacf_site_extra_field.keys()))
        self.rawacf_site_extra_field[keys[0]]['dummy'] = 'dummy'
        try:
            pyDARNio.BorealisWrite("test_rawacf.rawacf.hdf5",
                                 self.rawacf_site_extra_field,
                                 'rawacf', 'site')
        except pyDARNio.borealis_exceptions.BorealisExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_name, keys[0])

    def test_incorrect_data_format_site_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises BorealisDataFormatTypeError because the rawacf data has the
        wrong type for the scan_start_marker field
        """
        keys = sorted(list(self.rawacf_site_incorrect_fmt.keys()))
        self.rawacf_site_incorrect_fmt[keys[0]]['scan_start_marker'] = 1

        try:
            pyDARNio.BorealisWrite("test_rawacf.rawacf.hdf5",
                                 self.rawacf_site_incorrect_fmt,
                                 'rawacf', 'site')
        except pyDARNio.borealis_exceptions.BorealisDataFormatTypeError as err:
            self.assertEqual(
                err.incorrect_types['scan_start_marker'],
                "<class 'numpy.bool_'>")
            self.assertEqual(err.record_name, keys[0])

    def test_writing_site_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq file

        Expected behaviour
        ------------------
        bfiq file is produced
        """
        test_file = "./test_bfiq.bfiq.hdf5"
        pyDARNio.BorealisWrite(test_file, self.bfiq_site_data, 'bfiq', 'site')
        # only testing the file is created since it should only be created
        # at the last step after all checks have passed
        # Testing the integrity of the insides of the file will be part of
        # integration testing since we need BorealisSiteRead for that.
        self.assertTrue(os.path.isfile(test_file))
        reader = pyDARNio.BorealisRead(test_file, 'bfiq', 'site')
        records = reader.records
        dictionaries_are_same = \
            self.check_dictionaries_are_same(records, self.bfiq_site_data)
        self.assertTrue(dictionaries_are_same)
        os.remove(test_file)

    def test_missing_field_site_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisFieldMissingError - because the bfiq data is
        missing field antenna_arrays_order
        """
        keys = sorted(list(self.bfiq_site_missing_field.keys()))
        del self.bfiq_site_missing_field[keys[0]]['antenna_arrays_order']

        try:
            _ = pyDARNio.BorealisWrite("test_bfiq.bfiq.hdf5",
                                     self.bfiq_site_missing_field,
                                     'bfiq', 'site')
        except pyDARNio.borealis_exceptions.BorealisFieldMissingError as err:
            self.assertEqual(err.fields, {'antenna_arrays_order'})
            self.assertEqual(err.record_name, keys[0])

    def test_extra_field_site_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisExtraFieldError because the bfiq data
        has an extra field dummy
        """
        keys = sorted(list(self.bfiq_site_extra_field.keys()))
        self.bfiq_site_extra_field[keys[0]]['dummy'] = 'dummy'

        try:
            _ = pyDARNio.BorealisWrite("test_bfiq.bfiq.hdf5",
                                     self.bfiq_site_extra_field,
                                     'bfiq', 'site')
        except pyDARNio.borealis_exceptions.BorealisExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_name, keys[0])

    def test_incorrect_data_format_site_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises BorealisDataFormatTypeError because the bfiq data has the
        wrong type for the first_range_rtt field
        """
        keys = sorted(list(self.bfiq_site_incorrect_fmt.keys()))
        self.bfiq_site_incorrect_fmt[keys[0]]['first_range_rtt'] = 5

        try:
            _ = pyDARNio.BorealisWrite("test_bfiq.bfiq.hdf5",
                                     self.bfiq_site_incorrect_fmt,
                                     'bfiq', 'site')
        except pyDARNio.borealis_exceptions.BorealisDataFormatTypeError as err:
            self.assertEqual(
                err.incorrect_types['first_range_rtt'],
                "<class 'numpy.float32'>")
            self.assertEqual(err.record_name, keys[0])

    def test_writing_array_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf file

        Expected behaviour
        ------------------
        Rawacf file is produced
        """
        test_file = "test_rawacf.rawacf.hdf5"
        _ = pyDARNio.BorealisWrite(test_file,
                                 self.rawacf_array_data, 'rawacf',
                                 'array')
        self.assertTrue(os.path.isfile(test_file))
        reader = pyDARNio.BorealisRead(test_file, 'rawacf', 'array')
        data = reader.arrays
        dictionaries_are_same =\
            self.check_dictionaries_are_same(data, self.rawacf_array_data)
        self.assertTrue(dictionaries_are_same)
        os.remove(test_file)

    def test_missing_field_array_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisFieldMissingError - because the rawacf data is
        missing field num_sequences
        """

        del self.rawacf_array_missing_field['num_sequences']

        try:
            _ = pyDARNio.BorealisWrite("test_rawacf.rawacf.hdf5",
                                     self.rawacf_array_missing_field,
                                     'rawacf', 'array')
        except pyDARNio.borealis_exceptions.BorealisFieldMissingError as err:
            self.assertEqual(err.fields, {'num_sequences'})

    def test_extra_field_array_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisExtraFieldError because the rawacf data
        has an extra field dummy
        """
        self.rawacf_array_extra_field['dummy'] = 'dummy'
        try:
            _ = pyDARNio.BorealisWrite("test_rawacf.rawacf.hdf5",
                                     self.rawacf_array_extra_field,
                                     'rawacf', 'array')
        except pyDARNio.borealis_exceptions.BorealisExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})

    def test_incorrect_data_format_array_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises BorealisDataFormatTypeError because the rawacf data has the
        wrong type for the scan_start_marker field
        """
        num_records =\
            self.rawacf_array_incorrect_fmt['scan_start_marker'].shape[0]
        self.rawacf_array_incorrect_fmt['scan_start_marker'] = \
            np.array([1] * num_records)

        try:
            _ = pyDARNio.BorealisWrite("test_rawacf.rawacf.hdf5",
                                     self.rawacf_array_incorrect_fmt,
                                     'rawacf', 'array')
        except pyDARNio.borealis_exceptions.BorealisDataFormatTypeError as err:
            self.assertEqual(err.incorrect_types['scan_start_marker'],
                             "np.ndarray of <class 'numpy.bool_'>")

    def test_writing_array_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq file

        Expected behaviour
        ------------------
        bfiq file is produced
        """
        test_file = "test_bfiq.bfiq.hdf5"
        _ = pyDARNio.BorealisWrite(test_file, self.bfiq_array_data,
                                 'bfiq', 'array')
        self.assertTrue(os.path.isfile(test_file))
        reader = pyDARNio.BorealisRead(test_file, 'bfiq', 'array')
        data = reader.arrays
        dictionaries_are_same = \
            self.check_dictionaries_are_same(data,
                                             self.bfiq_array_data)
        self.assertTrue(dictionaries_are_same)
        os.remove(test_file)

    def test_missing_field_array_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisFieldMissingError - because the bfiq data is
        missing field antenna_arrays_order
        """
        del self.bfiq_array_missing_field['antenna_arrays_order']

        try:
            _ = pyDARNio.BorealisWrite("test_bfiq.bfiq.hdf5",
                                     self.bfiq_array_missing_field,
                                     'bfiq', 'array')
        except pyDARNio.borealis_exceptions.BorealisFieldMissingError as err:
            self.assertEqual(err.fields, {'antenna_arrays_order'})

    def test_extra_field_array_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq structure file for the
        given data

        Expected behaviour
        ------------------
        Raises BorealisExtraFieldError because the bfiq data
        has an extra field dummy
        """
        self.bfiq_array_extra_field['dummy'] = 'dummy'

        try:
            _ = pyDARNio.BorealisWrite("test_bfiq.bfiq.hdf5",
                                     self.bfiq_array_extra_field,
                                     'bfiq', 'array')
        except pyDARNio.borealis_exceptions.BorealisExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})

    def test_incorrect_data_format_array_bfiq(self):
        """
        Tests write_bfiq method - writes a bfiq structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises BorealisDataFormatTypeError because the bfiq data has the
        wrong type for the first_range_rtt field
        """
        self.bfiq_array_incorrect_fmt['first_range_rtt'] = 5

        try:
            _ = pyDARNio.BorealisWrite("test_bfiq.bfiq.hdf5",
                                     self.bfiq_array_incorrect_fmt,
                                     'bfiq', 'array')
        except pyDARNio.borealis_exceptions.BorealisDataFormatTypeError as err:
            self.assertEqual(
                err.incorrect_types['first_range_rtt'],
                "<class 'numpy.float32'>")

    # # WRITE FAILURE TESTS

    def test_wrong_borealis_filetype(self):
        """
        Provide the wrong filetype.
        """
        wrong_filetype_exceptions = \
            (pyDARNio.borealis_exceptions.BorealisExtraFieldError,
             pyDARNio.borealis_exceptions.BorealisFieldMissingError,
             pyDARNio.borealis_exceptions.BorealisDataFormatTypeError)
        self.assertRaises(wrong_filetype_exceptions, pyDARNio.BorealisWrite,
                          'test_write_borealis_file.bfiq.hdf5',
                          self.bfiq_site_data, 'antennas_iq', 'site')

    def test_wrong_borealis_file_structure(self):
        """
        Provide the wrong file structure.
        """
        self.assertRaises(pyDARNio.borealis_exceptions.BorealisStructureError,
                          pyDARNio.BorealisWrite,
                          'test_write_borealis_file.bfiq.hdf5',
                          self.bfiq_site_data,  'rawacf', 'array')
        self.assertRaises(pyDARNio.borealis_exceptions.BorealisStructureError,
                          pyDARNio.BorealisWrite,
                          'test_write_borealis_file.bfiq.hdf5',
                          self.bfiq_array_data, 'rawacf', 'site')


@unittest.skip("Not Re-written Yet")
class TestBorealisConvert(unittest.TestCase):
    """
    Tests BorealisConvert class
    """

    def setUp(self):
        self.rawacf_array_data = copy.deepcopy(borealis_array_rawacf_data)
        self.bfiq_array_data = copy.deepcopy(borealis_array_bfiq_data)

        # write some v0.4 data
        self.bfiq_test_file = "test_bfiq.bfiq.hdf5"
        _ = pyDARNio.BorealisWrite(self.bfiq_test_file,
                                 self.bfiq_array_data, 'bfiq', 'array')
        self.rawacf_test_file = "test_rawacf.rawacf.hdf5"
        _ = pyDARNio.BorealisWrite(self.rawacf_test_file,
                                 self.rawacf_array_data,
                                 'rawacf', 'array')

        # get v0.5 data from file
        self.bfiqv05_test_file = borealis_site_bfiq_file_v05
        self.rawacfv05_test_file = borealis_site_rawacf_file_v05

    def test_borealis_convert_to_rawacfv04(self):
        """
        Tests BorealisConvert to rawacf

        Expected behaviour
        ------------------
        write a SDARN DMap rawacf
        """
        _ = pyDARNio.BorealisConvert(self.rawacf_test_file, "rawacf",
                                   "test_rawacf.rawacf.dmap",
                                   borealis_slice_id=0,
                                   borealis_file_structure='array')
        self.assertTrue(os.path.isfile("test_rawacf.rawacf.dmap"))
        os.remove("test_rawacf.rawacf.dmap")

    def test_borealis_convert_to_iqdatv04(self):
        """
        Tests BorealisConvert to iqdat

        Expected behaviour
        ------------------
        write a SDARN DMap iqdat
        """

        _ = pyDARNio.BorealisConvert(self.bfiq_test_file, "bfiq",
                                   "test_bfiq.bfiq.dmap",
                                   borealis_slice_id=0,
                                   borealis_file_structure='array')
        self.assertTrue(os.path.isfile("test_bfiq.bfiq.dmap"))
        os.remove("test_bfiq.bfiq.dmap")

    def test_borealis_convert_to_rawacfv05(self):
        _ = pyDARNio.BorealisConvert(self.rawacfv05_test_file, "rawacf",
                                   "test_rawacf.rawacf.dmap",
                                   borealis_file_structure='site')
        self.assertTrue(os.path.isfile("test_rawacf.rawacf.dmap"))
        os.remove("test_rawacf.rawacf.dmap")

    def test_borealis_convert_to_iqdatv05(self):
        """
        Tests BorealisConvert to iqdat

        Expected behaviour
        ------------------
        write a SDARN DMap iqdat
        """
        _ = pyDARNio.BorealisConvert(self.bfiqv05_test_file, "bfiq",
                                   "test_bfiq.bfiq.dmap",
                                   borealis_file_structure='site')
        self.assertTrue(os.path.isfile("test_bfiq.bfiq.dmap"))
        os.remove("test_bfiq.bfiq.dmap")
