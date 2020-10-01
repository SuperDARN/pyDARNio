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


class TestBorealisWriteSite(file_utils.TestWriteBorealis):
    """
    Tests BorealisWrite class
    """

    def setUp(self):
        self.write_func = pyDARNio.BorealisWrite
        self.read_func = pyDARNio.BorealisRead
        self.data_type = None
        self.data = []
        self.temp_data = []
        self.nrec = 0
        self.temp_file = "not_a_file.acf"
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "site"

    def tearDown(self):
        self.remove_temp_file()
        del self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types, self.file_struct, self.nrec


class TestBorealisWriteArray(file_utils.TestWriteBorealis):
    """
    Tests BorealisWrite class
    """

    def setUp(self):
        self.write_func = pyDARNio.BorealisWrite
        self.read_func = pyDARNio.BorealisRead
        self.data_type = None
        self.data = []
        self.temp_data = []
        self.nrec = 0
        self.temp_file = "not_a_file.acf"
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "array"

    def tearDown(self):
        self.remove_temp_file()
        del self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types, self.file_struct, self.nrec


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
