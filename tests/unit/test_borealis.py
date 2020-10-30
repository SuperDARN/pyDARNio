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

import logging
import os

import pydarnio

import borealis_utils


class TestBorealisReadSitev04(borealis_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v04 Site data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pydarnio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "site"
        self.version = 0.4

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisReadSitev05(borealis_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v05 Site data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pydarnio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "site"
        self.version = 0.5

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisReadArrayv04(borealis_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v04 array data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pydarnio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "array"
        self.version = 0.4

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisReadArrayv05(borealis_utils.TestReadBorealis):
    """
    Testing class for reading Borealis v05 Array data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pydarnio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "array"
        self.version = 0.5

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version


class TestBorealisWriteSite(borealis_utils.TestWriteBorealis):
    """
    Tests BorealisWrite class
    """

    def setUp(self):
        self.write_func = pydarnio.BorealisWrite
        self.read_func = pydarnio.BorealisRead
        self.data_type = None
        self.data = []
        self.temp_data = []
        self.nrec = 0
        self.temp_file = "not_a_file.acf"
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "site"

    def tearDown(self):
        del self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types, self.file_struct, self.nrec


class TestBorealisWriteArray(borealis_utils.TestWriteBorealis):
    """
    Tests BorealisWrite class
    """

    def setUp(self):
        self.write_func = pydarnio.BorealisWrite
        self.read_func = pydarnio.BorealisRead
        self.data_type = None
        self.data = []
        self.temp_data = []
        self.nrec = 0
        self.temp_file = "not_a_file.acf"
        self.file_types = ["rawacf", "bfiq", "antennas_iq"]
        self.file_struct = "array"

    def tearDown(self):
        del self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types, self.file_struct, self.nrec


class TestBorealisConvertSitev04(borealis_utils.TestConvertBorealis):
    """
    Tests BorealisConvert class for V04 Site data conversion
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.temp_file = "fake.temp"
        self.test_dir = os.path.join("..", "testdir")
        self.file_types = ["rawacf", "bfiq"]
        self.file_struct = "site"
        self.version = 0.4

    def tearDown(self):
        del self.test_file, self.test_dir, self.file_types, self.file_struct
        del self.version, self.temp_file


class TestBorealisConvertArrayv04(borealis_utils.TestConvertBorealis):
    """
    Tests BorealisConvert class for V04 Array data conversion
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.temp_file = "fake.temp"
        self.test_dir = os.path.join("..", "testdir")
        self.file_types = ["rawacf", "bfiq"]
        self.file_struct = "array"
        self.version = 0.4

    def tearDown(self):
        del self.test_file, self.test_dir, self.file_types, self.file_struct
        del self.version, self.temp_file


class TestBorealisConvertSitev05(borealis_utils.TestConvertBorealis):
    """
    Tests BorealisConvert class for V05 Site data conversion
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.temp_file = "fake.temp"
        self.test_dir = os.path.join("..", "testdir")
        self.file_types = ["rawacf", "bfiq"]
        self.file_struct = "site"
        self.version = 0.5

    def tearDown(self):
        del self.test_file, self.test_dir, self.file_types, self.file_struct
        del self.version, self.temp_file


class TestBorealisConvertArrayv05(borealis_utils.TestConvertBorealis):
    """
    Tests BorealisConvert class for V05 Array data conversion
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.temp_file = "fake.temp"
        self.test_dir = os.path.join("..", "testdir")
        self.file_types = ["rawacf", "bfiq"]
        self.file_struct = "array"
        self.version = 0.5

    def tearDown(self):
        del self.test_file, self.test_dir, self.file_types, self.file_struct
        del self.version, self.temp_file
