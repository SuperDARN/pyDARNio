# Copyright (C) 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marina Schmidt, Angeline Burrell

import bz2
import collections
import copy
import logging
import numpy as np
import os
import unittest

import pyDARNio

import dmap_data_sets
import file_utils

pyDARNio_logger = logging.getLogger('pyDARNio')


class TestDmapRead(file_utils.TestRead):
    """
    Testing class for DmapRead class
    """
    def setUp(self):
        self.test_file = "somefile.rawacf"
        self.test_dir = os.path.join("..", "testfiles")
        self.data = None
        self.rec = None
        self.read_func = pyDARNio.DmapRead
        self.file_types = ["rawacf", "fitacf"]
        self.corrupt_read_type = "records"

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec
        del self.read_func, self.file_types, self.corrupt_read_type

    def test_read_dmap_file(self):
        """
        Tests DmapRead test read_dmap.

        Behaviour: raising no exceptions
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        # Load the data and read in the first record
        test_file_dict = get_test_files("good", test_dir=self.test_dir)
        self.test_file = test_file_dict['fitacf']
        self.load_file_record(file_type='records')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['bmnum'], pyDARNio.DmapScalar)
        self.assertIsInstance(self.rec[1]['ptab'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[7]['channel'].value, int)
        self.assertIsInstance(self.rec[2]['ltab'].value, np.ndarray)
        self.assertEqual(self.rec[0]['ptab'].dimension, 1)
        self.assertEqual(self.rec[50]['gflg'].value[1], 0)



@unittest.skip('skipping for no reason')
class TestDmapWrite(unittest.TestCase):
    """ Testing DmapWrite class"""
    def setUp(self):
        pass

    def test_incorrect_filename_input_using_write_methods(self):
        """
        Testing if a filename is not given to DmapWrite

        Expected behaviour
        ------------------
        Raises FilenameRequiredError - no filename was given to write and
        constructor
        """
        rawacf_data = copy.deepcopy(dmap_data_sets.dmap_data)
        dmap_data = pyDARNio.DmapWrite(rawacf_data)
        with self.assertRaises(pyDARNio.dmap_exceptions.FilenameRequiredError):
            dmap_data.write_dmap()

    def test_empty_data_check(self):
        """
        Testing if no data is given to DmapWrite

        Expected behaviour
        ------------------
        Raise DmapDataError - no data is given to write
        """
        with self.assertRaises(pyDARNio.dmap_exceptions.DmapDataError):
            dmap_write = pyDARNio.DmapWrite(filename="test.test")
            dmap_write.write_dmap()

    def test_writing_dmap(self):
        """
        Testing write_dmap method

        Expected behaviour
        ------------------
        File is produced
        """
        dmap_data = copy.deepcopy(dmap_data_sets.dmap_data)
        dmap = pyDARNio.DmapWrite(dmap_data)

        # Integration testing will test the integrity of the
        # writing procedure.
        dmap.write_dmap("test_dmap.dmap")
        self.assertTrue(os.path.isfile("test_dmap.dmap"))

        os.remove("test_dmap.dmap")

    def test_scalar(self):
        """
        Test DmapWrite writing a character scalar type.

        Behaviour: Raised DmapCharError
        Dmap cannot write characters as they are treated as strings and not
        int8 - RST standard for char types.
        """
        scalar = pyDARNio.DmapScalar('channel', 'c', 1, 'c')
        dmap_write = pyDARNio.DmapWrite([{'channel': scalar}])
        with self.assertRaises(pyDARNio.dmap_exceptions.DmapCharError):
            dmap_write.dmap_scalar_to_bytes(scalar)

    def test_String_array(self):
        """
        Test DmapWrite writing string arrays

        Behaviour: Raised DmapDataError
        DmapWrite doesn't support writing string arrays because DmapRead does
        not support string arrays.
        """
        array = pyDARNio.DmapArray('xcf', np.array(['dog', 'cat', 'mouse']),
                                 9, 's', 1, [3])
        dmap_write = pyDARNio.DmapWrite([{'xcf': array}])
        with self.assertRaises(pyDARNio.dmap_exceptions.DmapDataError):
            dmap_write.dmap_array_to_bytes(array)

    def test_character_array(self):
        """
        Test DmapWrite writing character arrays.

        Behaviour: Raised DmapCharError
        """
        array = pyDARNio.DmapArray('channel', np.array(['d', 'c', 'm']),
                                 1, 'c', 1, [3])
        dmap_write = pyDARNio.DmapWrite([{'channel': array}])
        with self.assertRaises(pyDARNio.dmap_exceptions.DmapCharError):
            dmap_write.dmap_array_to_bytes(array)
