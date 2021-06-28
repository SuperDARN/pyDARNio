# Copyright (C) 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marina Schmidt, Angeline Burrell

import collections
import logging
import numpy as np
import os

import pydarnio
from pydarnio.tests.utils import file_utils

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
        self.read_func = pydarnio.DmapRead
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
        test_file_dict = file_utils.get_test_files("good",
                                                   test_dir=self.test_dir)
        self.test_file = test_file_dict['fitacf']
        self.load_file_record(file_type='records')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['bmnum'], pydarnio.DmapScalar)
        self.assertIsInstance(self.rec[1]['ptab'], pydarnio.DmapArray)
        self.assertIsInstance(self.rec[7]['channel'].value, int)
        self.assertIsInstance(self.rec[2]['ltab'].value, np.ndarray)
        self.assertEqual(self.rec[0]['ptab'].dimension, 1)
        self.assertEqual(self.rec[50]['gflg'].value[1], 0)


class TestDmapWrite(file_utils.TestWrite):
    """ Testing DmapWrite class"""
    def setUp(self):
        self.write_class = pydarnio.DmapWrite
        self.write_func = None
        self.data_type = "dmap"
        self.data = []
        self.temp_file = "not_a_file.acf"
        self.file_types = ["dmap"]

    def tearDown(self):
        del self.write_class, self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types

    def test_bad_scalar_to_bytes(self):
        """
        Test raises DmapCharError when attempting to write char instead of int8
        """
        self.data = [{'channel': pydarnio.DmapScalar('channel', 'c', 1, 'c')}]
        darn = self.write_class(self.data)
        with self.assertRaises(pydarnio.dmap_exceptions.DmapCharError):
            darn.dmap_scalar_to_bytes(self.data[0]['channel'])

    def test_bad_array_to_byes(self):
        """
        Test raises appropriate Dmap Error when writing unsupported array types
        """

        self.data = [{'xcf': pydarnio.DmapArray('xcf', np.array(['dog', 'cat',
                                                                 'rat']),
                                                9, 's', 1, [3])},
                     {'channel': pydarnio.DmapArray('channel',
                                                    np.array(['d', 'c', 'r']),
                                                    1, 'c', 1, [3])}]
        errors = [pyDARNio.dmap_exceptions.DmapDataError,
                  pyDARNio.dmap_exceptions.DmapCharError]
        for i, val in enumerate(errors):
            with self.subTest(val=val):
                array = [dat_val for dat_val in self.data[i].values()][0]
                darn = self.write_class([self.data[i]])
                with self.assertRaises(val):
                    darn.dmap_array_to_bytes(array)
