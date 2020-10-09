# Copyright (C) 2020 NRL
# Author: Angeline Burrell

import bz2
import collections
import copy
from glob import glob
import numpy as np
import os
import unittest

import pydarnio

import dmap_data_sets
import fitacf_data_sets
import grid_data_sets
import iqdat_data_sets
import map_data_sets
import rawacf_data_sets


def get_test_files(test_file_type, test_dir=os.path.join("..", "testfiles")):
    """ Generate a dictionary containing the test filenames

    Parameters
    ----------
    test_file_type : str
        Accepts 'good', 'stream', 'empty', and 'corrupt', along with
        'borealis-vXX_' or 'borealis-vXX-site_' as a prefix to these options
    test_dir : str
        Directory containing the test files
        (default=os.path.join('..', 'testfiles'))

    Returns
    -------
    test_files : list or dict
        Dict of good files with keys pertaining to the file type, a list
        of corrupt files, or a list of stream files

    """
    # Ensure the test file type is lowercase
    test_file_type = test_file_type.lower()

    # See if this is a borealis test
    test_subdir = test_file_type.split('_')
    if len(test_subdir) > 1:
        # Only the last bit specifies test_file_type
        test_file_type = test_subdir[-1]

        # Keep the first part of the specifier intact and extend the
        # test directory
        test_dir = os.path.join(test_dir, '_'.join(test_subdir[:-1]))

    # Get a list of the available test files
    files = glob("ls {:s}".format(os.path.join(test_dir, test_file_type, "*")))

    # Prepare the test files in the necessary output format
    if test_file_type == "good":
        test_files = dict()
        for fname in files:
            # Split the filename by periods to get the SuperDARN file extention
            split_fname = fname.split(".")

            # HDF5 and netCDF versions of these files will have the SuperDARN
            # file type as the second to the last element in the split list
            # and Borealis site files have the SuperDARN file type as the
            # third element
            if split_fname[-1] in ['hd5f', 'h5', 'nc']:
                ext = split_fname[-2]
            elif(split_fname[-1] == 'site'
                 and split_fname[-2] in ['hd5f', 'h5', 'nc']):
                ext = "_".join(["site", split_fname[-3]])
            else:
                ext = split_fname[-1]

            # Save the filename, with keys organizing them by SuperDARN
            # file extension
            test_files[ext] = fname
    else:
        # The files don't need to be organized, just return in a list
        test_files = files

    return test_files


def remove_temp_file(temp_file):
    """ Utility for removing temporary files

    Parameters
    ----------
    temp_file : str
        Name of temporary file

    Returns
    -------
    bool
        True if file was removed, False if it did not exist
    """
    if os.path.isfile(temp_file):
        os.remove(temp_file)
        return True
    else:
        return False


class TestRead(unittest.TestCase):
    """ Testing class for reading classes
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.read_func = None
        self.file_types = ["rawacf", "fitacf", "fit", "iqdat", "grid", "map"]
        self.corrupt_read_type = "rawacf"

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec
        del self.read_func, self.file_types, self.corrupt_read_type

    def load_file_record(self, file_type='', stream=False):
        """ Load a test file data record
        """
        # Load the data with the current test file
        self.data = self.read_func(self.test_file, stream=stream)

        # Read the data
        local_read_func = getattr(self.data, "read_{:s}".format(file_type))
        _ = local_read_func()
        self.rec = self.data.get_dmap_records

    def test_incorrect_filepath(self):
        """
        Test raise FileNotFoundError with bad filename or path
        """
        for val in ["bad_dir", self.test_dir]:
            with self.subTest(val=val):
                # Create a test filename with path
                self.test_file = os.path.join(val, self.test_file)

                # Assert correct error and message for bad filename
                self.assertRaises(FileNotFoundError, self.read_func,
                                  self.test_file)

    def test_empty_file(self):
        """
        Tests raise EmptyFileError with an empty file
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        self.test_file = get_test_files("empty", test_dir=self.test_dir)[0]
        self.assertRaises(pydarnio.dmap_exceptions.EmptyFileError,
                          self.read_func, self.test_file)

    def test_good_open_file(self):
        """
        Test file opening, reading, and converting to a bytearray

        Checks:
            - bytearray instance is created from reading in the file
            - bytearray is not empty
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        test_file_dict = get_test_files("good", test_dir=self.test_dir)
        for val in self.file_types:
            with self.subTest(val=val):
                # Load the file
                self.test_file = test_file_dict[val]
                self.data = self.read_func(self.test_file)

                # Test the file data
                self.assertIsInstance(self.data.dmap_bytearr, bytearray)
                self.assertGreater(self.data.dmap_end_bytes, 0)

    def test_file_integrity(self):
        """
        Tests test_initial_data_integrity to ensure no file corruption
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        test_file_dict = get_test_files("good", test_dir=self.test_dir)
        for val in self.file_types:
            with self.subTest(val=val):
                self.test_file = test_file_dict[val]
                self.data = self.read_func(self.test_file)
                self.data.test_initial_data_integrity()

    def test_corrupt_files(self):
        """
        Test raises a dmap_exceptions Error when readig a corrupt file
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        corrupt_files = get_test_files("corrupt", test_dir=self.test_dir)

        for val in [(corrupt_files[0],
                     pydarnio.dmap_exceptions.DmapDataTypeError),
                    (corrupt_files[1],
                     pydarnio.dmap_exceptions.NegativeByteError)]:
            with self.subTest(val=val):
                self.test_file = val[0]
                with self.assertRaises(val[1]):
                    self.local_file_record(self.corrupt_read_type)

    def test_corrupt_file_integrity(self):
        """
        Test raises a dmap_exceptions when checking integrity of a corrupt file
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        corrupt_files = get_test_files("corrupt", test_dir=self.test_dir)

        for val in [(corrupt_files[0],
                     pydarnio.dmap_exceptions.MismatchByteError),
                    (corrupt_files[1],
                     pydarnio.dmap_exceptions.NegativeByteError)]:
            with self.subTest(val=val):
                self.data = self.read_func(val[0])
                with self.assertRaises(val[1]):
                    self.data.test_initial_data_integrity()

    def test_read_stream(self):
        """
        Test successful read of a stream formed from a bzip2 file

         Checks:
            - returns correct data structures
            - returns expected values
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        # bz2 opens the compressed file into a data
        # stream of bytes without actually uncompressing the file
        self.test_file = get_test_files("stream", test_dir=self.test_dir)[0]
        with bz2.open(self.test_file) as fp:
            self.test_file = fp.read()

        self.local_file_record(self.corrupt_read_type, stream=True)

        # Test the output of the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['channel'], pydarnio.DmapScalar)
        self.assertIsInstance(self.rec[1]['ptab'], pydarnio.DmapArray)
        self.assertIsInstance(self.rec[7]['channel'].value, int)
        self.assertIsInstance(self.rec[2]['xcfd'].value, np.ndarray)
        self.assertEqual(self.rec[0]['xcfd'].dimension, 3)

    def test_dmap_read_corrupt_stream(self):
        """
        Test raises pydmap exception when reading a corrupted stream from
        a compressed file

        Method - Read in a compressed file from a good stream, then insert
        some random bytes to produce a corrupt stream.
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        # Open the data stream
        self.test_file = get_test_files("stream", test_dir=self.test_dir)[0]
        with bz2.open(self.test_file) as fp:
            self.test_file = fp.read()

        # Load and corrupt data, converting to byte array for mutability
        # since bytes are immutable.
        self.data = bytearray(self.test_file[0:36])
        self.data[36:40] = bytearray(str(os.urandom(4)).encode('utf-8'))
        self.data[40:] = self.test_file[37:]

        # Assert data from corrupted stream is corrupted
        with self.assertRaises(pydarnio.dmap_exceptions.DmapDataError):
            self.local_file_record(self.corrupt_read_type, stream=True)


class TestWrite(unittest.TestCase):
    """ Testing class for writing classes
    """
    def setUp(self):
        self.write_class = None
        self.write_func = None
        self.data_type = None
        self.data = []
        self.temp_file = "not_a_file.acf"
        self.file_types = ["rawacf", "fitacf", "dmap", "iqdat", "grid", "map"]

    def tearDown(self):
        remove_temp_file(self.temp_file)
        del self.write_class, self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types

    def load_data_w_filename(self):
        """ Utility for loading data and constructing a temporary filename
        """
        if self.data_type == "rawacf":
            self.data = copy.deepcopy(rawacf_data_sets.rawacf_data)
        elif self.data_type == "fitacf":
            self.data = copy.deepcopy(fitacf_data_sets.fitacf_data)
        elif self.data_type == "iqdat":
            self.data = copy.deepcopy(iqdat_data_sets.iqdat_data)
        elif self.data_type == "grid":
            self.data = copy.deepcopy(grid_data_sets.grid_data)
        elif self.data_type == "map":
            self.data = copy.deepcopy(map_data_sets.map_data)
        elif self.data_type == "dmap":
            self.data = copy.deepcopy(dmap_data_sets.dmap_data)

        self.temp_file = "{:s}_test.{:s}".format(self.data_type,
                                                 self.data_type)

    def set_write_func(self):
        """ Utility to retrieve the writing function
        """
        darn = self.write_class(self.data)
        self.write_func = getattr(darn, "write_{:s}".format(self.data_type))

    def test_darn_write_constructor(self):
        """
        Tests SDarnWrite constructor for different file types

        Expected behaviour
        ------------------
        Contains file name of the data if given to it.
        """
        for val in self.file_types:
            with self.subTest(val=val):
                self.data_type = val
                self.load_data_w_filename()
                darn = self.write_class(self.data, self.temp_file)
                self.assertEqual(darn.filename, self.temp_file)
                self.assertFalse(remove_temp_file(self.temp_file))

    def test_incorrect_filename_input_using_write_methods(self):
        """
        Test raises FilenameRequiredError when no filename is given to write
        """
        for val in self.file_types:
            with self.subTest(val=val):
                self.data_type = val
                self.load_data_w_filename()

                # Attempt to write data without a filename
                self.set_write_func()
                with self.assertRaises(
                        pydarnio.dmap_exceptions.FilenameRequiredError):
                    self.write_func()

    def test_empty_record(self):
        """
        Test raises DmapDataError if an empty record is given
        """
        with self.assertRaises(pydarnio.dmap_exceptions.DmapDataError):
            self.set_write_func()
            self.write_func(self.temp_file)

    def test_writing_success(self):
        """
        Test successful file writing and removal of temporary file
        """
        for val in self.file_types:
            with self.subTest(val=val):
                self.data_type = val
                self.load_data_w_filename()
                self.set_write_func()

                # Only testing the file is created since it should only be
                # created at the last step after all checks have passed.
                # Testing the integrity of the insides of the file will be part
                # of integration testing since we need SDarnRead for that.
                self.write_func(self.temp_file)
                self.assertTrue(remove_temp_file(self.temp_file))
