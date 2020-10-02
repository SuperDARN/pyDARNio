# Copyright (C) 2020 NRL
# Author: Angeline Burrell

from collections import OrderedDict
import copy
import numpy as np
import os
import tables
import unittest

import pyDARNio
import pyDARNio.exceptions.borealis_exceptions as bor_exc

from file_utils import get_test_files, remove_temp_file
import borealis_rawacf_data_sets as borealis_rawacf
import borealis_bfiq_data_sets as borealis_bfiq
import borealis_antennas_iq_data_sets as borealis_antennas_iq


def get_borealis_type(file_type, file_struct, version):
    """ Helper function to build input needed for get_test_files

    Parameters
    ----------
    file_type : str
        Standard file_type input for get_test_files
    file_struct : str
        Borealis file structure (accepts 'array' or 'site')
    version : int
        Borealis version number

    Returns
    -------
    borealis_type : str
        Borealis-style input for get_test_files
    """
    borealis_type = "borealis-v{:02d}{:s}_{:s}".format(
        version, "" if file_struct == "array" else file_struct, file_type)
    return borealis_type


class TestReadBorealis(unittest.TestCase):
    """
    Testing class for reading Borealis data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.test_dir = os.path.join("..", "testdir")
        self.data = None
        self.rec = None
        self.arr = None
        self.read_func = pyDARNio.BorealisRead
        self.file_types = ["rawacf", "bfiq", "antennas_iq", "rawrf"]
        self.file_struct = "site"
        self.version = 4

    def tearDown(self):
        del self.test_file, self.test_dir, self.data, self.rec, self.arr
        del self.read_func, self.file_types, self.file_struct, self.version

    def load_file_record(self, file_type=''):
        """ Load a test file data record or array

        Parameters
        ----------
        file_type : str
            One of self.file_types
        """
        # Load the data with the current test file
        self.data = self.read_func(self.test_file, file_type, self.file_struct)

        # Read the data
        self.rec = self.data.records
        self.arr = self.data.arrays

    def test_return_reader(self):
        """
        Test ability of return_reader function to determin the file structure
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        test_file_dict = get_test_files(get_borealis_type(
            "good", self.file_struct, self.version), test_dir=self.test_dir)

        for val in self.file_types:
            with self.subTest(val=val):
                self.test_file = test_file_dict[val]
                self.load_file_record(val)

                # Site information and data arrays require different commands
                if self.file_struct == "site":
                    dkey = [rkey for rkey in self.rec.keys()][0]
                    self.assertIsInstance(self.rec[dkey]['num_slices'],
                                          np.int64)
                else:
                    self.assertIsInstance(self.arr['num_slices'], np.ndarray)

    def test_incorrect_filepath(self):
        """
        Test raise OSError with bad filename or path
        """
        for val in ["bad_dir", self.test_dir]:
            with self.subTest(val=val):
                # Create a test filename with path
                self.test_file = os.path.join(val, self.test_file)

                # Assert correct error and message for bad filename
                self.assertRaises(OSError, self.read_func,
                                  self.test_file, self.file_types[0],
                                  self.file_struct)

    def test_empty_file(self):
        """
        Tests raise OSError or HDF5ExtError with an empty file
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        self.test_file = get_test_files("empty", test_dir=self.test_dir)[0]
        self.assertRaises((OSError, tables.exceptions.HDF5ExtError),
                          self.read_func, self.test_file, self.file_types[0],
                          self.file_struct)

    def test_wrong_borealis_filetype(self):
        """
        Test raises Borealis Error when specifying the wrong filetype.
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        wrong_filetype_exceptions = (bor_exc.BorealisExtraFieldError,
                                     bor_exc.BorealisFieldMissingError,
                                     bor_exc.BorealisDataFormatTypeError)
        test_file_dict = get_test_files(get_borealis_type(
            "good", self.file_struct, self.version), test_dir=self.test_dir)

        for i, val in enumerate(self.file_types):
            with self.subTest(val=val):
                # Use a file that is not of the current file type
                self.test_file = test_file_dict[self.file_types[i - 1]]

                # Load the file, specifying the current file type
                with self.assertRaises(wrong_filetype_exceptions):
                    self.load_file_record(val)

    def test_wrong_borealis_file_structure(self):
        """
        Test raises BorealisStructureError when specifying wrong file structure
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        # Load the good files with the current file structure
        test_file_dict = get_test_files(get_borealis_type(
            "good", self.file_struct, self.version), test_dir=self.test_dir)

        # Change the file structure
        if self.file_struct == "site":
            self.file_struct = "array"
        else:
            self.file_struct = "site"

        # Cycle through the different file types
        for val in self.file_types:
            with self.subTest(val=val):
                self.test_file = test_file_dict[val]

                # Attempt to load the test file with the wrong structure
                with self.assertRaises(bor_exc.BorealisStructureError):
                    self.load_file_record(val)

    def test_read_good_data(self):
        """ Test successful reading of Borealis data
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        test_file_dict = get_test_files(get_borealis_type(
            "good", self.file_struct, self.version), test_dir=self.test_dir)

        for val in self.file_types:
            with self.subTest(val=val):
                # Set and load the test file data
                self.test_file = test_file_dict[val]
                self.load_file_record(val)

                # Test the first data record
                first_record = self.rec[self.data.record_names[0]]
                self.assertIsInstance(self.rec, OrderedDict)
                self.assertIsInstance(first_record, dict)
                self.assertIsInstance(first_record['num_slices'], np.int64)

                # Tset the first data array
                self.assertIsInstance(self.arr, dict)
                self.assertIsInstance(self.arr['num_slices'], np.ndarray)
                self.assertIsInstance(self.arr['num_slices'][0], np.int64)


class TestWriteBorealis(unittest.TestCase):
    """ Testing class for writing classes
    """
    def setUp(self):
        self.write_func = None
        self.read_func = None
        self.data_type = None
        self.data = []
        self.temp_data = []
        self.nrec = 0
        self.temp_file = "not_a_file.acf"
        self.file_types = ["rawacf", "bfiq", "antennas_iq", "rawrf"]
        self.file_struct = "site"

    def tearDown(self):
        del self.write_func, self.data_type, self.data
        del self.temp_file, self.file_types, self.file_struct, self.nrec

    def load_data_w_filename(self):
        """ Utility for loading data and constructing a temporary filename
        """
        if self.data_type == "bfiq":
            self.nrec = borealis_bfiq.num_records
            if self.file_struct == "site":
                self.data = copy.deepcopy(
                    borealis_bfiq.borealis_site_bfiq_data)
            else:
                self.data = copy.deepcopy(
                    borealis_bfiq.borealis_array_bfiq_data)
        elif self.data_type == "antennas_iq":
            self.nrec = borealis_antennas_iq.num_records
            if self.file_struct == "site":
                self.data = copy.deepcopy(
                    borealis_antennas_iq.borealis_site_antennas_iq_data)
            else:
                self.data = copy.deepcopy(
                    borealis_antennas_iq.borealis_array_antennas_iq_data)
        elif self.data_type == "rawacf":
            self.nrec = borealis_rawacf.num_records
            if self.file_struct == "site":
                self.data = copy.deepcopy(
                    borealis_rawacf.borealis_site_rawacf_data)
            else:
                self.data = copy.deepcopy(
                    borealis_rawacf.borealis_array_rawacf_data)

        self.temp_file = "{:s}_{:s}_test.{:s}.hdf5".format(
            self.data_type, self.file_struct, self.data_type)

    def load_temp_file(self, file_type=''):
        """ Load a test file data record or array

        Parameters
        ----------
        file_type : str
            One of self.file_types
        """
        # Load the data with the current test file
        self.temp_data = self.read_func(self.temp_file, self.data_type,
                                        self.file_struct)

    def test_writing_success(self):
        """
        Tests Borealis file writing and reading
        """
        for val in self.file_types:
            with self.subTest(val=val):
                # Load the sample data
                self.data_type = val
                self.load_data_w_filename()

                # Write the temporary file
                self.write_func(self.temp_file, self.data, val,
                                self.file_struct)

                # Read the temporary file
                self.load_temp_file(file_type=val)

                # Test that the data sets are the same
                self.assertListEqual(
                    sorted([dkey for dkey in self.data.keys()]),
                    sorted([dkey for dkey in self.temp_data.keys()]))

                for dkey, temp_val in self.temp_data.items():
                    if isinstance(temp_val, dict):
                        self.assertDictEqual(temp_val, self.data[dkey])
                    elif isinstance(temp_val, OrderedDict):
                        self.assertDictEqual(dict(temp_val),
                                             dict(self.data[dkey]))
                    elif isinstance(temp_val, np.ndarray):
                        self.assertTrue((temp_val == self.data[dkey]).all())
                    else:
                        self.assertEqual(temp_val, self.data[dkey])

                # Remove the temporary file
                self.assertTrue(remove_temp_file(self.temp_file))

    def test_missing_field(self):
        """
        Test raises BorealisFieldMissingError when missing required field
        """
        missing_field = 'num_slices'

        for val in self.file_types:
            with self.subTest(val=val):
                # Load the sample data
                self.data_type = val
                self.load_data_w_filename()

                # Remove a required field
                dkeys = [dkey for dkey in self.data.keys()]
                del self.data[dkeys[0]][missing_field]

                # Test raises appropriate error
                with self.assertRaises(
                        bor_exc.BorealisFieldMissingError) as err:
                    self.write_func(self.temp_file, self.data, val,
                                    self.file_struct)

                self.assertEqual(err.fields, {missing_field})

                # Remove the temporary file, if it was created
                remove_temp_file(self.temp_file)

    def test_extra_field(self):
        """
        Test raises BorealisFieldMissingError when unknown field supplied
        """
        extra_field = 'dummy'

        for val in self.file_types:
            with self.subTest(val=val):
                # Load the sample data
                self.data_type = val
                self.load_data_w_filename()

                # Add a fake field
                dkeys = [dkey for dkey in self.data.keys()]
                self.data[dkeys[0]][extra_field] = extra_field

                # Test raises appropriate error
                with self.assertRaises(
                        bor_exc.BorealisExtraFieldError) as err:
                    self.write_func(self.temp_file, self.data, val,
                                    self.file_struct)

                self.assertEqual(err.fields, {extra_field})
                self.assertEqual(err.record_name, dkeys[0])

                # Remove the temporary file, if it was created
                remove_temp_file(self.temp_file)

    def test_incorrect_data_format(self):
        """
        Test raises BorealisDataFormatTypeError with badly formatted data
        """
        bad_data_key = {'rawacf': 'scan_start_marker',
                        'bfiq': 'first_range_rrt',
                        'antenna_iq': 'num_slices'}
        bad_data_val = {'rawacf': 1, 'bfiq': 5, 'antenna_iq': 'a'}
        bad_data_msg = {'rawacf': "<class 'numpy.bool_'>",
                        'bfiq': "<class 'numpy.float32'>",
                        'antenna_iq': "<class 'numpy.int64'>"}

        for val in self.file_types:
            with self.subTest(val=val):
                # Load the sample data
                self.data_type = val
                self.load_data_w_filename()

                # Add a fake field
                dkeys = [dkey for dkey in self.data.keys()]
                self.data[dkeys[0]][bad_data_key[val]] = bad_data_val[val]

                # Test raises appropriate error
                with self.assertRaises(
                        bor_exc.BorealisDataFormatTypeError) as err:
                    self.write_func(self.temp_file, self.data, val,
                                    self.file_struct)

                self.assertGreater(
                    err.incorrect_types[bad_data_key[val]].find(
                        bad_data_msg[val]), 0)
                self.assertEqual(err.record_name, dkeys[0])

                # Remove the temporary file, if it was created
                remove_temp_file(self.temp_file)

    def test_wrong_borealis_filetype(self):
        """
        Test raises Borealis Error when specifying the wrong filetype.
        """
        wrong_filetype_exceptions = (bor_exc.BorealisExtraFieldError,
                                     bor_exc.BorealisFieldMissingError,
                                     bor_exc.BorealisDataFormatTypeError)

        for i, val in enumerate(self.file_types):
            with self.subTest(val=val):
                # Load the sample data that is not of the current file type
                self.data_type = self.file_types[i - 1]
                self.load_data_w_filename()

                # Try to write the data, specifying the current file type
                with self.assertRaises(wrong_filetype_exceptions):
                    self.write_func(self.temp_file, self.data, val,
                                    self.file_struct)

                # Remove the temporary file, if it was created
                remove_temp_file(self.temp_file)

    def test_wrong_borealis_file_structure(self):
        """
        Test raises BorealisStructureError when specifying wrong file structure
        """
        # Get the opposite of the current structure
        fstruct = "site" if self.file_struct == "array" else "array"

        # Cycle through the different file types
        for val in self.file_types:
            with self.subTest(val=val):
                # Load the sample data for the current structure
                self.data_type = val
                self.load_data_w_filename()

                # Attempt to write the test data with the wrong structure
                with self.assertRaises(bor_exc.BorealisStructureError):
                    self.write_func(self.temp_file, self.data, val, fstruct)

                # Remove the temporary file, if it was created
                remove_temp_file(self.temp_file)


class TestConvertBorealis(unittest.TestCase):
    """
    Testing class for converting Borealis data to standard SuperDARN data
    """

    def setUp(self):
        self.test_file = "fake.file"
        self.temp_file = "fake.temp"
        self.test_dir = os.path.join("..", "testdir")
        self.file_types = ["rawacf", "bfiq"]
        self.file_struct = "site"
        self.version = 4

    def tearDown(self):
        del self.test_file, self.test_dir, self.file_types, self.file_struct
        del self.version, self.temp_file

    def test_convert_to_dmap(self):
        """ Test successful conversion of Borealis data to DMap types
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        test_file_dict = get_test_files(get_borealis_type(
            "good", self.file_struct, self.version), test_dir=self.test_dir)

        for val in self.file_types:
            with self.subTest(val=val):
                # Set the test file data
                self.test_file = test_file_dict[val]
                self.temp_file = "{:s}.temp.dmap".format(self.test_file)

                # Run the data convertion
                pyDARNio.BorealisConvert(
                    self.test_file, val, self.temp_file,
                    borealis_file_structure=self.file_struct)

                # Test that the file was created
                self.assertTrue(remove_temp_file(self.temp_file))
