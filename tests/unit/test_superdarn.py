# Copyright (C) 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marina Schmidt, Angeline Burrell
"""
This test suite is to test the implementation for the following classes:
    SDarnRead
    DarnUtilities
    SDarnWrite
Support for the following SuperDARN file types:
    iqdat
    rawacf
    fitacf
    grid
    map
"""

import bz2
import copy
import collections
import logging
import numpy as np
import os
import unittest

import pyDARNio
from pyDARNio import superdarn_exceptions as sdarn_exp

import map_data_sets
import grid_data_sets
import fitacf_data_sets
import iqdat_data_sets
import rawacf_data_sets

pydarnio_logger = logging.getLogger('pydarnio')

# Define the test files and directory
#
# If these files change, the unit tests will need to be updated
test_dir = os.path.join("..", "testfiles")
rawacf_stream = "20170410.1801.00.sas.stream.rawacf.bz2"
test_file_dict = {"rawacf": "20170410.1801.00.sas.rawacf",
                  "fitacf": "20160331.2201.00.mcm.a.fitacf",
                  "map": "20170114.map",
                  "iqdat": "20160316.1945.01.rkn.iqdat",
                  "grid": "20180220.C0.rkn.grid"}
corrupt_files = ["20070117.1001.00.han.rawacf", "20090320.1601.00.pgr.rawacf"]


@unittest.skipIf(not os.path.isdir(test_dir),
                 'test directory is not included with pyDARNio')
class TestSDarnRead(unittest.TestCase):
    """
    Testing class for SDarnRead class
    """
    def setUp(self):
        self.test_file = "somefile.rawacf"
        self.data = None
        self.rec = None

    def tearDown(self):
        del self.test_file, self.data, self.rec

    def test_incorrect_filepath(self):
        """
        Test raise FileNotFoundError with bad filename or path
        """
        for tdir in ["bad_dir", test_dir]:
            with self.subTest(val=tdir):
                # Create a test filename with path
                self.test_file = os.path.join(val, self.test_file)

                # Assert correct error and message for bad filename
                self.assertRaises(FileNotFoundError, pyDARNio.SDarnRead,
                                  self.test_file)

    def test_empty_file(self):
        """
        Tests raise EmptyFileError with an empty file
        """
        self.test_file = os.path.join(test_dir, "empty.rawacf")
        self.assertRaises(pyDARNio.dmap_exceptions.EmptyFileError,
                          pyDARNio.SDarnRead, self.test_file)

    def test_good_open_file(self):
        """
        Test SDarn file opening, reading, and converting to a bytearray

        Checks:
            - bytearray instance is created from reading in the file
            - bytearray is not empty
        """
        for tfile in test_file_dict.values():
            with self.subTest(val=tfile):
                # Create a test filename with path
                self.test_file = os.path.join(test_dir, val)

                # Load the file
                self.data = pyDARNio.SDarnRead(self.test_file)

                # Test the file data
                self.assertIsInstance(self.data.dmap_bytearr, bytearray)
                self.assertGreater(self.data.dmap_end_bytes, 0)

    def load_test_file_record(self, file_type=''):
        """ Load a test file data record
        """
        # Build the filename and load the data
        self.test_file = os.path.join(test_dir, test_file_dict[file_type])
        self.data = pyDARNio.SDarnRead(self.test_file)

        # Read the data
        read_func = getattr(self.data, "read_{:s}".format(file_type))
        _ = read_func()
        self.rec = self.data.get_dmap_records

    def test_read_iqdat(self):
        """
        Test reading records from iqdat.

        Checks:
            - returns correct data structures
            - returns expected values
        """
        # Load the data and read in the first record
        load_test_file_record(file_type='iqdat')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[0]['rxrise'], pyDARNio.DmapScalar)
        self.assertIsInstance(self.rec[3]['tsc'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[5]['mppul'].value, int)
        self.assertIsInstance(self.rec[6]['tnoise'].value, np.ndarray)
        self.assertEqual(self.rec[7]['channel'].value, 0)
        self.assertEqual(self.rec[10]['data'].dimension, 1)

    def test_read_rawacf(self):
        """
        Test reading records from rawacf.

        Checks:
            - returns correct data structures
            - returns expected values
        """
        # Load the data and read in the first record
        load_test_file_record(file_type='rawacf')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['channel'], pyDARNio.DmapScalar)
        self.assertIsInstance(self.rec[1]['ptab'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[7]['channel'].value, int)
        self.assertIsInstance(self.rec[2]['xcfd'].value, np.ndarray)
        self.assertEqual(self.rec[0]['xcfd'].dimension, 3)

    def test_read_fitacf(self):
        """
        Test reading records from fitacf.

        Checks:
            - returns correct data structures
            - returns expected values
        """
        # Load the data and read in the first record
        load_test_file_record(file_type='fitacf')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['bmnum'], pyDARNio.DmapScalar)
        self.assertIsInstance(self.rec[1]['ptab'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[7]['channel'].value, int)
        self.assertIsInstance(self.rec[2]['ltab'].value, np.ndarray)
        self.assertEqual(self.rec[0]['ptab'].dimension, 1)

    def test_read_grid(self):
        """
        Test reading records from grid file.

        Checks:
            - returns correct data structures
            - returns expected values
        """
        # Load the data and read in the first record
        load_test_file_record(file_type='grid')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['start.year'], pyDARNio.DmapScalar)
        self.assertIsInstance(self.rec[1]['v.max'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[7]['end.day'].value, int)
        self.assertIsInstance(self.rec[2]['stid'].value, np.ndarray)
        self.assertEqual(self.rec[0]['nvec'].dimension, 1)

    def test_read_map(self):
        """
        Test reading records from map file.

        Checks:
            - returns correct data structures
            - returns expected values
        """
        # Load the data and read in the first record
        load_test_file_record(file_type='map')

        # Test the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[2]['IMF.flag'],
                              pyDARNio.io.datastructures.DmapScalar)
        self.assertIsInstance(self.rec[3]['stid'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[8]['IMF.flag'].value, int)
        self.assertIsInstance(self.rec[10]['stid'].value, np.ndarray)
        self.assertEqual(self.rec[3]['stid'].dimension, 1)
        # this will be file dependent... future working test project.
        self.assertEqual(self.rec[0]['stid'].shape[0], 14)

    def test_read_corrupt_files(self):
        """
        Test raises a dmap_exceptions Error when readig a corrupt file
        """
        for val in [(corrupt_files[0],
                     pyDARNio.dmap_exceptions.DmapDataTypeError),
                    (corrupt_files[1],
                     pyDARNio.dmap_exceptions.NegativeByteError)]:
            with self.subTest(val=val):
                self.test_file = os.path.join(test_dir, val[0])
                self.data = pyDARNio.SDarnRead(self.test_file)
                with self.assertRaises(val[1]):
                    dmap.read_rawacf()

    def test_dmap_read_stream(self):
        """
        Test read_records on dmap data stream formed from a bzip2 file

         Checks:
            - returns correct data structures
            - returns expected values
        """
        # bz2 opens the compressed file into a data
        # stream of bytes without actually uncompressing the file
        self.test_file = os.path.join(test_dir, rawacf_stream)
        with bz2.open(self.test_file) as fp:
            dmap_stream = fp.read()
        self.data = pyDARNio.SDarnRead(dmap_stream, True)
        _ = self.data.read_rawacf()
        self.rec = self.data.get_dmap_records

        # Test thee output of the first record
        self.assertIsInstance(self.rec, collections.deque)
        self.assertIsInstance(self.rec[0], collections.OrderedDict)
        self.assertIsInstance(self.rec[4]['channel'], pyDARNio.DmapScalar)
        self.assertIsInstance(self.rec[1]['ptab'], pyDARNio.DmapArray)
        self.assertIsInstance(self.rec[7]['channel'].value, int)
        self.assertIsInstance(self.rec[2]['xcfd'].value, np.ndarray)
        self.assertEqual(self.rec[0]['xcfd'].dimension, 3)

    def test_dmap_read_corrupt_stream(self):
        """
        Test raises pydmap exception when reading a corrupted stream from
        a compressed file

        Method - Reead in a compressed file from a good stream, then insert
        some random bytes to produce a corrupt stream.
        """
        # Open the data stream
        self.test_file = os.path.join(test_dir, rawacf_stream)
        with bz2.open(self.test_filee) as fp:
            dmap_stream = fp.read()

        # Load and corrupt data, converting to byte array for mutability
        # since bytes are immutable.
        self.data = bytearray(dmap_stream[0:36])
        self.data[36:40] = bytearray(str(os.urandom(4)).encode('utf-8'))
        self.data[40:] = dmap_stream[37:]
        self.rec = pyDARNio.SDarnRead(self.data, True)

        # Assert data from corrupted stream is corrupted
        with self.assertRaises(pyDARNio.dmap_exceptions.DmapDataError):
            self.rec.read_rawacf()


class TestDarnUtilities(unittest.TestCase):
    """
    Testing DarnUtilities class.

    Notes
    -----
    All methods in this class are static so there is no constructor testing
    """
    def setUp(self):
        self.tdicts = [{'a': 's', 'c': 'i', 'd': 'f'},
                       {'rst': '4.1', 'stid': 3, 'vel': [2.3, 4.5]},
                       {'fitacf': 'f', 'rawacf': 's', 'map': 'm'}]

    def tearDown(self):
        del self.tdicts

    def test_dict_key_diff(self):
        """
        Test the difference in keys between two dictionaries, order dependent
        """
        self.tdicts[1] = {'1': 'a', 'c': 2, 'z': 'fun', 'd': 'dog'}

        for val in [(self.tdicts[0], self.tdicts[1], {'a'}),
                    (self.tdicts[1], self.tdicts[0], {'1', 'z'})]:
            with self.subTest(val=val):
                out = pyDARNio.SDarnUtilities.dict_key_diff(val[0], val[1])
                self.assertEqual(val[2], out)

    def test_dict_list2set(self):
        """
        Test conversion of lists of dictionaries into concatenated full sets

        Expected behaviour
        ------------------
        Returns only a single set the comprises of the dictionary keys
        given in the list
        """
        dict_keys = {'a', 'c', 'd', 'rst', 'stid', 'vel', 'fitacf', 'rawacf',
                     'map'}
        out = pyDARNio.SDarnUtilities.dict_list2set(self.tdicts)
        self.assertEqual(dict_keys, out)

    def test_extra_field_check_pass(self):
        """
        Test extra_field_check success

        Method - this method checks if there are differences in the key sets of
        dictionaries that when passed a record and field names it will indicate
        if there is an extra field in the record key set

        Expected behaviour
        ------------------
        Silent success - if there are no differences in the key set then
        nothing is returned or raised
        """
        out = {'a': 3, 'c': 3, 'd': 3, 'rst': 1, 'vel': 'd'}
        pyDARNio.SDarnUtilities.extra_field_check(self.tdicts, out, 1)

    def test_extra_field_check_fail(self):
        """
        Test extra_field_check failur raises SuperDARNExtraFieldError

        Method - this method checks if there are  differences in the key sets
        of dictionaries that when passed a record and field names it will
        indicate if there is an extra field in the record key set

        Expected behaviour
        -----------------
        Raises SuperDARNExtraFieldError because there are differences between
        the two dictionary sets
        """
        out = {'a': 3, 'b': 3, 'c': 2, 'd': 3, 'rst': 1, 'vel': 'd'}
        with self.assertRaises(sdarn_exp.SuperDARNExtraFieldError) as err:
            pyDARNio.SDarnUtilities.extra_field_check(self.tdicts, out, 1)

        self.assertEqual(err.exception.fields, {'b'})

    def test_missing_field_check_pass(self):
        """
        Testing missing_field_check - Reverse idea of the extra_field_check,
        should find missing fields in a record when compared to a key set of
        SuperDARN field names

        Expected behaviour
        ------------------
        Nothing - if there is not differences then nothing happens
        """
        in_list = [dict(self.tdicts[0]),
                   {'a': 3, 'c': 2, 'd': 2, 'stid': 's', 'rst': 1, 'vel': 'd'},
                   dict(self.tdicts[1])]
        in_list[0].update(self.tdicts[-1])
        in_list[2].update(self.tdicts[2])
        for val in in_list:
            with self.subTest(val=val):
                pyDARNio.SDarnUtilities.missing_field_check(self.tdicts, val, 1)

    def test_missing_field_check_fail(self):
        """
        Test raises SuperDARNFieldMissingError with appropriate fields missing

        Method - Reverse idea of the extra_field_check, should find missing
        fields in a record when compared to a key set of SuperDARN field names

        Expected behaviour
        ------------------
        Raise SuperDARNFieldMissingError - raised when there is a difference
        between dictionary key sets
        """

        for val in [({'a': 3, 'b': 3, 'd': 2, 'stid': 's', 'vel': 'd'},
                     {'c', 'rst'}),
                    ({'a': 3, 'b': 3, 'd': 2, 'stid': 's', 'rst': 1,
                      'vel': 'd', 'fitacf': 3, 'map': 4}, {'c', 'rawacf'})]:
            with self.subTest(val=val):
                with self.assertRaises(
                        sdarn_exp.SuperDARNFieldMissingError) as err:
                    pyDARNio.SDarnUtilities.missing_field_check(self.tdicts,
                                                                val[0], 1)

                # Items in tdicts, but not in rdict
                self.assertEqual(err.exception.fields, val[1])

    def test_incorrect_types_check_pass(self):
        """
        Test incorrect_types_check - this method checks if the field data
        format type is not correct to specified SuperDARN field type.

        Note
        ----
        This method only works on pyDARNio DMAP record data structure

        Expected Behaviour
        ------------------
        Nothing - should not return or raise anything if the fields
        are the correct data format type
        """
        rdict = {'a': pyDARNio.DmapScalar('a', 1, 1, self.tdicts[0]['a']),
                 'c': pyDARNio.DmapScalar('a', 1, 1, self.tdicts[0]['c']),
                 'd': pyDARNio.DmapArray('a', np.array([2.4, 2.4]), 1,
                                         self.tdicts[0]['d'], 1, [3]),
                 'fitacf': pyDARNio.DmapScalar('a', 1, 1,
                                               self.tdicts[-1]['fitacf']),
                 'rawacf': pyDARNio.DmapScalar('a', 1, 1,
                                               self.tdicts[-1]['rawacf']),
                 'map': pyDARNio.DmapScalar('a', 1, 1,
                                            self.tdicts[-1]['map'])}

        pyDARNio.SDarnUtilities.incorrect_types_check([self.tdicts[0],
                                                       self.tdicts[-1]],
                                                      rdict, 1)

    def test_incorrect_types_check_fail(self):
        """
        Test incorrect_types_check - this method checks if the field data
        format type is not correct to specified SuperDARN field type.

        Note
        ----
        This method only works on pyDARNio DMAP record data structure

        Expected Behaviour
        ------------------
        Raises SuperDARNDataFormatTypeError - because the field format types
        should not be the same.
        """
        rdict = {'a': pyDARNio.DmapScalar('a', 1, 1, self.tdicts[0]['a']),
                 'c': pyDARNio.DmapScalar('a', 1, 1, self.tdicts[0]['c']),
                 'd': pyDARNio.DmapArray('a', np.array([2.4, 2.4]), 1,
                                         self.tdicts[0]['d'], 1, [3]),
                 'fitacf': pyDARNio.DmapScalar('a', 1, 1,
                                               self.tdicts[-1]['rawacf']),
                 'rawacf': pyDARNio.DmapScalar('a', 1, 1,
                                               self.tdicts[-1]['rawacf']),
                 'map': pyDARNio.DmapScalar('a', 1, 1,
                                            self.tdicts[-1]['map'])}

        with self.assertRaises(sdarn_exp.SuperDARNDataFormatTypeError) as err:
            pyDARNio.SDarnUtilities.incorrect_types_check([self.tdicts[0],
                                                           self.tdicts[-1]],
                                                          rdict, 1)

        self.assertEqual(err.exception.incorrect_params, {'fitacf': 'f'})


@unittest.skipIf(not os.path.isdir(test_dir),
                 'test directory is not included with pyDARNio')
class TestSDarnWrite(unittest.TestCase):
    """
    Tests SDarnWrite class
    """
    def setUp(self):
        pass

    def test_darn_write_constructor(self):
        """
        Tests SDarnWrite constructor

        Expected behaviour
        ------------------
        Contains file name of the data if given to it.
        """
        rawacf_data = copy.deepcopy(rawacf_data_sets.rawacf_data)
        darn = pyDARNio.SDarnWrite(rawacf_data, "rawacf_test.rawacf")
        self.assertEqual(darn.filename, "rawacf_test.rawacf")

    def test_empty_record(self):
        """
        Tests if an empty record is given. This will later be changed for
        real-time implementation.

        Expected behaviour
        ------------------
        Raise DmapDataError if no data is provided to the constructor
        """
        with self.assertRaises(pyDARNio.dmap_exceptions.DmapDataError):
            pyDARNio.SDarnWrite([], 'dummy_file.acf')

    def test_incorrect_filename_input_using_write_methods(self):
        """
        Tests if a file name is not provided to any of the write methods

        Expected behaviour
        ------------------
        All should raise a FilenameRequiredError - if no file name is given
        what do we write to.
        """
        rawacf_data = copy.deepcopy(rawacf_data_sets.rawacf_data)
        dmap_data = pyDARNio.SDarnWrite(rawacf_data)
        with self.assertRaises(pyDARNio.dmap_exceptions.FilenameRequiredError):
            dmap_data.write_rawacf()
            dmap_data.write_fitacf()
            dmap_data.write_iqdat()
            dmap_data.write_grid()
            dmap_data.write_map()
            dmap_data.write_dmap()

    def test_SDarnWrite_missing_field_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNFieldMissingError - because the rawacf data is
        missing field nave
        """
        rawacf_missing_field = copy.deepcopy(rawacf_data_sets.rawacf_data)
        del rawacf_missing_field[2]['nave']

        dmap = pyDARNio.SDarnWrite(rawacf_missing_field)

        try:
            dmap.write_rawacf("test_rawacf.rawacf")
        except sdarn_exp.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'nave'})
            self.assertEqual(err.record_number, 2)

    def test_extra_field_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNExtraFieldError because the rawacf data
        has an extra field dummy
        """
        rawacf_extra_field = copy.deepcopy(rawacf_data_sets.rawacf_data)
        rawacf_extra_field[1]['dummy'] = pyDARNio.DmapScalar('dummy', 'nothing',
                                                           chr(1), 's')
        dmap = pyDARNio.SDarnWrite(rawacf_extra_field)

        try:
            dmap.write_rawacf("test_rawacf.rawacf")
        except sdarn_exp.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 1)

    def test_incorrect_data_format_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises SuperDARNDataFormatTypeError because the rawacf data has the
        wrong type for the scan field
        """
        rawacf_incorrect_fmt = copy.deepcopy(rawacf_data_sets.rawacf_data)
        rawacf_incorrect_fmt[2]['scan'] = \
            rawacf_incorrect_fmt[2]['scan']._replace(data_type_fmt='c')
        dmap = pyDARNio.SDarnWrite(rawacf_incorrect_fmt)

        try:
            dmap.write_rawacf("test_rawacf.rawacf")
        except sdarn_exp.SuperDARNDataFormatTypeError as err:
            self.assertEqual(err.incorrect_params['scan'], 'h')
            self.assertEqual(err.record_number, 2)

    def test_writing_rawacf(self):
        """
        Tests write_rawacf method - writes a rawacf file

        Expected behaviour
        ------------------
        Rawacf file is produced
        """
        rawacf_data = copy.deepcopy(rawacf_data_sets.rawacf_data)

        dmap = pyDARNio.SDarnWrite(rawacf_data)

        dmap.write_rawacf("test_rawacf.rawacf")
        # only testing the file is created since it should only be created
        # at the last step after all checks have passed
        # Testing the integrity of the insides of the file will be part of
        # integration testing since we need SDarnRead for that.
        self.assertTrue(os.path.isfile("test_rawacf.rawacf"))
        os.remove("test_rawacf.rawacf")

    def test_writing_fitacf(self):
        """
        Tests write_fitacf method - writes a fitacf file

        Expected behaviour
        ------------------
        fitacf file is produced
        """
        fitacf_data = copy.deepcopy(fitacf_data_sets.fitacf_data)
        dmap = pyDARNio.SDarnWrite(fitacf_data)

        dmap.write_fitacf("test_fitacf.fitacf")
        self.assertTrue(os.path.isfile("test_fitacf.fitacf"))
        os.remove("test_fitacf.fitacf")

    def test_missing_fitacf_field(self):
        """
        Tests write_fitacf method - writes a fitacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNFieldMissingError - because the fitacf data is
        missing field stid
        """
        fitacf_missing_field = copy.deepcopy(fitacf_data_sets.fitacf_data)
        del fitacf_missing_field[0]['stid']
        dmap = pyDARNio.SDarnWrite(fitacf_missing_field)

        try:
            dmap.write_fitacf("test_fitacf.fitacf")
        except sdarn_exp.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'stid'})
            self.assertEqual(err.record_number, 0)

    def test_extra_fitacf_field(self):
        """
        Tests write_fitacf method - writes a fitacf structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNExtraFieldError because the fitacf data
        has an extra field dummy
        """
        fitacf_extra_field = copy.deepcopy(fitacf_data_sets.fitacf_data)
        fitacf_extra_field[1]['dummy'] = pyDARNio.DmapArray('dummy',
                                                          np.array([1, 2]),
                                                          chr(1), 'c', 1, [2])
        dmap = pyDARNio.SDarnWrite(fitacf_extra_field)

        try:
            dmap.write_fitacf("test_fitacf.fitacf")
        except sdarn_exp.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 1)

    def test_incorrect_fitacf_data_type(self):
        """
        Tests write_fitacf method - writes a fitacf structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises SuperDARNDataFormatTypeError because the fitacf data has the
        wrong type for the ltab field
        """

        fitacf_incorrect_fmt = copy.deepcopy(fitacf_data_sets.fitacf_data)
        fitacf_incorrect_fmt[1]['ltab'] = \
            fitacf_incorrect_fmt[1]['ltab']._replace(data_type_fmt='s')
        dmap = pyDARNio.SDarnWrite(fitacf_incorrect_fmt)

        try:
            dmap.write_fitacf("test_fitacf.fitacf")
        except sdarn_exp.SuperDARNDataFormatTypeError as err:
            self.assertEqual(err.incorrect_params['ltab'], 'h')
            self.assertEqual(err.record_number, 1)

    def test_writing_iqdat(self):
        """
        Tests write_iqdat method - writes a iqdat file

        Expected behaviour
        ------------------
        iqdat file is produced
        """
        iqdat_data = copy.deepcopy(iqdat_data_sets.iqdat_data)
        dmap = pyDARNio.SDarnWrite(iqdat_data)

        dmap.write_iqdat("test_iqdat.iqdat")
        self.assertTrue(os.path.isfile("test_iqdat.iqdat"))
        os.remove("test_iqdat.iqdat")

    def test_missing_iqdat_field(self):
        """
        Tests write_iqdat method - writes a iqdat structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNFieldMissingError - because the iqdat data is
        missing field chnnum
        """

        iqdat_missing_field = copy.deepcopy(iqdat_data_sets.iqdat_data)
        del iqdat_missing_field[1]['chnnum']
        dmap = pyDARNio.SDarnWrite(iqdat_missing_field)

        try:
            dmap.write_iqdat("test_iqdat.iqdat")
        except sdarn_exp.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'chnnum'})
            self.assertEqual(err.record_number, 1)

    def test_extra_iqdat_field(self):
        """
        Tests write_iqdat method - writes a iqdat structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNExtraFieldError because the iqdat data
        has an extra field dummy
        """
        iqdat_extra_field = copy.deepcopy(iqdat_data_sets.iqdat_data)
        iqdat_extra_field[2]['dummy'] = \
            pyDARNio.DmapArray('dummy', np.array([1, 2]), chr(1), 'c', 1, [2])
        dmap = pyDARNio.SDarnWrite(iqdat_extra_field)

        try:
            dmap.write_iqdat("test_iqdat.iqdat")
        except sdarn_exp.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 2)

    def test_incorrect_iqdat_data_type(self):
        """
        Tests write_iqdat method - writes a iqdat structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises SuperDARNDataFormatTypeError because the iqdat data has the
        wrong type for the lagfr field
        """
        iqdat_incorrect_fmt = copy.deepcopy(iqdat_data_sets.iqdat_data)
        iqdat_incorrect_fmt[2]['lagfr'] = \
            iqdat_incorrect_fmt[2]['lagfr']._replace(data_type_fmt='d')
        dmap = pyDARNio.SDarnWrite(iqdat_incorrect_fmt)

        try:
            dmap.write_iqdat("test_iqdat.iqdat")
        except sdarn_exp.SuperDARNDataFormatTypeError as err:
            self.assertEqual(err.incorrect_params['lagfr'], 'h')
            self.assertEqual(err.record_number, 2)

    def test_writing_map(self):
        """
        Tests write_map method - writes a map file

        Expected behaviour
        ------------------
        map file is produced
        """
        map_data = copy.deepcopy(map_data_sets.map_data)
        dmap = pyDARNio.SDarnWrite(map_data)

        dmap.write_map("test_map.map")
        self.assertTrue(os.path.isfile("test_map.map"))
        os.remove("test_map.map")

    def test_missing_map_field(self):
        """
        Tests write_map method - writes a map structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNFieldMissingError - because the map data is
        missing field stid
        """
        map_missing_field = copy.deepcopy(map_data_sets.map_data)
        del map_missing_field[0]['IMF.Kp']
        dmap = pyDARNio.SDarnWrite(map_missing_field)

        try:
            dmap.write_map("test_map.map")
        except sdarn_exp.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'IMF.Kp'})
            self.assertEqual(err.record_number, 0)

    def test_extra_map_field(self):
        """
        Tests write_map method - writes a map structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNExtraFieldError because the map data
        has an extra field dummy
        """
        map_extra_field = copy.deepcopy(map_data_sets.map_data)
        map_extra_field[1]['dummy'] = \
            pyDARNio.DmapArray('dummy', np.array([1, 2]), chr(1), 'c', 1, [2])
        dmap = pyDARNio.SDarnWrite(map_extra_field)

        try:
            dmap.write_map("test_map.map")
        except sdarn_exp.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 1)

    def test_incorrect_map_data_type(self):
        """
        Tests write_map method - writes a map structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises SuperDARNDataFormatTypeError because the map data has the
        wrong type for the IMF.Bx field
        """
        map_incorrect_fmt = copy.deepcopy(map_data_sets.map_data)
        map_incorrect_fmt[2]['IMF.Bx'] = \
            map_incorrect_fmt[2]['IMF.Bx']._replace(data_type_fmt='i')
        dmap = pyDARNio.SDarnWrite(map_incorrect_fmt)

        try:
            dmap.write_map("test_map.map")
        except sdarn_exp.SuperDARNDataFormatTypeError as err:
            self.assertEqual(err.incorrect_params.keys(), {'IMF.Bx'})
            self.assertEqual(err.record_number, 2)

    def test_writing_grid(self):
        """
        Tests write_grid method - writes a grid file

        Expected behaviour
        ------------------
        grid file is produced
        """
        grid_data = copy.deepcopy(grid_data_sets.grid_data)
        dmap = pyDARNio.SDarnWrite(grid_data)

        dmap.write_grid("test_grid.grid")
        self.assertTrue(os.path.isfile("test_grid.grid"))
        os.remove("test_grid.grid")

    def test_missing_grid_field(self):
        """
        Tests write_grid method - writes a grid structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNFieldMissingError - because the grid data is
        missing field stid
        """
        grid_missing_field = copy.deepcopy(grid_data_sets.grid_data)
        del grid_missing_field[1]['start.year']
        dmap = pyDARNio.SDarnWrite(grid_missing_field)

        try:
            dmap.write_grid("test_grid.grid")
        except sdarn_exp.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'start.year'})
            self.assertEqual(err.record_number, 1)

    def test_extra_grid_field(self):
        """
        Tests write_grid method - writes a grid structure file for the
        given data

        Expected behaviour
        ------------------
        Raises SuperDARNExtraFieldError because the grid data
        has an extra field dummy
        """
        grid_extra_field = copy.deepcopy(grid_data_sets.grid_data)
        grid_extra_field[0]['dummy'] = \
            pyDARNio.DmapArray('dummy', np.array([1, 2]), chr(1), 'c', 1, [2])
        dmap = pyDARNio.SDarnWrite(grid_extra_field)

        try:
            dmap.write_grid("test_grid.grid")
        except sdarn_exp.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 0)

    def test_incorrect_grid_data_type(self):
        """
        Tests write_grid method - writes a grid structure file for the
        given data

        Expected Behaviour
        -------------------
        Raises SuperDARNDataFormatTypeError because the grid data has the
        wrong type for the v.min field
        """
        grid_incorrect_fmt = copy.deepcopy(grid_data_sets.grid_data)
        grid_incorrect_fmt[2]['v.min'] = \
            grid_incorrect_fmt[2]['v.min']._replace(data_type_fmt='d')
        dmap = pyDARNio.SDarnWrite(grid_incorrect_fmt)

        try:
            dmap.write_grid("test_grid.grid")
        except sdarn_exp.SuperDARNDataFormatTypeError as err:
            self.assertEqual(err.incorrect_params.keys(), {'v.min'})
            self.assertEqual(err.record_number, 2)
