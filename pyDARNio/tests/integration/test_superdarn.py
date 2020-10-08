# Copyright (C) 2019 SuperDARN
# Author: Marina Schmidt, Angeline G. Burrell
# --------------------------------------------

import logging
import numpy as np
import os

import pyDARNio
from pyDARNio import superdarn_exceptions as sdarn_exc
from pyDARNio.tests.utils import file_utils

pydarnio_logger = logging.getLogger('pyDARNio')


class TestSDarnReadWrite(file_utils.TestReadWrite):
    def setUp(self):
        self.test_dir = os.path.join("..", "testdir")
        self.read_class = pyDARNio.SDarnRead
        self.write_class = pyDARNio.SDarnWrite
        self.read_func = None
        self.write_func = None
        self.read_dmap = None
        self.written_dmap = None
        self.temp_file = "temp.file"
        self.file_types = ['rawacf', 'fitacf', 'iqdat', 'grid', 'map']

    def tearDown(self):
        del self.read_func, self.write_func, self.read_dmap, self.written_dmap
        del self.temp_file, self.file_types, self.test_dir, self.read_func
        del self.write_func


class TestSDarnReadDmapWrite(file_utils.TestReadWrite):
    def setUp(self):
        self.test_dir = os.path.join("..", "testdir")
        self.read_class = pyDARNio.SDarnRead
        self.write_class = pyDARNio.DmapWrite
        self.read_func = None
        self.write_func = None
        self.read_dmap = None
        self.written_dmap = None
        self.temp_file = "temp.file"
        self.file_types = ['rawacf', 'fitacf', 'iqdat', 'grid', 'map']

    def tearDown(self):
        del self.read_func, self.write_func, self.read_dmap, self.written_dmap
        del self.temp_file, self.file_types, self.test_dir, self.read_func
        del self.write_func

    def test_write_missing_success_read_error(self):
        """Raise SuperDARNFieldMissingError when reading partially written file
        """

        # Set the missing field name and record number
        missing_field = {'rawacf': 'nave', 'fitacf': 'nave', 'iqdat': 'nave',
                         'grid': 'stid', 'map': 'stid'}
        rnum = 0

        # Test each of the file types
        for val in self.file_types:
            with self.subTest(val=val):
                # Get the locally stored data and delete a neccesary value
                (self.read_dmap,
                 self.temp_file) = file_utils.load_data_w_filename(val)
                del self.read_dmap[rnum][missing_field[val]]

                # Write the partial data
                self.write_func = file_utils.set_write_func(self.write_class,
                                                            self.read_dmap,
                                                            val)
                self.write_func(self.temp_file)

                # Read in the patial data
                self.set_read_func(val)

                with self.assertRaises(
                        sdarn_exc.SuperDARNFieldMissingError) as err:
                    self.written_dmap = self.read_func(self.temp_file)

                # Test the error message raised
                self.assertEqual(err.fields, {missing_field[val]})
                self.assertEqual(err.record_number, rnum)

                # Remove the the temporary file
                self.assertTrue(file_utils.remove_temp_file(self.temp_file))

    def test_write_extra_read_error(self):
        """Raise SuperDARNExtraFieldError when reading file written with extra
        field
        """
        # Set the missing field name and record number
        rnum = 0
        extra_field = 'dummy'

        # Test each of the file types
        for val in self.file_types:
            with self.subTest(val=val):
                # Get the locally stored data and add an extra value
                (self.read_dmap,
                 self.temp_file) = file_utils.load_data_w_filename(val)
                self.read_dmap[rnum].update({extra_field: extra_field})
                self.read_dmap[rnum].move_to_end(extra_field, last=False)

                # Write the extra data
                self.write_func = file_utils.set_write_func(self.write_class,
                                                            self.read_dmap,
                                                            val)
                self.write_func(self.temp_file)

                # Read in the extra data
                self.set_read_func(val)

                with self.assertRaises(
                        sdarn_exc.SuperDARNExtraFieldError) as err:
                    self.written_dmap = self.read_func(self.temp_file)

                # Test the error message raised
                self.assertEqual(err.fields, {extra_field})
                self.assertEqual(err.record_number, rnum)

                # Remove the the temporary file
                self.assertTrue(file_utils.remove_temp_file(self.temp_file))

    def test_write_incorrect_read_rawacf_from_dict(self):
        """Raise SuperDARNDataFormatTypeError when reading badly written dict
        """
        # Test the only file type that currently has a data dict
        val = 'rawacf_dict'

        # Get the locally stored data and add a bad dictionary record
        read_dict, self.temp_file = file_utils.load_data_w_filename(val)
        read_dict[0]['stid'] = np.int8(read_dict[0]['stid'])

        # Convert from dict to Dmap and update the file type (val)
        self.read_dmap = pyDARNio.dict2dmap(read_dict)
        val = val.split('_')[0]

        # Write the Dmap data
        self.write_func = file_utils.set_write_func(self.write_class,
                                                    self.read_dmap, val)
        self.write_func(self.temp_fil)

        # Read from the written file
        self.set_read_func(val)
        with self.assertRaises(sdarn_exc.SuperDARNDataFormatTypeError):
            self.written_dmap = self.read_func(self.temp_file)


class TestDmapReadSDarnWrite(file_utils.TestReadWrite):
    def setUp(self):
        self.test_dir = os.path.join("..", "testdir")
        self.read_class = pyDARNio.DmapRead
        self.write_class = pyDARNio.SDarnWrite
        self.read_func = None
        self.write_func = None
        self.read_dmap = None
        self.written_dmap = None
        self.temp_file = "temp.file"
        self.file_types = ['rawacf', 'fitacf', 'iqdat', 'grid', 'map']

    def tearDown(self):
        del self.read_func, self.write_func, self.read_dmap, self.written_dmap
        del self.temp_file, self.file_types, self.test_dir, self.read_func
        del self.write_func

    def test_dict2dmap_write_rawacf(self):
        """Use dict2dmap to convert a dictionary to DMap then SDarnWrite file
        """
        # Test the only file type that currently has a data dict
        val = 'rawacf_dict'

        # Get the locally stored data
        read_dict, self.temp_file = file_utils.load_data_w_filename(val)

        # Convert from dict to Dmap and update the file type (val)
        self.read_dmap = pyDARNio.dict2dmap(read_dict)
        val = val.split('_')[0]

        # Write the Dmap data
        self.write_func = file_utils.set_write_func(self.write_class,
                                                    self.read_dmap, val)
        self.write_func(self.temp_file)

        # Read in the Dmap data
        self.set_read_func(val)
        self.written_dmap = self.read_func(self.temp_file)

        # Compare the read and written data
        self.dmap_list_compare()

        # Remove the the temporary file
        self.assertTrue(file_utils.remove_temp_file(self.temp_file))

    def test_write_incorrect_rawacf_from_dict(self):
        """Raise SuperDARNDataFormatTypeError when writing badly formatted dict
        """
        # Test the only file type that currently has a data dict
        val = 'rawacf_dict'

        # Get the locally stored data and add a bad dictionary record
        read_dict, self.temp_file = file_utils.load_data_w_filename(val)
        read_dict[0]['stid'] = np.int8(read_dict[0]['stid'])

        # Convert from dict to Dmap and update the file type (val)
        self.read_dmap = pyDARNio.dict2dmap(read_dict)
        val = val.split('_')[0]

        # Write the Dmap data
        self.write_func = file_utils.set_write_func(self.write_class,
                                                    self.read_dmap, val)

        with self.assertRaises(sdarn_exc.SuperDARNDataFormatTypeError):
            self.write_func(self.temp_file)

    def test_DmapRead_SDarnWrite_SDarnRead(self):
        """Test read/write/read with DmapRead, SDarnWrite, and SDarnRead
        """
        if not os.path.isdir(self.test_dir):
            self.skipTest('test directory is not included with pyDARNio')

        test_file_dict = file_utils.get_test_files("good",
                                                   test_dir=self.test_dir)

        for val in self.file_types:
            with self.subTest(val=val):
                # Read in the test file
                self.set_read_func(val)
                self.read_dmap = self.read_func(test_file_dict[val])

                # Write the data
                self.write_func = file_utils.set_write_func(self.write_class,
                                                            self.read_dmap,
                                                            val)
                self.write_func(self.temp_file)

                # Read in again using SDarnRead
                self.read_class = pyDARNio.SDarnRead
                self.set_read_func(val)
                self.written_dmap = self.read_func(self.temp_file)
                self.read_class = pyDARNio.DmapRead  # Reset the reading class

                # Assert the read and written data are the same
                self.dmap_list_compare()

                # Test the file creation and remove the temp file
                self.assertTrue(file_utils.remove_temp_file(self.temp_file))
