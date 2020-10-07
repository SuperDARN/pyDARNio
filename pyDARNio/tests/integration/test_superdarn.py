# Copyright (C) 2019 SuperDARN
# Author: Marina Schmidt, Angeline G. Burrell
import bz2
import copy
import logging
import numpy as np
import os
import unittest

import pyDARNio
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


class IntegrationSuperdarnio(unittest.TestCase):
    def setUp(self):
        pass

    def test_DmapWrite_missing_SDarnRead_rawacf(self):
        """
        Test DmapWrite writes a rawacf file missing the field nave in record 2
        and SDarnRead reads the file

        Behaviour: Raise SuperDARNFieldMissingError
        """
        rawacf_missing_field = copy.deepcopy(rawacf_data_sets.rawacf_data)
        del rawacf_missing_field[0]['nave']
        dmap_write = pydarnio.DmapWrite(rawacf_missing_field)
        dmap_write.write_dmap("test_missing_rawacf.rawacf")

        darn_read = pydarnio.SDarnRead("test_missing_rawacf.rawacf")
        try:
            darn_read.read_rawacf()
        except pydarnio.superdarn_exceptions.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'nave'})
            self.assertEqual(err.record_number, 0)

        os.remove("test_missing_rawacf.rawacf")

    def test_DmapWrite_extra_SDarnRead_rawacf(self):
        """
        Test DmapWrite writes a rawacf file with an extra field and SDarnRead
        reads the file

        Behaviour: Raised SuperDARNExtraFieldError
        """
        rawacf_extra_field = copy.deepcopy(rawacf_data_sets.rawacf_data)
        rawacf_extra_field[0].update({'dummy': 'dummy'})
        rawacf_extra_field[0].move_to_end('dummy', last=False)
        dmap_write = pydarnio.DmapWrite(rawacf_extra_field, )
        dmap_write.write_dmap("test_extra_rawacf.rawacf")

        darn_read = pydarnio.SDarnRead("test_extra_rawacf.rawacf")
        try:
            darn_read.read_rawacf()
        except pydarnio.superdarn_exceptions.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 0)
        os.remove("test_extra_rawacf.rawacf")

    def test_dict2dmap_SDarnWrite_rawacf(self):
        """
        Test dict2dmap to convert a dictionary to dmap then SDarnWrite write
        rawacf file
        """
        rawacf_dict_data = copy.deepcopy(rawacf_dict_sets.rawacf_dict_data)
        dmap_rawacf = pydarnio.dict2dmap(rawacf_dict_data)
        darn_read = pydarnio.SDarnWrite(dmap_rawacf)
        darn_read.write_rawacf("test_rawacf.rawacf")
        dmap_read = pydarnio.DmapRead("test_rawacf.rawacf")
        dmap_data = dmap_read.read_records()
        dmap_data = dmap_read.get_dmap_records
        self.dmap_compare(dmap_data, dmap_rawacf)
        os.remove("test_rawacf.rawacf")

    def test_SDarnWrite_incorrect_rawacf_from_dict(self):
        """
        Test convert dictionary with incorrect type to dmap and SDarnWrite
        write the rawacf file

        Behaviour: Raise SuperDARNDataFormatTypeError
        """
        rawacf_dict_data = copy.deepcopy(rawacf_dict_sets.rawacf_dict_data)
        rawacf_dict_data[0]['stid'] = np.int8(rawacf_dict_data[0]['stid'])
        dmap_rawacf = pydarnio.dict2dmap(rawacf_dict_data)
        darn_write = pydarnio.SDarnWrite(dmap_rawacf)
        with self.assertRaises(pydarnio.superdarn_exceptions.
                               SuperDARNDataFormatTypeError):
            darn_write.write_rawacf("test_rawacf.rawacf")

    def test_DmapWrite_incorrect_SDarnRead_rawacf_from_dict(self):
        """
        Test write an incorrect data type from a dict converting from dict2dmap
        with DmapWrite then SDarnRead reads the file

        Behaviour: Raises SuperDARNDataFormatTypeError
        """
        rawacf_dict_data = copy.deepcopy(rawacf_dict_sets.rawacf_dict_data)
        rawacf_dict_data[0]['stid'] = np.int8(rawacf_dict_data[0]['stid'])
        dmap_rawacf = pydarnio.dict2dmap(rawacf_dict_data)
        dmap_write = pydarnio.DmapWrite(dmap_rawacf)
        dmap_write.write_dmap("test_incorrect_rawacf.rawacf")

        darn_read = pydarnio.SDarnRead("test_incorrect_rawacf.rawacf")
        with self.assertRaises(pydarnio.superdarn_exceptions.
                               SuperDARNDataFormatTypeError):
            darn_read.read_rawacf()


    def test_DmapRead_SDarnWrite_SDarnRead_fitacf(self):
        """
        Test DmapRead reading a fitacf file then writing it with SDarnWrite
        then reading it again with SDarnRead
        """
        dmap = pydarnio.DmapRead(fitacf_file)
        dmap_data = dmap.read_records()
        dmap_write = pydarnio.SDarnWrite(dmap_data)
        dmap_write.write_fitacf("test_fitacf.fitacf")
        darn_read = pydarnio.SDarnRead("test_fitacf.fitacf")
        fitacf_data = darn_read.read_fitacf()
        self.dmap_compare(dmap_data, fitacf_data)
        os.remove("test_fitacf.fitacf")



    def test_DmapWrite_stream_SDarnRead_fitacf(self):
        """
        Test DmapWrite to write to a stream and have SDarnRead
        the fitacf stream
        """
        fitacf_data = copy.deepcopy(fitacf_data_sets.fitacf_data)
        fitacf_write = pydarnio.DmapWrite()
        fitacf_stream = fitacf_write.write_dmap_stream(fitacf_data)

        fitacf_read = pydarnio.SDarnRead(fitacf_stream, True)
        fitacf_read_data = fitacf_read.read_fitacf()
        self.dmap_compare(fitacf_read_data, fitacf_data)

    def test_DmapWrite_missing_SDarnRead_fitacf(self):
        """
        Test DmapWrite writes a fitacf file missing the field nave in record 2
        and SDarnRead reads the file

        Behaviour: Raise SuperDARNFieldMissingError
        """
        fitacf_missing_field = copy.deepcopy(fitacf_data_sets.fitacf_data)
        del fitacf_missing_field[0]['nave']
        dmap_write = pydarnio.DmapWrite(fitacf_missing_field)
        dmap_write.write_dmap("test_missing_fitacf.fitacf")

        darn_read = pydarnio.SDarnRead("test_missing_fitacf.fitacf")
        try:
            darn_read.read_fitacf()
        except pydarnio.superdarn_exceptions.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'nave'})
            self.assertEqual(err.record_number, 0)

        os.remove("test_missing_fitacf.fitacf")

    def test_DmapWrite_extra_SDarnRead_fitacf(self):
        """
        Test DmapWrite writes a fitacf file with an extra field and SDarnRead
        reads the file

        Behaviour: Raised SuperDARNExtraFieldError
        """

        fitacf_extra_field = copy.deepcopy(fitacf_data_sets.fitacf_data)
        fitacf_extra_field[0].update({'dummy': 'dummy'})
        fitacf_extra_field[0].move_to_end('dummy', last=False)
        dmap_write = pydarnio.DmapWrite(fitacf_extra_field, )
        dmap_write.write_dmap("test_extra_fitacf.fitacf")

        darn_read = pydarnio.SDarnRead("test_extra_fitacf.fitacf")
        try:
            darn_read.read_fitacf()
        except pydarnio.superdarn_exceptions.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 0)
        os.remove("test_extra_fitacf.fitacf")

    def test_DmapRead_SDarnWrite_SDarnRead_iqdat(self):
        """
        Test SDarnRead reads from a stream and SDarnWrite writes
        to a iqdat file
        """
        dmap = pydarnio.DmapRead(iqdat_file)
        dmap_data = dmap.read_records()
        dmap_write = pydarnio.SDarnWrite(dmap_data)
        dmap_write.write_iqdat("test_iqdat.iqdat")
        darn_read = pydarnio.SDarnRead("test_iqdat.iqdat")
        iqdat_data = darn_read.read_iqdat()
        self.dmap_compare(dmap_data, iqdat_data)
        os.remove("test_iqdat.iqdat")


    def test_DmapWrite_stream_SDarnRead_iqdat(self):
        """
        Test DmapWrite to write to a stream and have SDarnRead
        the iqdat stream
        """
        iqdat_data = copy.deepcopy(iqdat_data_sets.iqdat_data)
        iqdat_write = pydarnio.DmapWrite()
        iqdat_stream = iqdat_write.write_dmap_stream(iqdat_data)

        iqdat_read = pydarnio.SDarnRead(iqdat_stream, True)
        iqdat_read_data = iqdat_read.read_iqdat()
        self.dmap_compare(iqdat_read_data, iqdat_data)

    def test_DmapWrite_missing_SDarnRead_iqdat(self):
        """
        Test DmapWrite writes a iqdat file missing the field nave in record 2
        and SDarnRead reads the file

        Behaviour: Raise SuperDARNFieldMissingError
        """
        iqdat_missing_field = copy.deepcopy(iqdat_data_sets.iqdat_data)
        del iqdat_missing_field[0]['nave']
        dmap_write = pydarnio.DmapWrite(iqdat_missing_field)
        dmap_write.write_dmap("test_missing_iqdat.iqdat")

        darn_read = pydarnio.SDarnRead("test_missing_iqdat.iqdat")
        try:
            darn_read.read_iqdat()
        except pydarnio.superdarn_exceptions.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'nave'})
            self.assertEqual(err.record_number, 0)

        os.remove("test_missing_iqdat.iqdat")

    def test_DmapWrite_extra_SDarnRead_iqdat(self):
        """
        Test DmapWrite writes a iqdat file with an extra field and SDarnRead
        reads the file

        Behaviour: Raised SuperDARNExtraFieldError
        """
        iqdat_extra_field = copy.deepcopy(iqdat_data_sets.iqdat_data)
        iqdat_extra_field[0].update({'dummy': 'dummy'})
        iqdat_extra_field[0].move_to_end('dummy', last=False)
        dmap_write = pydarnio.DmapWrite(iqdat_extra_field, )
        dmap_write.write_dmap("test_extra_iqdat.iqdat")

        darn_read = pydarnio.SDarnRead("test_extra_iqdat.iqdat")
        try:
            darn_read.read_iqdat()
        except pydarnio.superdarn_exceptions.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 0)
        os.remove("test_extra_iqdat.iqdat")



    def test_DmapRead_SDarnWrite_SDarnRead_grid(self):
        """
        Test DmapRead reading a grid file then writing it with SDarnWrite
        then reading it again with SDarnRead
        """
        dmap = pydarnio.DmapRead(grid_file)
        dmap_data = dmap.read_records()
        dmap_write = pydarnio.SDarnWrite(dmap_data)
        dmap_write.write_grid("test_grid.grid")
        darn_read = pydarnio.SDarnRead("test_grid.grid")
        grid_data = darn_read.read_grid()
        self.dmap_compare(dmap_data, grid_data)
        os.remove("test_grid.grid")

    def test_DmapWrite_missing_SDarnRead_grid(self):
        """
        Test DmapWrite writes a grid file missing the field nave in record 2
        and SDarnRead reads the file

        Behaviour: Raise SuperDARNFieldMissingError
        """
        grid_missing_field = copy.deepcopy(grid_data_sets.grid_data)
        del grid_missing_field[0]['stid']
        dmap_write = pydarnio.DmapWrite(grid_missing_field)
        dmap_write.write_dmap("test_missing_grid.grid")

        darn_read = pydarnio.SDarnRead("test_missing_grid.grid")
        try:
            darn_read.read_grid()
        except pydarnio.superdarn_exceptions.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'stid'})
            self.assertEqual(err.record_number, 0)

        os.remove("test_missing_grid.grid")

    def test_DmapWrite_extra_SDarnRead_grid(self):
        """
        Test DmapWrite writes a grid file with an extra field and SDarnRead
        reads the file

        Behaviour: Raised SuperDARNExtraFieldError
        """
        grid_extra_field = copy.deepcopy(grid_data_sets.grid_data)
        grid_extra_field[0].update({'dummy': 'dummy'})
        grid_extra_field[0].move_to_end('dummy', last=False)
        dmap_write = pydarnio.DmapWrite(grid_extra_field, )
        dmap_write.write_dmap("test_extra_grid.grid")

        darn_read = pydarnio.SDarnRead("test_extra_grid.grid")
        try:
            darn_read.read_grid()
        except pydarnio.superdarn_exceptions.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 0)
        os.remove("test_extra_grid.grid")


    def test_DmapRead_SDarnWrite_SDarnRead_map(self):
        """
        Test DmapRead reading a map file then writing it with SDarnWrite
        then reading it again with SDarnRead
        """
        dmap = pydarnio.DmapRead(map_file)
        dmap_data = dmap.read_records()
        dmap_write = pydarnio.SDarnWrite(dmap_data)
        dmap_write.write_map("test_map.map")
        darn_read = pydarnio.SDarnRead("test_map.map")
        map_data = darn_read.read_map()
        self.dmap_compare(dmap_data, map_data)
        os.remove("test_map.map")

    def test_DmapWrite_stream_SDarnRead_map(self):
        """
        Test DmapWrite to write to a stream and have SDarnRead
        the map stream
        """
        map_data = copy.deepcopy(map_data_sets.map_data)
        map_write = pydarnio.DmapWrite()
        map_stream = map_write.write_dmap_stream(map_data)

        map_read = pydarnio.SDarnRead(map_stream, True)
        map_read_data = map_read.read_map()
        self.dmap_compare(map_read_data, map_data)

    def test_DmapWrite_missing_SDarnRead_map(self):
        """
        Test DmapWrite writes a fitacf file missing the field nave in record 2
        and SDarnRead reads the file

        Behaviour: Raise SuperDARNFieldMissingError
        """
        map_missing_field = copy.deepcopy(map_data_sets.map_data)
        del map_missing_field[0]['stid']
        dmap_write = pydarnio.DmapWrite(map_missing_field)
        dmap_write.write_dmap("test_missing_map.map")

        darn_read = pydarnio.SDarnRead("test_missing_map.map")
        try:
            darn_read.read_map()
        except pydarnio.superdarn_exceptions.SuperDARNFieldMissingError as err:
            self.assertEqual(err.fields, {'stid'})
            self.assertEqual(err.record_number, 0)

        os.remove("test_missing_map.map")

    def test_DmapWrite_extra_SDarnRead_map(self):
        """
        Test DmapWrite writes a map file with an extra field and SDarnRead
        reads the file

        Behaviour: Raised SuperDARNExtraFieldError
        """
        map_extra_field = copy.deepcopy(map_data_sets.map_data)
        map_extra_field[0].update({'dummy': 'dummy'})
        map_extra_field[0].move_to_end('dummy', last=False)
        dmap_write = pydarnio.DmapWrite(map_extra_field, )
        dmap_write.write_dmap("test_extra_map.map")

        darn_read = pydarnio.SDarnRead("test_extra_map.map")
        try:
            darn_read.read_map()
        except pydarnio.superdarn_exceptions.SuperDARNExtraFieldError as err:
            self.assertEqual(err.fields, {'dummy'})
            self.assertEqual(err.record_number, 0)
        os.remove("test_extra_map.map")
