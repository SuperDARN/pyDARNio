# Copyright (C) 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marina Schmidt


import unittest
import numpy as np
from collections import OrderedDict

import pydarnio
from pydarnio import DmapScalar, DmapArray


class Test_Conversions(unittest.TestCase):
    """
    Class to test the conversion functions
    """
    def setUp(self):
        """
        Creates the testing data

        Attributes
        ----------
        dmap_list : List[dict]
            List of dictionaries containing fields and values
        dmap_records : List[dict]
            List of ordered dictionaries containing dmap data structure
            DmapScalar and DmapArray
        """
        self.dmap_list = [{'stid': 1, 'channel': 0,
                           'ptab': np.array([0, 9, 12, 20, 22, 26, 27],
                                            dtype=np.int64)},
                          {'bmnum': np.int16(15), 'combf': "$Id: twofsound",
                           'pwr0': np.array([58.081821, 52.241421, 32.936508,
                                             35.562561, 35.344330, 31.501854,
                                             25.313326, 13.731517, 3.482957,
                                             -5.032664, -9.496454, 3.254651],
                                            dtype=np.float32)},
                          {'radar.revision.major': np.int8(1),
                           'radar.revision.minor': np.int8(18),
                           'float test': float(3.5),
                           'float2 test': 3.65,
                           'channel': 'a',
                           'double test': np.array([[2.305015, 2.0251],
                                                   [16548548, 78687686]],
                                                   dtype=np.float64)},
                          {'time.us': 508473,
                           'negative int': -42,
                           'long int': np.int64(215610516132151613),
                           'unsigned char': np.uint8(3),
                           'unsigned short': np.uint16(45),
                           'unsigned int': np.uint32(100),
                           'unsigned long': np.uint64(1250000000000),
                           'list test': [np.int64(1), np.int64(2),
                                         np.int64(34), np.int64(45)]}]
        self.dmap_records = \
            [OrderedDict([('stid', DmapScalar('stid', 1, 3, 'i')),
                          ('channel', DmapScalar('channel', 0, 3, 'i')),
                          ('ptab', DmapArray('ptab',
                                             np.array([0, 9, 12, 20, 22, 26,
                                                       27], dtype=np.int64),
                                             10, 'q', 1, [7]))]),
             OrderedDict([('bmnum', DmapScalar('bmnum', 15, 2, 'h')),
                          ('combf', DmapScalar('combf', "$Id: twofsound", 9,
                                               's')),
                          ('pwr0', DmapArray('pwr0',
                                             np.array([58.081821, 52.241421,
                                                       32.936508, 35.562561,
                                                       35.344330, 31.501854,
                                                       25.313326, 13.731517,
                                                       3.482957, -5.032664,
                                                       -9.496454, 3.254651],
                                                      dtype=np.float32),
                                             4, 'f', 1, [12]))]),
             OrderedDict([('radar.revision.major',
                           DmapScalar('radar.revision.major', np.int8(1),
                                      1, 'c')),
                          ('radar.revision.minor',
                           DmapScalar('radar.revision.minor', np.int8(18),
                                      1, 'c')),
                          ('float test',
                           DmapScalar('float test', float(3.5), 4, 'f')),
                          ('float2 test',
                           DmapScalar('float2 test', 3.65, 4, 'f')),
                          ('channel', DmapScalar('channel', 'a', 9, 's')),
                          ('double test',
                           DmapArray('double test',
                                     np.array([[2.305015, 2.0251],
                                               [16548548, 78687686]],
                                              dtype=np.float64), 8,
                                     'd', 2, [2, 2]))]),
             OrderedDict([('time.us', DmapScalar('time.us', 508473, 3, 'i')),
                          ('negative int',
                           DmapScalar('negative int', -42, 3, 'i')),
                          ('long int',
                           DmapScalar('long int',
                                      np.int64(215610516132151613), 10, 'q')),
                          ('unsigned char',
                           DmapScalar('unsigned char', np.uint8(3), 16, 'B')),
                          ('unsigned short',
                           DmapScalar('unsigned short', np.uint16(45),
                                      17, 'H')),
                          ('unsigned int',
                           DmapScalar('unsigned int', np.uint32(100),
                                      18, 'I')),
                          ('unsigned long',
                           DmapScalar('unsigned long',
                                      np.uint64(1250000000000), 19, 'Q')),
                          ('list test',
                           DmapArray('list test', np.array([1, 2, 34, 45],
                                                           dtype=np.int64),
                                     10, 'q', 1, [4]))])]
        self.array_attrs = ['name', 'data_type', 'data_type_fmt', 'dimension']

    def tearDown(self):
        """ Clean up the testing environment
        """
        del self.dmap_list, self.dmap_records, self.array_attrs

    def dmap_compare(self, dmap1: list):
        """
        Evaluate equivalency of an input dmap data structure to dmap_records
        """
        # Quick simple tests that can be done before looping
        # over the list
        self.assertEqual(len(dmap1), len(self.dmap_records))

        # NamedTuple are comparison capabilities
        for record1, record2 in zip(dmap1, self.dmap_records):
            self.assertSetEqual(set(record1), set(record2))
            for field, val_obj in record1.items():
                if isinstance(val_obj, DmapScalar):
                    self.compare_dmap_attrs(val_obj, record2[field],
                                            self.array_attrs[:-1])
                    self.assertEqual(val_obj, record2[field])
                else:
                    self.compare_dmap_attrs(val_obj, record2[field],
                                            self.array_attrs)
                    self.assertTrue(np.array_equal(val_obj.value,
                                                   record2[field].value))

    def compare_dmap_attrs(self, dmaparr1, dmaparr2, attr_list):
        """
        Evaluates equivalency of DmapArray array attributes
        """
        for val in attr_list:
            with self.subTest(val=val):
                # Ensure each array has the desired attribute
                self.assertTrue(hasattr(dmaparr1, val))
                self.assertTrue(hasattr(dmaparr2, val))

                # Ensure the attributes in each array are the same
                self.assertEqual(getattr(dmaparr1, val),
                                 getattr(dmaparr2, val))

    def test_dict2dmap(self):
        """
        From utils package, testing dict2dmap function
        """
        dmap_records_test = pydarnio.dict2dmap(self.dmap_list)
        self.dmap_compare(dmap_records_test)

    def test_dmap2dict(self):
        """
        From utils package, testing dmap2dict function
        """
        # need to break up the list of dictionaries to properly
        # compare each field value
        dmap_list_test = pydarnio.dmap2dict(self.dmap_records)
        for j in range(len(dmap_list_test)):
            for key, value in dmap_list_test[j].items():
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.array_equal(value,
                                                   self.dmap_list[j][key]))
                else:
                    self.assertEqual(value, self.dmap_list[j][key])
