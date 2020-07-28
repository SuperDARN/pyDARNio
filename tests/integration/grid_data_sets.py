# Copyright (C) 2019 SuperDARN
# Author: Marina Schmidt

"""
Test data sets for DmapWrite
"""
import numpy as np

from collections import OrderedDict

from pydarnio import DmapScalar, DmapArray

grid_data = \
    [OrderedDict([('start.year',
                   DmapScalar(name='start.year', value=2018, data_type=2, data_type_fmt='h')),
                  ('start.month', DmapScalar(name='start.month', value=2, data_type=2, data_type_fmt='h')),
                  ('start.day', DmapScalar(name='start.day', value=20, data_type=2, data_type_fmt='h')),
                  ('start.hour', DmapScalar(name='start.hour', value=0, data_type=2, data_type_fmt='h')),
                  ('start.minute', DmapScalar(name='start.minute', value=6, data_type=2, data_type_fmt='h')),
                  ('start.second', DmapScalar(name='start.second', value=0.0040569305419921875, data_type=8, data_type_fmt='d')),
                  ('end.year', DmapScalar(name='end.year', value=2018, data_type=2, data_type_fmt='h')),
                  ('end.month', DmapScalar(name='end.month', value=2, data_type=2, data_type_fmt='h')),
                  ('end.day', DmapScalar(name='end.day', value=20, data_type=2, data_type_fmt='h')),
                  ('end.hour', DmapScalar(name='end.hour', value=0, data_type=2, data_type_fmt='h')),
                  ('end.minute', DmapScalar(name='end.minute', value=8, data_type=2, data_type_fmt='h')),
                  ('end.second', DmapScalar(name='end.second', value=0.0040569305419921875, data_type=8, data_type_fmt='d')),
                  ('stid', DmapArray(name='stid', value=np.array([65], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('channel', DmapArray(name='channel', value=np.array([0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('nvec', DmapArray(name='nvec', value=np.array([45], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('freq', DmapArray(name='freq', value=np.array([11348.719], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('major.revision', DmapArray(name='major.revision', value=np.array([2], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('minor.revision', DmapArray(name='minor.revision', value=np.array([0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('program.id', DmapArray(name='program.id', value=np.array([3505], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('noise.mean', DmapArray(name='noise.mean', value=np.array([10.59375], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('noise.sd', DmapArray(name='noise.sd', value=np.array([1.5636357], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('gsct', DmapArray(name='gsct', value=np.array([1], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('v.min', DmapArray(name='v.min', value=np.array([35.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('v.max', DmapArray(name='v.max', value=np.array([2500.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('p.min', DmapArray(name='p.min', value=np.array([3.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('p.max', DmapArray(name='p.max', value=np.array([60.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('w.min', DmapArray(name='w.min', value=np.array([10.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('w.max', DmapArray(name='w.max', value=np.array([1000.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('ve.min', DmapArray(name='ve.min', value=np.array([0.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('ve.max', DmapArray(name='ve.max', value=np.array([200.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('vector.mlat', DmapArray(name='vector.mlat', value=np.array([72.5, 73.5, 74.5, 79.5, 79.5, 80.5, 73.5, 74.5, 80.5, 72.5, 79.5,
                   80.5, 82.5, 82.5, 83.5, 81.5, 84.5, 82.5, 80.5, 81.5, 79.5, 81.5,
                   73.5, 80.5, 83.5, 79.5, 80.5, 81.5, 82.5, 83.5, 79.5, 81.5, 72.5,
                   80.5, 78.5, 79.5, 80.5, 79.5, 80.5, 73.5, 74.5, 78.5, 79.5, 78.5,
                   79.5], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[45])),
                  ('vector.mlon', DmapArray(name='vector.mlon', value=np.array([335., 333.52942, 331.875, 319.0909, 313.63635,
                   308.1356, 337.05884, 335.625, 314.23727, 338.33334,
                   324.54544, 320.339, 310.21277, 317.87234, 311.7073,
                   329.43396, 313.7143, 333.1915, 338.64407, 336.2264,
                   340.9091, 343.01886, 340.58823, 344.74576, 346.82925,
                   346.36365, 350.84744, 349.8113, 356.17023, 355.60974,
                   351.81818, 356.60376, 341.66666, 356.94916, 352.5,
                   357.27274, 3.0508475, 2.7272727, 9.152542, 344.11765,
                   346.875, 2.5, 8.181818, 7.5, 13.636364],
                   dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[45])),
                  ('vector.kvect', DmapArray(name='vector.kvect', value=np.array([150.48602, 150.38976, 149.73271, -44.11724, -48.90261,
                   -52.720737, 168.25351, 159.23058, -44.17332, 174.83675,
                   149.24484, 145.81033, 133.7058, 143.56154, 139.32442,
                   166.77945, 143.30453, 175.14447, 179.95691, 175.5118,
                   -172.71599, -172.94543, -163.22404, -167.09595, -164.52101,
                   -161.34946, -155.4211, -159.47032, -153.30176, -153.61163,
                   -149.5504, -146.24246, -149.24686, -143.59334, -142.18036,
                   -138.36761, -132.77902, -129.27661, -123.71623, -143.01823,
                   -140.66672, -125.94862, -121.524704, -120.69907, -114.96321],
                   dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[45])),
                  ('vector.stid', DmapArray(name='vector.stid', value=np.array([65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                   65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                   65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[45])),
                  ('vector.channel', DmapArray(name='vector.channel', value=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[45])),
                  ('vector.index', DmapArray(name='vector.index', value=np.array([72100, 73094, 74088, 79058, 79057, 80050, 73095, 74089, 80051,
                   72101, 79059, 80052, 82040, 82041, 83035, 81048, 84030, 82043,
                   80055, 81049, 79062, 81050, 73096, 80056, 83039, 79063, 80057,
                   81051, 82046, 83040, 79064, 81052, 72102, 80058, 78070, 79065,
                   80000, 79000, 80001, 73097, 74092, 78000, 79001, 78001, 79002],
                   dtype=np.int32), data_type=3, data_type_fmt='i', dimension=1, shape=[45])),
                  ('vector.vel.median', DmapArray(name='vector.vel.median', value=np.array([211.40865, 287.75946, 230.77419, 187.97984, 257.93872,
                   260.47934, 178.85757, 232.12292, 237.11732, 157.37292,
                   122.41863, 100.79391, 46.440826, 54.64365, 61.027603,
                   90.12276, 114.797516, 106.21644, 172.39407, 132.68246,
                   205.91148, 207.2965, 116.40394, 219.84409, 223.87404,
                   234.68289, 242.02846, 253.77455, 248.37881, 231.41563,
                   215.56255, 250.5544, 103.61616, 230.06058, 220.61739,
                   232.87077, 236.41026, 244.11392, 243.74884, 66.45725,
                   60.761314, 239.27852, 245.77727, 253.88075, 260.77255],
                   dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[45])),
                  ('vector.vel.sd', DmapArray(name='vector.vel.sd', value=np.array([31.622776, 35.35534, 35.35534, 63.331505, 40.82483,
                   50., 17.149858, 25.897175, 57.988403, 21.821789,
                   53.313816, 130.36421, 57.735027, 50., 50.,
                   50., 70.71068, 50., 28.867514, 50.,
                   44.72136, 40.82483, 18.898224, 44.72136, 44.72136,
                   40.82483, 35.35534, 70.71068, 57.735027, 70.71068,
                   50., 50., 30.151134, 28.867514, 57.735027,
                   35.35534, 35.35534, 31.622776, 50., 40.82483,
                   50., 44.72136, 40.82483, 50., 44.72136],
                   dtype=np.float32), data_type=4, data_type_fmt='f',
                                              dimension=1, shape=[45]))]),
     OrderedDict([('start.year', DmapScalar(name='start.year', value=2018, data_type=2, data_type_fmt='h')),
                  ('start.month', DmapScalar(name='start.month', value=2, data_type=2, data_type_fmt='h')),
                  ('start.day', DmapScalar(name='start.day', value=20, data_type=2, data_type_fmt='h')),
                  ('start.hour', DmapScalar(name='start.hour', value=13, data_type=2, data_type_fmt='h')),
                  ('start.minute', DmapScalar(name='start.minute', value=23, data_type=2, data_type_fmt='h')),
                  ('start.second', DmapScalar(name='start.second', value=0.011981964111328125, data_type=8, data_type_fmt='d')),
                  ('end.year', DmapScalar(name='end.year', value=2018, data_type=2, data_type_fmt='h')),
                  ('end.month', DmapScalar(name='end.month', value=2, data_type=2, data_type_fmt='h')),
                  ('end.day', DmapScalar(name='end.day', value=20, data_type=2, data_type_fmt='h')),
                  ('end.hour', DmapScalar(name='end.hour', value=13, data_type=2, data_type_fmt='h')),
                  ('end.minute', DmapScalar(name='end.minute', value=25, data_type=2, data_type_fmt='h')),
                  ('end.second', DmapScalar(name='end.second', value=0.011981964111328125, data_type=8, data_type_fmt='d')),
                  ('stid', DmapArray(name='stid', value=np.array([65], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('channel', DmapArray(name='channel', value=np.array([0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('nvec', DmapArray(name='nvec', value=np.array([32], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('freq', DmapArray(name='freq', value=np.array([11355.719], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('major.revision', DmapArray(name='major.revision', value=np.array([2], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('minor.revision', DmapArray(name='minor.revision', value=np.array([0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('program.id', DmapArray(name='program.id', value=np.array([3505], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('noise.mean', DmapArray(name='noise.mean', value=np.array([14.8125], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('noise.sd', DmapArray(name='noise.sd', value=np.array([5.087161], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('gsct', DmapArray(name='gsct', value=np.array([1], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('v.min', DmapArray(name='v.min', value=np.array([35.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('v.max', DmapArray(name='v.max', value=np.array([2500.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('p.min', DmapArray(name='p.min', value=np.array([3.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('p.max', DmapArray(name='p.max', value=np.array([60.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('w.min', DmapArray(name='w.min', value=np.array([10.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('w.max', DmapArray(name='w.max', value=np.array([1000.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('ve.min', DmapArray(name='ve.min', value=np.array([0.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('ve.max', DmapArray(name='ve.max', value=np.array([200.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('vector.mlat', DmapArray(name='vector.mlat', value=np.array([75.5, 76.5, 77.5, 79.5, 74.5, 75.5, 76.5, 77.5, 83.5, 84.5, 84.5,
                   85.5, 76.5, 77.5, 86.5, 86.5, 75.5, 74.5, 76.5, 75.5, 74.5, 78.5,
                   75.5, 77.5, 76.5, 78.5, 77.5, 73.5, 74.5, 75.5, 76.5, 77.5],
                   dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[32])),
                  ('vector.mlon', DmapArray(name='vector.mlon', value=np.array([330., 327.85715, 325.3846, 313.63635, 335.625, 334.,
                   332.14285, 330., 267.80487, 272.57144, 262.2857, 263.57144,
                   336.42856, 334.6154, 270., 253.63637, 338., 339.375,
                   340.7143, 342., 343.125, 347.5, 346., 348.46155,
                   349.2857, 352.5, 353.07693, 344.11765, 346.875, 350.,
                   353.57144, 357.69232], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[32])),
                  ('vector.kvect', DmapArray(name='vector.kvect', value=np.array([148.69537, 141.54613, 142.8571, 131.84319, 170.03053,
                   160.40012, 157.37341, 89.2536, -93.46017, -87.30513,
                   -92.89313, -85.86758, 8.765994, -11.932242, -77.360016,
                   -86.02396, 141.57924, 17.019497, 6.932431, 17.441008,
                   25.992363, 27.482807, 32.788364, 31.261826, 36.400013,
                   37.104523, 41.330643, 37.179756, 38.110252, 41.75692,
                   45.872223, 49.977745], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[32])),
                  ('vector.stid', DmapArray(name='vector.stid', value=np.array([65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                   65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65],
                   dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[32])),
                  ('vector.channel', DmapArray(name='vector.channel', value=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[32])),
                  ('vector.index', DmapArray(name='vector.index', value=np.array([75082, 76076, 77070, 79057, 74089, 75083, 76077, 77071, 83030,
                   84026, 84025, 85020, 76078, 77072, 86016, 86015, 75084, 74090,
                   76079, 75085, 74091, 78069, 75086, 77075, 76081, 78070, 77076,
                   73097, 74092, 75087, 76082, 77077], dtype=np.int32), data_type=3, data_type_fmt='i', dimension=1, shape=[32])),
                  ('vector.vel.median', DmapArray(name='vector.vel.median', value=np.array([86.22557, 55.99529, 77.1204, 138.94217, 53.439465,
                   83.13799, 65.69839, 8.872497, 140.93784, 136.43436,
                   125.74353, 107.80217, 10.187439, 137.04274, 118.45908,
                   98.89083, 5.5379543, 81.491165, 50.020416, 95.97311,
                   182.38617, 154.7921, 158.80106, 146.97353, 85.35316,
                   205.68631, 121.89034, 218.41821, 200.13217, 186.16425,
                   74.23934, 202.2707], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[32])),
                  ('vector.vel.sd', DmapArray(name='vector.vel.sd', value=np.array([31.622776, 45.96998, 57.735027, 70.71068, 22.96239, 25.858837,
                   33.51713, 53.530884, 70.71068, 70.71068, 50., 40.82483,
                   23.570227, 31.957285, 57.735027, 70.71068, 22.566723, 27.893995,
                   23.570227, 26.726124, 22.094053, 74.42324, 22.941574, 37.93341,
                   27.748476, 49.585766, 31.773098, 58.097965, 31.622776, 35.35534,
                   31.622776, 34.513107], dtype=np.float32), data_type=4,
                                              data_type_fmt='f', dimension=1,
                                              shape=[32]))]),
     OrderedDict([('start.year', DmapScalar(name='start.year', value=2018, data_type=2, data_type_fmt='h')),
                  ('start.month', DmapScalar(name='start.month', value=2, data_type=2, data_type_fmt='h')),
                  ('start.day', DmapScalar(name='start.day', value=20, data_type=2, data_type_fmt='h')),
                  ('start.hour', DmapScalar(name='start.hour', value=23, data_type=2, data_type_fmt='h')),
                  ('start.minute', DmapScalar(name='start.minute', value=50, data_type=2, data_type_fmt='h')),
                  ('start.second', DmapScalar(name='start.second', value=0.025077104568481445, data_type=8, data_type_fmt='d')),
                  ('end.year', DmapScalar(name='end.year', value=2018, data_type=2, data_type_fmt='h')),
                  ('end.month', DmapScalar(name='end.month', value=2, data_type=2, data_type_fmt='h')),
                  ('end.day', DmapScalar(name='end.day', value=20, data_type=2, data_type_fmt='h')),
                  ('end.hour', DmapScalar(name='end.hour', value=23, data_type=2, data_type_fmt='h')),
                  ('end.minute', DmapScalar(name='end.minute', value=52, data_type=2, data_type_fmt='h')),
                  ('end.second', DmapScalar(name='end.second', value=0.025077104568481445, data_type=8, data_type_fmt='d')),
                  ('stid', DmapArray(name='stid', value=np.array([65], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('channel', DmapArray(name='channel', value=np.array([0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('nvec', DmapArray(name='nvec', value=np.array([48], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('freq', DmapArray(name='freq', value=np.array([11333.656], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('major.revision', DmapArray(name='major.revision', value=np.array([2], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('minor.revision', DmapArray(name='minor.revision', value=np.array([0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('program.id', DmapArray(name='program.id', value=np.array([3505], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('noise.mean', DmapArray(name='noise.mean', value=np.array([12.53125], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('noise.sd', DmapArray(name='noise.sd', value=np.array([1.5820909], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('gsct', DmapArray(name='gsct', value=np.array([1], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[1])),
                  ('v.min', DmapArray(name='v.min', value=np.array([35.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('v.max', DmapArray(name='v.max', value=np.array([2500.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('p.min', DmapArray(name='p.min', value=np.array([3.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('p.max', DmapArray(name='p.max', value=np.array([60.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('w.min', DmapArray(name='w.min', value=np.array([10.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('w.max', DmapArray(name='w.max', value=np.array([1000.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('ve.min', DmapArray(name='ve.min', value=np.array([0.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('ve.max', DmapArray(name='ve.max', value=np.array([200.], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[1])),
                  ('vector.mlat', DmapArray(name='vector.mlat', value=np.array([79.5, 79.5, 80.5, 79.5, 80.5, 81.5, 80.5, 81.5, 82.5, 80.5, 81.5,
                   82.5, 82.5, 83.5, 84.5, 80.5, 81.5, 83.5, 84.5, 81.5, 82.5, 84.5,
                   82.5, 83.5, 84.5, 80.5, 81.5, 82.5, 83.5, 84.5, 81.5, 82.5, 83.5,
                   80.5, 81.5, 82.5, 83.5, 80.5, 81.5, 82.5, 80.5, 81.5, 81.5, 79.5,
                   80.5, 80.5, 79.5, 79.5], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[48])),
                  ('vector.mlon', DmapArray(name='vector.mlon', value=np.array([319.0909, 313.63635, 314.23727, 324.54544, 320.339,
                   315.84906, 326.44067, 322.6415, 317.87234, 332.54236,
                   329.43396, 325.53192, 333.1915, 329.26828, 324.,
                   338.64407, 336.2264, 338.04877, 334.2857, 343.01886,
                   340.85107, 344.57144, 348.51065, 346.82925, 354.85715,
                   350.84744, 349.8113, 356.17023, 355.60974, 5.142857,
                   356.60376, 3.8297873, 4.390244, 356.94916, 3.3962264,
                   11.489362, 13.170732, 3.0508475, 10.18868, 19.148935,
                   9.152542, 16.981133, 23.773584, 8.181818, 15.254237,
                   21.355932, 13.636364, 19.09091], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[48])),
                  ('vector.kvect', DmapArray(name='vector.kvect', value=np.array([1.3776143e+02, 1.3067482e+02, 1.3548923e+02, 1.5051083e+02,
                   1.4419740e+02, 1.4045711e+02, 1.5574120e+02, 1.5187080e+02,
                   1.4506064e+02, 1.6567052e+02, 1.6373940e+02, 1.5515234e+02,
                   -5.0249748e+00, -1.6617077e+01, -1.8794113e+01, 1.7961696e+02,
                   1.7558623e+02, -8.2697503e-02, -6.7042108e+00, 9.9063320e+00,
                   4.3262248e+00, 6.0504212e+00, 1.6758739e+01, 1.5605538e+01,
                   2.2159294e+01, 2.5681110e+01, 2.0525904e+01, 2.8494555e+01,
                   2.7147511e+01, 3.3622570e+01, 3.1760328e+01, 4.1792263e+01,
                   3.9918575e+01, 3.5687832e+01, 4.3404911e+01, 5.2630638e+01,
                   5.2512428e+01, 4.8060982e+01, 5.4476559e+01, 6.0645481e+01,
                   5.5574818e+01, 6.2865593e+01, 7.1008202e+01, 5.8256989e+01,
                   6.3870285e+01, 7.1036896e+01, 6.4951691e+01, 7.0386543e+01],
                   dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[48])),
                  ('vector.stid', DmapArray(name='vector.stid', value=np.array([65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                   65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
                   65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65],
                   dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[48])),
                  ('vector.channel', DmapArray(name='vector.channel', value=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0], dtype=np.int16), data_type=2, data_type_fmt='h', dimension=1, shape=[48])),
                  ('vector.index', DmapArray(name='vector.index', value=np.array([79058, 79057, 80051, 79059, 80052, 81046, 80053, 81047, 82041,
                   80054, 81048, 82042, 82043, 83037, 84031, 80055, 81049, 83038,
                   84032, 81050, 82044, 84033, 82045, 83039, 84034, 80057, 81051,
                   82046, 83040, 84000, 81052, 82000, 83000, 80058, 81000, 82001,
                   83001, 80000, 81001, 82002, 80001, 81002, 81003, 79001, 80002,
                   80003, 79002, 79003], dtype=np.int32), data_type=3, data_type_fmt='i', dimension=1, shape=[48])),
                  ('vector.vel.median', DmapArray(name='vector.vel.median', value=np.array([168.87584, 378.16104, 390.48422, 264.72037, 340.24277,
                   261.96246, 166.4799, 175.58917, 170.59193, 100.241035,
                   117.02003, 113.83192, 110.63128, 124.69313, 85.883835,
                   59.58456, 29.183453, 149.87598, 87.57454, 106.41398,
                   144.1112, 100.86157, 209.35103, 177.68849, 153.74188,
                   261.6218, 219.8409, 224.89569, 188.59256, 165.06386,
                   293.19037, 210.94766, 194.56616, 271.47748, 303.23712,
                   199.45433, 204.04663, 253.03906, 293.68582, 228.18443,
                   280.49426, 255.60298, 223.19438, 196.14743, 268.67493,
                   253.27917, 202.5746, 262.70676], dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[48])),
                  ('vector.vel.sd', DmapArray(name='vector.vel.sd', value=np.array([65.60137, 56.20039, 39.431385, 52.887394, 69.346954, 43.638535,
                   52.373154, 39.7046, 39.206783, 44.322094, 31.622776, 39.10426,
                   42.038414, 50., 44.72136, 35.902393, 50.005493, 28.867514,
                   40.82483, 32.031635, 40.82483, 40.82483, 40.82483, 40.82483,
                   35.35534, 48.513336, 41.15344, 35.35534, 40.82483, 44.72136,
                   35.531113, 35.35534, 35.35534, 36.263626, 35.35534, 40.82483,
                   40.82483, 40.907024, 35.35534, 40.82483, 40.82483, 40.82483,
                   40.82483, 40.82483, 50., 35.35534, 44.72136, 57.735027],
                   dtype=np.float32), data_type=4, data_type_fmt='f', dimension=1, shape=[48]))])]
