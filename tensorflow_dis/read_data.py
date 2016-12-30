import collections
import csv
import numpy as np
import random
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv(filename, target_dtype, target_column=-1, has_header=True):
    with gfile.Open(filename) as csv_file:
	data_file = csv.reader(csv_file)
	data, target = [], []
	for ir in data_file:
	    target.append(ir.pop(target_column))
            data.append(ir)
        target = np.array(target).reshape(len(target),1)
        data = np.array(data).reshape(len(data),428)
    return Dataset(data=data, target=target)

#train_data = load_csv("/home/chuanxin.tcx/Test/Data/train_shuffled_part-00300", dtypes.float32, 0, False)
#print train_data.data
