
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random

src_file_path = sys.argv[1]
test_file = open(src_file_path + "_test","w")
train_file = open(src_file_path + "_train","w")
test_rate = float(sys.argv[2])

with open(src_file_path) as f:
  for line in f.readlines():
    if(random.random()<test_rate):
      test_file.write(line)
    else:
      train_file.write(line)

test_file.close()
train_file.close()

