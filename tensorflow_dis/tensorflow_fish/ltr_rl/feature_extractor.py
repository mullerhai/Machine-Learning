from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random
import numpy as np

# number of ad feature:ctr,cvr,max_price,match_type,escore
ad_feature = 5
# number of feature after feature extract
n_feature = 63

class LTRData(object):
  def __init__(self,data_file, mean_var_file):

    with open(mean_var_file) as f:
      (self.ctr_mean,self.ctr_var,self.cvr_mean,self.cvr_var,self.bid_mean,self.bid_var,self.escore_mean,self.escore_var) = (float(s) for s in f.readline().strip().split(","))

    self.data_file = data_file
    f = open(data_file)
    # number of data
    self.n_sample = int(f.readline())
    self.x = np.ndarray(shape=(self.n_sample,n_feature),dtype=np.float)
    self.label = np.ndarray(shape=(self.n_sample),dtype=np.float)
    self.weight = np.ndarray(shape=(self.n_sample),dtype=np.float)
    self.ads = np.ndarray(shape=(self.n_sample,ad_feature),dtype=np.float)
    self.batch_idx = 0
    idx = 0
    for line in f.readlines():
      ss = line.strip().split(",")
      if len(ss) !=4:
        print(line)
        continue
      self.label[idx] = float(ss[0])
      self.weight[idx] = float(ss[1])
      ad_ss = ss[2].split("_")
      if len(ad_ss) != ad_feature:
        print(line)
        continue
      for i in range(0,ad_feature):
        self.ads[idx][i] = float(ad_ss[i])
      ad_list = [ [float(ad) for ad in ad_str.split("_")] for ad_str in ss[3].split(":")]
      self._feature_extract(ad_list, idx)
      idx += 1
    f.close()

  # batch size of (label, weight, ad, ad_set)
  def next_batch(self,batch_size):
    if batch_size > self.n_sample:
      raise Exception("Batch size is too big.")
    else:
      self.batch_idx = (self.batch_idx + batch_size) % self.n_sample
      start = self.batch_idx - batch_size
      end = self.batch_idx
      if start < 0:
        label = np.concatenate((self.label[start:self.n_sample],self.label[0:end]))
        weight = np.concatenate((self.weight[start:self.n_sample],self.weight[0:end]))
        ads = np.concatenate((self.ads[start:self.n_sample],self.ads[0:end]))
        x = np.concatenate((self.x[start:self.n_sample],self.x[0:end]))
      else:
        label = self.label[start:end]
        weight = self.weight[start:end]
        ads = self.ads[start:end]
        x = self.x[start:end]
      return (label,weight,ads,x)
    

  def _feature_extract(self, ad_list, idx):
    truncated = 4
    feature = []
    tmp_list = sorted(ad_list,key = lambda ad:ad[0])
    feature.append([ tmp for tmp in ])
    self.x[idx,0:truncated] = [] 
    pass
    

data = LTRData("data/toy_sample","data/avg_var")
#data.next_batch(2)
#data.next_batch(2)
