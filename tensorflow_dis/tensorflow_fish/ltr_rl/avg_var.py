
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import random

n_feature = 63

src_file = sys.argv[1]

def mv_avg(avg_,num_,v):
 return(avg_+(v-avg_)/(num_+1)) 

def avg(data_file):
  avg_ctr = 0.0
  avg_ctr2 = 0.0
  ctr_num = 0.0
  avg_cvr = 0.0
  avg_cvr2 = 0.0
  cvr_num = 0.0
  avg_bid = 0.0
  avg_bid2 = 0.0
  bid_num = 0.0
  avg_escore = 0.0
  avg_escore2 = 0.0
  escore_num = 0.0
  with open(data_file) as f:
    for line in f.readlines():
      strs = line.strip().split(",")
      if(strs[2].strip() == ""):
        continue
      for ad in strs[2].split(":"):
        ss = ad.split("_")
        ctr = float(ss[0])
        cvr = float(ss[1])
        bid = int(ss[2])
        escore = float(ss[4])
        avg_ctr = mv_avg(avg_ctr,ctr_num,ctr)
        avg_ctr2 = mv_avg(avg_ctr2,ctr_num,ctr * ctr)
	ctr_num += 1
        avg_cvr = mv_avg(avg_cvr,cvr_num,cvr)
        avg_cvr2 = mv_avg(avg_cvr2,cvr_num,cvr * cvr)
	cvr_num += 1
        avg_bid = mv_avg(avg_bid,bid_num,bid)
        avg_bid2 = mv_avg(avg_bid2,bid_num,bid * bid)
	bid_num += 1
        avg_escore = mv_avg(avg_escore,escore_num,escore)
        avg_escore2 = mv_avg(avg_escore2,escore_num,escore * escore)
	escore_num += 1
  return(avg_ctr,avg_ctr2 - avg_ctr * avg_ctr,avg_cvr, avg_cvr2 - avg_cvr * avg_cvr ,avg_bid, avg_bid2 - avg_bid * avg_bid , avg_escore, avg_escore2 - avg_escore * avg_escore)

print(avg("data/test"))
