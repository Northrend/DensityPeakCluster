#! /usr/bin/env python
#-*- coding: utf-8 -*-
#
# data reference : R. A. Fisher (1936). "The use of multiple measurements in taxonomic problems"

import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'distance/'))
from distance_builder import *
from distance import * 

import numpy as np

if __name__ == '__main__':
  builder = DistanceBuilder(500)
  builder.load_points_from_npy(sys.argv[1])
  # builder.build_distance_file_for_cluster(SqrtDistance(), r'../data/data_others/aggregation_distance.dat')
  builder.build_distance_file_for_cluster(distance.SqrtDistance(), sys.argv[2])
  
