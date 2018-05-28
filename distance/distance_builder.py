#! /usr/bin/env python
#-*- coding: utf-8 -*-

import time
import math

from distance import *

import mxnet as mx
import numpy as np

class DistanceBuilder(object):

  """
  Build distance file for cluster
  """  

  def __init__(self, batch_size=500):
    self.vectors = []
    self.batch_size = batch_size 

  def load_points(self, filename):
    '''
    Load all points from file(x dimension vectors)

    Args:
        filename : file's name that contains all points. Format is a vector one line, each dimension value split by blank space
    '''
    with open(filename, 'r') as fp:
      for line in fp:
        self.vectors.append(np.array(map(float, line.split('\t')[:-1]), dtype = np.float32))
    self.vectors = np.array(self.vectors, dtype = np.float32)


  def load_points_from_npy(self, filename):
    '''
    '''
    with open(filename, 'r') as fp:
      self.vectors = np.load(fp)
    self.vectors = np.array(self.vectors, dtype = np.float32)


  def build_distance_file_for_cluster(self, distance_obj, filename):
    '''
    Save distance and index into file

    Args:
        distance_obj : distance.Distance object for compute the distance of two point
        filename     : file to save the result for cluster
    '''
    def _double_loops(vec, distance_obj, filename):
      fo = open(filename, 'w')
      for i in xrange(len(vec) - 1):
        for j in xrange(i, len(vec)):
          fo.write(str(i + 1) + ' ' + str(j + 1) + ' ' + str(distance_obj.distance(vec[i], vec[j])) + '\n')
      fo.close()
    
    def _vectorized(vec, distance_obj, filename):
      with open(filename, 'w') as f:
        d_n, d_f = vec.shape
        # x = np.matmul(np.ones((d_f,d_n,1)), vec.reshape(d_f,1,d_n))
        # y = np.matmul(vec.reshape(d_f,d_n,1), np.ones((d_f,1,d_n)))
        # x = vec.reshape(d_n, 1, d_f)
        # y = vec.reshape(1, d_n, d_f)
        x = np.broadcast_arrays(vec.reshape(d_n, 1, d_f), np.zeros((d_n, d_n, d_f)))[0]
        y = np.broadcast_arrays(vec.reshape(1, d_n, d_f), np.zeros((d_n, d_n, d_f)))[0]
        s = np.linalg.norm((x-y), axis=2)
        print(x.shape,y.shape,s.shape)

    def _vectorized_gpu(vec, distance_obj, filename, batch_size):
      with open(filename, 'w') as f:
        d_n, d_f = vec.shape
        d_p = batch_size 
        x = np.broadcast_arrays(vec.reshape(d_n, 1, d_f), np.zeros((d_n, d_n, d_f)))[0]
        y = np.broadcast_arrays(vec.reshape(1, d_n, d_f), np.zeros((d_n, d_n, d_f)))[0]
        # x = mx.nd.array(x, mx.gpu(0))
        # y = mx.nd.array(y, mx.gpu(0))
        diff = x-y
        res = mx.nd.zeros((d_n, d_n))
        for i in range(int(d_n/d_p)-1):
          _ = mx.nd.array(diff[i*d_p:(i+1)*d_p], mx.gpu(0))
          # print(_.shape)
          res[i*d_p:(i+1)*d_p,] = mx.nd.norm(_, axis=2)
        _ = mx.nd.array(diff[(int(d_n/d_p)-1)*d_p:], mx.gpu(0))
        # print(_.shape)
        res[(int(d_n/d_p)-1)*d_p:,] = mx.nd.norm(_, axis=2)
        

    tic = time.time()
    # _double_loops(self.vectors, distance_obj, filename)
    # _vectorized(self.vectors, distance_obj, filename)
    _vectorized_gpu(self.vectors.astype(np.float16), distance_obj, filename, self.batch_size)
    print("cost: {:.6f}s".format(time.time()-tic))
#end DistanceBuilder
