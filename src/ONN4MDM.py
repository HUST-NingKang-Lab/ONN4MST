#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import math
import json
import pickle
from graph_builder import model
from utils import *



#读取数据，label按倒数第一列进行拼接，拼接完成后，label是一个m*n的张量，m是样本数量，n是label总数
def npzload(ifn):
  data = np.load(ifn)
  feature = data['matrices']
  label0,label1,label2,label3,label4,label5 = data['label_0'],data['label_1'],data['label_2'],data['label_3'],data['label_4'],data['label_5']
  return(feature,label0,label1,label2,label3,label4,label5)

def npzload1(ifn):
  data = np.load(ifn)
  feature = data['matrices']
  return(feature)

def get_feature_size(feature):
  size = len(feature[0])
  return(size)
  
def get_label_size(labels):
  size = []
  for i in range(len(labels)):
    nodes_count = len(labels[i][0])
    size.append(nodes_count)
  return(size)

def loading_model(fn,Model):
  matrices = npzload1(fn)
  samplenum = len(matrices)
  matrices = matrices.reshape(samplenum,-1)

  #sample a batch
  N = len(matrices)
  all_batch = []
  for i in range(N):
    all_batch.append(i)
  all_batch = np.array(all_batch)

  #feed the feature to our model
  feed = {Model.x: matrices[all_batch]}
  y_pred = Model.sess.run([Model.y_pred], feed)
  return(y_pred)

def loading_biome_tree(tree):
  matrices,label0,label1,label2,label3,label4,label5 = npzload(tree)
  samplenum = len(matrices)
  matrices = matrices.reshape(samplenum,-1)
  labels = []
  labels.append(label0)
  labels.append(label1)
  labels.append(label2)
  labels.append(label3)
  labels.append(label4)
  labels.append(label5)
  return(matrices,labels)

def label_process(y_pred,labels_size):
  treeidx = [0]
  for i in range(len(labels_size)):
    idx = 0
    for j in range(i+1):
      idx += labels_size[j]
    treeidx.append(idx)
  #treeidx = [0,4,8,22,76,130,152]
  for i in range(len(y_pred)):
    for j in range(len(labels_size)):
      layerj = y_pred[i][treeidx[j]:treeidx[j+1]]
      if(np.all(layerj == 0)):
        print(treeidx[-1])
        print("this is y_pred", y_pred[i])
        y_pred[i][treeidx[j]:treeidx[-1]] = 0
        print("this is modified y_pred", y_pred[i])
        print("this is i", i)
        break
  return(y_pred)

def main():
  #get args
  parser = get_parser()
  args = parser.parse_args()
  gpus = args.gpus
  tree = args.tree
  ifn = args.ifn
  ofn = args.ofn
  name = args.name
  mdl = args.model
  threshold = args.threshold

  #gpu devices
  if(gpus == 1):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
    print('Using gpu device for predicting!')
  elif(gpus == 0):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    print('Using cpu device for predicting!')
    #config = tf.ConfigProto(device_count={"CPU": cpu_num}, inter_op_parallelism_threads = cpu_num, intra_op_parallelism_threads = cpu_num, log_device_placement=True)

  #loading biome tree
  matrices,labels = loading_biome_tree(tree)
  labels_size = get_label_size(labels)
  matrices_size = get_feature_size(matrices)

  #loading trained model
  Model = model(feature_size = matrices_size, label_size = labels_size, gpu_mode = gpus)
  Model.load_json(mdl)

  #loading unkonwn biome samples and performing prediction
  y_pred = loading_model(ifn,Model)[0]
  y_pred[y_pred < threshold]=0
  y_pred[y_pred >= threshold]=1  

  #to avoid internal layer have no pred condition
  y_pred = label_process(y_pred,labels_size)

  #extract name and write to result
  res = open(ofn,'a')
  label_name = get_label_name(name)
  for i in range(len(y_pred)):
    res.write('sample' + str(i+1) + '\t')
    if(np.all(y_pred[i] == 0)):
      res.write('unknown' + '\n')
      continue
    for j in range(len(y_pred[i])):
      if(y_pred[i][j] == 1):
        res.write(label_name[j] + '\t')
    res.write('\n')
  res.close()
  return 0

if(__name__ == '__main__'):
  main()
