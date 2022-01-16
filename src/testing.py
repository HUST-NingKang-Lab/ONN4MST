# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
import math
import json
import pickle
from graph_builder import model
from gen_ontology import get_biome_source
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def npzload(ifn):
  data = np.load(ifn)
  feature = data['matrices']
  label0,label1,label2,label3,label4 = data['label_0'],data['label_1'],data['label_2'],data['label_3'],data['label_4']
  return(feature,label0,label1,label2,label3,label4)

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

def eval_labels(y_true, y_pred):
  for i in range(len(y_true)):
    if float(y_true[i]) == 1 and float(y_pred[i]) < 0.5:
      return 0
    if float(y_true[i]) == 0 and float(y_pred[i]) >= 0.5:
      return 0
  return 1

def eval(y_true, y_pred):
  cnt, s = 0.0, 0.0
  for i in range(len(y_true)):
    cnt += 1
    s += eval_labels(y_true[i], y_pred[i])
  if cnt == 0:
    return 0
  return(s/cnt)


def test_model(fn,Model,ontology,ofn):\
  print('loading data...', end='')
  matrices,label0,label1,label2,label3,label4 = npzload(fn)
  samplenum = len(matrices)
  matrices = matrices.reshape(samplenum,-1)
  print('done!')
  all_batch = np.arange(matrices.shape[0])
  feed = {Model.x: matrices[all_batch]}
  feed[Model.y_0] = label0[all_batch]
  feed[Model.y_1] = label1[all_batch]
  feed[Model.y_2] = label2[all_batch]
  feed[Model.y_3] = label3[all_batch]
  feed[Model.y_4] = label4[all_batch]
  loss,y_pred,y = Model.sess.run([Model.losses,Model.y_pred,Model.y_true], feed)
  em = eval(y,y_pred)
  print("exactly match rate:", em)
  #writing results to output
  label_name = get_biome_source(ontology)
  log = open(ofn,'a')
  y_pred = np.around(y_pred, decimals=4)
  for i in range(len(y)):
    log.write('sample' + str(i) + '|')
    for j in range(len(y_pred[i])):
      if(j == (len(y_pred[i])-1)):
        log.write(str(label_name[j]))
      else:
        log.write(str(label_name[j]) + ',')
    log.write('\t')
    log.write('sample' + str(i) + '|')
    true_label = []
    for j in range(len(y[i])):
      if(y[i][j] >= 0.5):
        true_label.append(label_name[j])
    for j in range(len(true_label)):
      if(j == (len(true_label)-1)):
        log.write(str(true_label[j]))
      else:
        log.write(str(true_label[j]) + ',')
    log.write('\t')
    for j in range(len(y_pred[i])):
      if(j == (len(y_pred[i])-1)):
        log.write(str(y_pred[i][j]))
      else:
        log.write(str(y_pred[i][j]) + ',')
    log.write('\n')
  log.close()

def main():
  ifn = sys.argv[1]
  trainedmodel = sys.argv[2]
  ontology = sys.argv[3]
  ofn = sys.argv[4]
  labels_size = [4,7,22,56,43]
  matrices_size = [312676,10234,14133]
  Model = model(feature_size = matrices_size[0], label_size = labels_size)
  Model.load_json(trainedmodel)
  test_model(ifn,Model,ontology,ofn)

if(__name__ == '__main__'):
  main()

