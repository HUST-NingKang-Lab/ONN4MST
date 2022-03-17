# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import sys
from graph_builder import model
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#function for loading data
def npzload(ifn):
  data = np.load(ifn)
  feature = data['matrices']
  label0,label1,label2,label3,label4 = data['label_0'],data['label_1'],data['label_2'],data['label_3'],data['label_4']
  return(feature,label0,label1,label2,label3,label4)

#function for getting the shape of feature 
def get_feature_size(feature):
  size = len(feature[0])
  return(size)

#function for getting the shape of label
def get_label_size(labels):
  size = []
  for i in range(len(labels)):
    nodes_count = len(labels[i][0])
    size.append(nodes_count)
  return(size)

#function for evaluating truth and predictions
def eval_labels(y_true, y_pred):
  for i in range(len(y_true)):
    if float(y_true[i]) == 1 and float(y_pred[i]) < 0.5:
      return 0
    if float(y_true[i]) == 0 and float(y_pred[i]) >= 0.5:
      return 0
  return 1

#function for computing the exact match rate
def eval(y_true, y_pred):
  cnt, s = 0.0, 0.0
  for i in range(len(y_true)):
    cnt += 1
    s += eval_labels(y_true[i], y_pred[i])
  if cnt == 0:
    return 0
  return(s/cnt)


def train_model(trainfn,testfn,model_save_path,epochs,batch_size):
  print('loading data...', end='')
  matrices,label0,label1,label2,label3,label4 = npzload(trainfn)
  samplenum = len(matrices)
  matrices = matrices.reshape(samplenum,-1)
  labels = []
  labels.append(label0)
  labels.append(label1)
  labels.append(label2)
  labels.append(label3)
  labels.append(label4)
  labels_size = get_label_size(labels)
  matrices_size = get_feature_size(matrices)
  print('done!')
  print('building model...', end='')
  Model = model(feature = matrices, feature_size = matrices_size, label = labels, label_size = labels_size, lr = 1e-4)
  print('done!')

  for itr in range(epochs):
    #randomly sample a batch
    ind_batch = np.random.randint(0,len(matrices), batch_size)
    ind_batch1 = np.random.randint(0,len(matrices), 1024)
    #feed the feature to model
    feed = {Model.x: matrices[ind_batch]}
    feed[Model.y_0] = label0[ind_batch]
    feed[Model.y_1] = label1[ind_batch]
    feed[Model.y_2] = label2[ind_batch]
    feed[Model.y_3] = label3[ind_batch]
    feed[Model.y_4] = label4[ind_batch]
    loss,y_pred,y,logits,_ = Model.sess.run([Model.losses,Model.y_pred,Model.y_true, Model.logits, Model.train_op], feed)
    minloss = 0.01
    if((itr) % (epochs // 20) == 0):
      feed1= {Model.x: matrices[ind_batch1]}
      feed1[Model.y_0] = label0[ind_batch1]
      feed1[Model.y_1] = label1[ind_batch1]
      feed1[Model.y_2] = label2[ind_batch1]
      feed1[Model.y_3] = label3[ind_batch1]
      feed1[Model.y_4] = label4[ind_batch1]

      test_loss,test_y_pred,test_y= Model.sess.run([Model.losses,Model.y_pred,Model.y_true], feed1)
      print("step:", (itr), "loss:",test_loss)
      em = eval(test_y,test_y_pred)
      print("exact match rate:", em)
      if(minloss > loss):
        minloss = loss
        Model.save_json(model_save_path)

def main():
  train_model(sys.argv[1],sys.argv[2],sys.argv[3],30000,1024)

if(__name__ == '__main__'):
  main()

