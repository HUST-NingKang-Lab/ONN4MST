# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import argparse
import tensorflow as tf
import numpy as np
import sys
import math
import json
import pickle
import time
from graph_builder import model
from gen_ontology import get_biome_source
from utils import *

def get_gid(gid):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gid)
  dd = '{}'.format(gid)
  print(dd)

def npzload1(ifn):
  data = np.load(ifn)
  feature = data['matrices']
  return(feature)

def Modelrecv(mdl, feature_size, label_size, gpus):
  Model = model(feature_size = feature_size, label_size = label_size, gpu_mode = gpus)
  Model.load_json(mdl)
  return(Model)

def Modelload(fn,Model):
  matrices = npzload1(fn)
  samplenum = len(matrices)
  matrices = matrices.reshape(samplenum, -1)
  #sample a batch
  all_batch = []
  for i in range(samplenum):
    all_batch.append(i)
  all_batch = np.array(all_batch)
  #feed the model with these samples
  feed = {Model.x: matrices[all_batch]}
  y_pred = Model.sess.run([Model.y_pred], feed)
  y_pred = y_pred[0]
  return(y_pred)

def scale_prob(y_pred):
  matrices_size,label_size = get_size(0)
  sn = len(y_pred)
  #L2
  start2,end2 = 0,label_size[0]
  pred_l2 = y_pred[:,start2:end2]
  l2_unknown = []
  for i in range(sn):
    total_prob,l2_unknown_tmp = 0,[]
    for j in range(len(pred_l2[0])):
      total_prob += pred_l2[i,j]
    if(total_prob >= 1):
      unknown_prob = 0
      l2_unknown_tmp.append(unknown_prob)
      for j in range(len(pred_l2[0])):
        pred_l2[i,j] = pred_l2[i,j] / total_prob
    else:
      unknown_prob = 1-total_prob
      l2_unknown_tmp.append(unknown_prob)
    l2_unknown.append(l2_unknown_tmp)
  l2_unknown = np.array(l2_unknown)

  #L3
  start3,end3 = end2,end2 + label_size[1]
  pred_l3 = y_pred[:,start3:end3]
  l3_unknown = []
  for i in range(sn):
    total_prob,l3_unknown_tmp = 0,[]
    for j in range(len(pred_l3[0])):
      total_prob += pred_l3[i,j]
    if(total_prob >= 1):
      unknown_prob = 0
      l3_unknown_tmp.append(unknown_prob)
      for j in range(len(pred_l3[0])):
        pred_l3[i,j] = pred_l3[i,j] / total_prob
    else:
      unknown_prob = 1-total_prob
      l3_unknown_tmp.append(unknown_prob)
    l3_unknown.append(l3_unknown_tmp)
  l3_unknown = np.array(l3_unknown)

  #L4
  start4,end4 = end3,end3+label_size[2]
  pred_l4 = y_pred[:,start4:end4]
  l4_unknown = []
  for i in range(sn):
    total_prob,l4_unknown_tmp = 0,[]
    for j in range(len(pred_l4[0])):
      total_prob += pred_l4[i,j]
    if(total_prob >= 1):
      unknown_prob = 0
      l4_unknown_tmp.append(unknown_prob)
      for j in range(len(pred_l4[0])):
        pred_l4[i,j] = pred_l4[i,j] / total_prob
    else:
      unknown_prob = 1-total_prob
      l4_unknown_tmp.append(unknown_prob)
    l4_unknown.append(l4_unknown_tmp)
  l4_unknown = np.array(l4_unknown)

  #L5
  start5,end5 = end4,end4+label_size[3]
  pred_l5 = y_pred[:,start5:end5]
  l5_unknown = []
  for i in range(sn):
    total_prob,l5_unknown_tmp = 0,[]
    for j in range(len(pred_l5[0])):
      total_prob += pred_l5[i,j]
    if(total_prob >= 1):
      unknown_prob = 0
      l5_unknown_tmp.append(unknown_prob)
      for j in range(len(pred_l5[0])):
        pred_l5[i,j] = pred_l5[i,j] / total_prob
    else:
      unknown_prob = 1-total_prob
      l5_unknown_tmp.append(unknown_prob)
    l5_unknown.append(l5_unknown_tmp)
  l5_unknown = np.array(l5_unknown)

  #L6
  start6,end6 = end5,end5+label_size[4]
  pred_l6 = y_pred[:,start6:end6]
  l6_unknown = []
  for i in range(sn):
    total_prob,l6_unknown_tmp = 0,[]
    for j in range(len(pred_l6[0])):
      total_prob += pred_l6[i,j]
    if(total_prob >= 1):
      unknown_prob = 0
      l6_unknown_tmp.append(unknown_prob)
      for j in range(len(pred_l6[0])):
        pred_l6[i,j] = pred_l6[i,j] / total_prob
    else:
      unknown_prob = 1-total_prob
      l6_unknown_tmp.append(unknown_prob)
    l6_unknown.append(l6_unknown_tmp)
  l6_unknown = np.array(l6_unknown)

  pred = np.concatenate((pred_l2,pred_l3,pred_l4,pred_l5,pred_l6), axis=1)
  unknown = np.concatenate((l2_unknown,l3_unknown,l4_unknown,l5_unknown,l6_unknown), axis = 1)
  return(pred,unknown)


def threshold_process(th,y_pred):
  sn = len(y_pred)
  li = layerindex = [0,4,11,33,89,132]
  for i in range(sn):
    tmp = y_pred[i]
    for j in range(1,len(li)):
      e = tmp[li[j-1]:li[j]]
      e[e < th] = 0
      if(np.any(e)):
        continue
      else:
        y_pred[i][li[j-1]:] = 0
        break
  #y_pred[y_pred < th] = 0
  #y_pred[y_pred >= th] = 1
  return(y_pred)

def res2txt_mode1(th,y_pred,unknown,ontology,ofn):
  os.popen('rm -f {} >/dev/null'.format(ofn))
  time.sleep(3)
  res = open(ofn,'w')
  label_name = get_biome_source(ontology)
  for i in range(len(y_pred)):
    res.write('>Sample_' + str(i+1) + '\t')

    #L2
    l2_ln = label_name[0:4]
    l2_pred = y_pred[i,0:4]
    true_l2_ln = []
    res.write('Layer2|')
    for j in range(len(l2_pred)):
      if(l2_pred[j] >= th):
        true_l2_ln.append(l2_ln[j])
    for j in range(len(true_l2_ln)):
      if(j == (len(true_l2_ln)-1)):
        res.write(str(true_l2_ln[j]))
      else:
        res.write(str(true_l2_ln[j]) + ',')
    #if(len(true_l2_ln) == 0 and unknown[i,0] >= th):
      #res.write('Unknown_L2')
    #if(len(true_l2_ln) != 0 and unknown[i,0] >= th):
      #res.write(',Unknown_L2')
    res.write('\t')

    #L3
    l3_ln = label_name[4:11]
    l3_pred = y_pred[i,4:11]
    true_l3_ln = []
    res.write('Layer3|')
    for j in range(len(l3_pred)):
      if(l3_pred[j] >= th):
        true_l3_ln.append(l3_ln[j])
    for j in range(len(true_l3_ln)):
      if(j == (len(true_l3_ln)-1)):
        res.write(str(true_l3_ln[j]))
      else:
        res.write(str(true_l3_ln[j]) + ',')
    #if(len(true_l3_ln) == 0 and unknown[i,1] >= th):
      #res.write('Unknown_L3')
    #if(len(true_l3_ln) != 0 and unknown[i,1] >= th):
      #res.write(',Unknown_L3')
    res.write('\t')

    #L4
    l4_ln = label_name[11:33]
    l4_pred = y_pred[i,11:33]
    true_l4_ln = []
    res.write('Layer4|')
    for j in range(len(l4_pred)):
      if(l4_pred[j] >= th):
        true_l4_ln.append(l4_ln[j])
    for j in range(len(true_l4_ln)):
      if(j == (len(true_l4_ln)-1)):
        res.write(str(true_l4_ln[j]))
      else:
        res.write(str(true_l4_ln[j]) + ',')
    #if(len(true_l4_ln) == 0 and unknown[i,2] >= th):
      #res.write('Unknown_L4')
    #if(len(true_l4_ln) != 0 and unknown[i,2] >= th):
      #res.write(',Unknown_L4')
    res.write('\t')

    #L5
    l5_ln = label_name[33:89]
    l5_pred = y_pred[i,33:89]
    true_l5_ln = []
    res.write('Layer5|')
    for j in range(len(l5_pred)):
      if(l5_pred[j] >= th):
        true_l5_ln.append(l5_ln[j])
    for j in range(len(true_l5_ln)):
      if(j == (len(true_l5_ln)-1)):
        res.write(str(true_l5_ln[j]))
      else:
        res.write(str(true_l5_ln[j]) + ',')
    #if(len(true_l5_ln) == 0 and unknown[i,3] >= th):
      #res.write('Unknown_L5')
    #if(len(true_l5_ln) != 0 and unknown[i,3] >= th):
      #res.write(',Unknown_L5')
    res.write('\t')

    #L6
    l6_ln = label_name[89:132]
    l6_pred = y_pred[i,89:132]
    true_l6_ln = []
    res.write('Layer6|')
    for j in range(len(l6_pred)):
      if(l6_pred[j] >= th):
        true_l6_ln.append(l6_ln[j])
    for j in range(len(true_l6_ln)):
      if(j == (len(true_l6_ln)-1)):
        res.write(str(true_l6_ln[j]))
      else:
        res.write(str(true_l6_ln[j]) + ',')
    #if(len(true_l6_ln) == 0 and unknown[i,4] >= th):
      #res.write('Unknown_L6')
    #if(len(true_l6_ln) != 0 and unknown[i,4] >= th):
      #res.write(',Unknown_L6')
    res.write('\n')
  res.close()
  return 0

def res2txt_mode2(th,y_pred,unknown,ontology,ofn):
  os.popen('rm -f {} >/dev/null'.format(ofn))
  time.sleep(3)
  res = open(ofn,'w')
  label_name = get_biome_source(ontology)
  for i in range(len(y_pred)):
    res.write('>Sample_' + str(i+1) + '\t')
    prob_line = '>Sample_' + str(i+1) + '\t' + 'Layer2|'
    #L2
    l2_ln = label_name[0:4]
    l2_pred = y_pred[i,0:4]
    true_l2_ln = []
    true_l2_pred = []
    res.write('Layer2|')
    for j in range(len(l2_pred)):
      if(l2_pred[j] >= th):
        true_l2_ln.append(l2_ln[j])
        true_l2_pred.append(l2_pred[j])
    for j in range(len(true_l2_ln)):
      if(j == (len(true_l2_ln)-1)):
        res.write(str(true_l2_ln[j]))
        prob_line = prob_line + str(true_l2_pred[j])
      else:
        res.write(str(true_l2_ln[j]) + ',')
        prob_line = prob_line + str(true_l2_pred[j]) + ','
    res.write('\t')
    prob_line = prob_line + '\t' + 'Layer3|'

    #L3
    l3_ln = label_name[4:11]
    l3_pred = y_pred[i,4:11]
    true_l3_ln = []
    true_l3_pred = []
    res.write('Layer3|')
    for j in range(len(l3_pred)):
      if(l3_pred[j] >= th):
        true_l3_ln.append(l3_ln[j])
        true_l3_pred.append(l3_pred[j])
    for j in range(len(true_l3_ln)):
      if(j == (len(true_l3_ln)-1)):
        res.write(str(true_l3_ln[j]))
        prob_line = prob_line + str(true_l3_pred[j])
      else:
        res.write(str(true_l3_ln[j]) + ',')
        prob_line = prob_line + str(true_l3_pred[j]) + ','
    res.write('\t')
    prob_line = prob_line + '\t' + 'Layer4|'

    #L4
    l4_ln = label_name[11:33]
    l4_pred = y_pred[i,11:33]
    true_l4_ln = []
    true_l4_pred = []
    res.write('Layer4|')
    for j in range(len(l4_pred)):
      if(l4_pred[j] >= th):
        true_l4_ln.append(l4_ln[j])
        true_l4_pred.append(l4_pred[j])
    for j in range(len(true_l4_ln)):
      if(j == (len(true_l4_ln)-1)):
        res.write(str(true_l4_ln[j]))
        prob_line = prob_line + str(true_l4_pred[j])
      else:
        res.write(str(true_l4_ln[j]) + ',')
        prob_line = prob_line + str(true_l4_pred[j]) + ','
    res.write('\t')
    prob_line = prob_line + '\t' + 'Layer5|'

    #L5
    l5_ln = label_name[33:89]
    l5_pred = y_pred[i,33:89]
    true_l5_ln = []
    true_l5_pred = []
    res.write('Layer5|')
    for j in range(len(l5_pred)):
      if(l5_pred[j] >= th):
        true_l5_ln.append(l5_ln[j])
        true_l5_pred.append(l5_pred[j])
    for j in range(len(true_l5_ln)):
      if(j == (len(true_l5_ln)-1)):
        res.write(str(true_l5_ln[j]))
        prob_line = prob_line + str(true_l5_pred[j])
      else:
        res.write(str(true_l5_ln[j]) + ',')
        prob_line = prob_line + str(true_l5_pred[j]) + ','
    res.write('\t')
    prob_line = prob_line + '\t' + 'Layer6|'

    #L6
    l6_ln = label_name[89:132]
    l6_pred = y_pred[i,89:132]
    true_l6_ln = []
    true_l6_pred = []
    res.write('Layer6|')
    for j in range(len(l6_pred)):
      if(l6_pred[j] >= th):
        true_l6_ln.append(l6_ln[j])
        true_l6_pred.append(l6_pred[j])
    for j in range(len(true_l6_ln)):
      if(j == (len(true_l6_ln)-1)):
        res.write(str(true_l6_ln[j]))
        prob_line = prob_line + str(true_l6_pred[j])
      else:
        res.write(str(true_l6_ln[j]) + ',')
        prob_line = prob_line + str(true_l6_pred[j]) + ','
    res.write('\n')
    res.write(str(prob_line) + '\n')
  res.close()
  return 0

def res2txt_mode3(th,y_pred,unknown,ontology,ofn):
  os.popen('rm -f {} >/dev/null'.format(ofn))
  time.sleep(3)
  res = open(ofn,'w')
  label_name = get_biome_source(ontology)
  for i in range(len(y_pred)):
    res.write('>Sample_' + str(i+1) + '\t')
    prob_line = '>Sample_' + str(i+1) + '\t' + 'Layer2|'
    #L2
    l2_ln = label_name[0:4]
    l2_pred = y_pred[i,0:4]
    true_l2_ln = []
    res.write('Layer2|')
    for j in range(len(l2_pred)):
      if(l2_pred[j] >= 0):
        true_l2_ln.append(l2_ln[j])
    for j in range(len(true_l2_ln)):
      if(j == (len(true_l2_ln)-1)):
        res.write(str(true_l2_ln[j]))
        prob_line = prob_line + str(l2_pred[j])
      else:
        res.write(str(true_l2_ln[j]) + ',')
        prob_line = prob_line + str(l2_pred[j]) + ','
    if(len(true_l2_ln) != 0 and unknown[i,0] >= 0):
      res.write(',unknown')
    res.write('\t')
    prob_line = prob_line + ',' + str(unknown[i,0]) + '\t' + 'Layer3|'

    #L3
    l3_ln = label_name[4:11]
    l3_pred = y_pred[i,4:11]
    true_l3_ln = []
    res.write('Layer3|')
    for j in range(len(l3_pred)):
      if(l3_pred[j] >= 0):
        true_l3_ln.append(l3_ln[j])
    for j in range(len(true_l3_ln)):
      if(j == (len(true_l3_ln)-1)):
        res.write(str(true_l3_ln[j]))
        prob_line = prob_line + str(l3_pred[j])
      else:
        res.write(str(true_l3_ln[j]) + ',')
        prob_line = prob_line + str(l3_pred[j]) + ','
    if(len(true_l3_ln) != 0 and unknown[i,1] >= 0):
      res.write(',unknown')
    res.write('\t')
    prob_line = prob_line + ',' + str(unknown[i,1]) + '\t' + 'Layer4|'

    #L4
    l4_ln = label_name[11:33]
    l4_pred = y_pred[i,11:33]
    true_l4_ln = []
    res.write('Layer4|')
    for j in range(len(l4_pred)):
      if(l4_pred[j] >= 0):
        true_l4_ln.append(l4_ln[j])
    for j in range(len(true_l4_ln)):
      if(j == (len(true_l4_ln)-1)):
        res.write(str(true_l4_ln[j]))
        prob_line = prob_line + str(l4_pred[j])
      else:
        res.write(str(true_l4_ln[j]) + ',')
        prob_line = prob_line + str(l4_pred[j]) + ','
    if(len(true_l4_ln) != 0 and unknown[i,2] >= 0):
      res.write(',unknown')
    res.write('\t')
    prob_line = prob_line + ',' + str(unknown[i,2]) + '\t' + 'Layer5|'

    #L5
    l5_ln = label_name[33:89]
    l5_pred = y_pred[i,33:89]
    true_l5_ln = []
    res.write('Layer5|')
    for j in range(len(l5_pred)):
      if(l5_pred[j] >= 0):
        true_l5_ln.append(l5_ln[j])
    for j in range(len(true_l5_ln)):
      if(j == (len(true_l5_ln)-1)):
        res.write(str(true_l5_ln[j]))
        prob_line = prob_line + str(l5_pred[j])
      else:
        res.write(str(true_l5_ln[j]) + ',')
        prob_line = prob_line + str(l5_pred[j]) + ','
    if(len(true_l5_ln) != 0 and unknown[i,3] >= 0):
      res.write(',unknown')
    res.write('\t')
    prob_line = prob_line + ',' + str(unknown[i,3]) + '\t' + 'Layer6|'

    #L6
    l6_ln = label_name[89:132]
    l6_pred = y_pred[i,89:132]
    true_l6_ln = []
    res.write('Layer6|')
    for j in range(len(l6_pred)):
      if(l6_pred[j] >= 0):
        true_l6_ln.append(l6_ln[j])
    for j in range(len(true_l6_ln)):
      if(j == (len(true_l6_ln)-1)):
        res.write(str(true_l6_ln[j]))
        prob_line = prob_line + str(l6_pred[j])
      else:
        res.write(str(true_l6_ln[j]) + ',')
        prob_line = prob_line + str(l6_pred[j]) + ','
    if(len(true_l6_ln) != 0 and unknown[i,4] >= 0):
      res.write(',unknown')
    res.write('\n')
    prob_line = prob_line + ',' + str(unknown[i,4])
    res.write(str(prob_line) + '\n')
  res.close()
  return 0


  


def sort_lst(true_label,prob_lst):
  tmp_lst = []
  for i in range(len(true_label)):
    my_lst = []
    my_lst.append(true_label[i])
    my_lst.append(prob_lst[i])
    tmp_lst.append(my_lst)
  print(tmp_lst)
  tmp_lst.sort(key=lambda x:x[1], reverse=True)
  print(tmp_lst)
  return(tmp_lst)

def get_topn(y_pred,ontology,ofn,th,n):
  os.popen('rm -f {} >/dev/null'.format(ofn))
  time.sleep(3)
  res = open(ofn,'w')
  label_name = get_biome_source(ontology)

  for i in range(len(y_pred)):
    res.write('sample' + str(i+1) + '|')
    if(np.all(y_pred[i] < th)):
      res.write('Unknown' + '\n')
      continue
    true_label = []
    prob_lst = []
    for j in range(len(y_pred[i])):
      probb = y_pred[i][j]
      if(probb >= th):
        true_label.append(label_name[j])
        prob_lst.append(probb)

    label_prob = sort_lst(true_label,prob_lst)
    if(len(label_prob) >= n):
      for j in range(n):
        if(j == (n-1)):
          res.write(str(label_prob[j][0]) + '\n')
        else:
          res.write(str(label_prob[j][0]) + ',')
    elif(len(label_prob) < n):
      for j in range(len(label_prob)):
        if(j == (len(label_prob)-1)):
          res.write(str(label_prob[j][0]) + '\n')
        else:
          res.write(str(label_prob[j][0]) + ',')
  res.close()
  return 0

