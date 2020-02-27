# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import sys
import pickle
import argparse

def get_ontology_shape(ontology):
  shape = []
  for i in range(len(ontology)):
    shape.append(len(ontology[i]))
  return(shape)

def get_label(ifn):
  with open(ifn,'rb') as f:
    label = pickle.load(f)
  return(label)


def get_label_name(ifn):
  with open(ifn,'rb') as f:
    label = pickle.load(f)
  label0,label1,label2,label3,label4,label5 = label['label_0'],label['label_1'],label['label_2'],label['label_3'],label['label_4'],label['label_5']
  labellist = label0 + label1 + label2 + label3 + label4 + label5
  return(labellist)

def get_tree(ifn):
  with open(ifn,'rb') as f:
    label = pickle.load(f)
  label0,label1,label2,label3,label4,label5 = label['label_0'],label['label_1'],label['label_2'],label['label_3'],label['label_4'],label['label_5']
  tree = []
  tree.append(len(label0))
  tree.append(len(label1))
  tree.append(len(label2))
  tree.append(len(label3))
  tree.append(len(label4))
  tree.append(len(label5))
  return(tree)

def get_parser():
  parser = argparse.ArgumentParser(description='This script performs fast and accurate biome source tracking based on Ontology Neural Network. Any problems please send email to hugozha@hust.edu.cn, thank you for your cooperation!')
  parser.add_argument('-g', '--gpus', type=int, default = 0, help="This parameter means whether run this on gpu devices, 0 means on cpu, 1 means on gpu, default is 0.")
  parser.add_argument('-t', '--tree', type=str, default = './config/biome_rf.tree', help='The tree file, you should provid a file, which contains the hierarchical structure of the microbiome. The default is "biome_rf.tree".')
  parser.add_argument('-n', '--name', type=str, default = './config/labels.pydata', help='The name is a pickle file which store the actual name of each biome node.')
  parser.add_argument('-i', '--ifn', type=str, default = None, help='The input file must be a numpy array file in npz format, whcih could be produced by using preprocessing.py with a biome abundance tsv file.')
  parser.add_argument('-o', '--ofn', type=str, default = './result.txt', help='The output file, this file will save the results of all the samples in the same order. The default is "./result.txt"')
  parser.add_argument('-m', '--model', type=str, default = './config/model_rf.json', help='A well trained model, we provid the default model that you can find it at "./config/model_rf.json".')
  parser.add_argument('-th', '--threshold', type=float, default = 0.5, help='The threshold control the sensitivity of predicting, it is a float number, which locate in [0,1]. The default is 0.5.')
  #parser.add_argument('-c', '--cpu', type=int, default = 1, help='when you set -g with "0", you will use cpu for working, -c indicates how many threads you wish to use in your task. The default is 1.')
  return(parser)
