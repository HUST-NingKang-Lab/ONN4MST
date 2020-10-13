# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import sys
import argparse

def get_ontology_shape(ontology):
  shape = []
  for i in range(len(ontology)):
    shape.append(len(ontology[i]))
  return(shape)

def get_size(sf):
  matrices_size = [312676,10234,14133]
  label_size = [4,7,22,56,43]
  return(matrices_size[sf],label_size)

def get_parser():
  parser = argparse.ArgumentParser(description='This script performs fast and accurate biome source tracking based on Ontology Neural Network. Any problems please see (https://github.com/HUST-NingKang-Lab/ONN4MDM/) for detail infomation or send email to authors! Thank you for using onn4mst!')
  parser.add_argument('ifn', type=str, default = None, help='The input file must be a numpy array file in npz format, whcih could be produced by using preprocessing.py with a biome abundance tsv file.')
  parser.add_argument('ofn', type=str, default = None, help='The output file, this file will save the results of all the samples.')
  parser.add_argument('-g', '--gpus', type=int, choices = [0,1], default = 0, help="This parameter means whether run this program on gpu devices, 0 means on cpu, 1 means on gpu. Default is 0.")
  parser.add_argument('-gid', '--gpu_core_id', type=str, default = '0', help="If you set the \'-g\'=1, then you should indicate which gpu core you want to use. For example, '0,1,4,6' means these four gpu cores is useable for the program. Default is \'0\'.")
  parser.add_argument('-s', '--selfea', type=int, choices = [0,1], default = 0, help="If you have performed feature selection with the input file, set it to 1. Default is 0.")
  parser.add_argument('-t', '--tree',type=str, default = None, help='The program need a tree file which stored the microbiome ontology. Default is \"./config/microbiome.tree\"')
  parser.add_argument('-m', '--model', type=str, default = None, help='A well trained model. Default is \"config/model_df.json\". If you set the \'-s\' to 1, you should set it to \"config/model_sf.json\"')
  parser.add_argument('-th', '--threshold', type=float, default = 0.3, help='The threshold control the sensitivity of predicting, which locate in [0,1]. Default is 0.3.')
  parser.add_argument('-mp', '--mapping', type=str, default = '0', help='The mapping file which records the name of sample should be provided for better understanding the output results. This file usually locat at the ./data/npzs/')
  parser.add_argument('-of', '--outfmt', type=int, choices = [1,2,3], default = 1, help='The output format of result. Deafult is 1')
  #parser.add_argument('-N', '--NUM', type=int, default = 0, help='Sort the biome sources by prediction score form large to small, return the TOP N biome sources. Note: If the number of biome sources with a prediction score larger than threshold is less than N, the output will only contian those biome sources with a perdiction score larger than threshold. The suggestion value of N is 10 with a threshold (0.2 or smaller). If set N=0, the ouput will contain those biome sources with a prediction score larger than threshold. Default is 0')
  return(parser)
