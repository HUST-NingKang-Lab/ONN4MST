#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import argparse
import numpy as np
import sys
import math
import json
import time
from graph_builder import model
from gen_ontology import get_biome_source
from utils import *
from predicting import *

def get_gid(gid):
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gid)

def main():
  #get args
  parser = get_parser()
  args = parser.parse_args()
  gpus, ifn, ofn, ontology, mdl, threshold, sf, gid, mp, ofmt = args.gpus, args.ifn, args.ofn, args.tree, args.model, args.threshold, args.selfea, args.gpu_core_id, args.mapping, args.outfmt
  matrices_size,label_size = get_size(sf)
  if(gpus == 1):
    print("gpu mode...")
    get_gid(gid)
  else:
    print("cpu mode...")
  print("recovering model...")
  Model = Modelrecv(mdl, matrices_size, label_size, gpus)
  print("performing prediction...")
  y_pred = Modelload(ifn,Model)
  print("generating predicting result...")
  y_pred,unknown = scale_prob(y_pred)
  if(ofmt ==  1):
    print("the output format is 1...")
    y_pred = threshold_process(threshold,y_pred)
    res2txt_mode1(threshold,y_pred,unknown,ontology,mp,ofn)
  if(ofmt == 2):
    print("the output format is 2...")
    y_pred = threshold_process(threshold,y_pred)
    res2txt_mode2(threshold,y_pred,unknown,ontology,mp,ofn)
  if(ofmt == 3):
    print("the output format is 3...")
    res2txt_mode3(threshold,y_pred,unknown,ontology,mp,ofn)
  print("Done!")
if(__name__ == '__main__'):
  main()

