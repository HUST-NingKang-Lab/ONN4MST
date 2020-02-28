#!/usr/bin/env python3
import numpy as np
import sys
import os

def npzload(ifn):
  data = np.load(ifn)
  matrices,label0,label1,label2,label3,label4,label5 = data['matrices'],data['label_0'],data['label_1'],data['label_2'],data['label_3'],data['label_4'],data['label_5']
  return(matrices,label0,label1,label2,label3,label4,label5)

def npzmerge(ifn1,ifn2,ifn3):
  matrices_x,label0_x,label1_x,label2_x,label3_x,label4_x,label5_x = npzload(ifn1)
  matrices_y,label0_y,label1_y,label2_y,label3_y,label4_y,label5_y = npzload(ifn2)
  matrices = np.concatenate([matrices_x,matrices_y], axis=0)
  label0 = np.concatenate([label0_x,label0_y], axis=0)
  label1 = np.concatenate([label1_x,label1_y], axis=0)
  label2 = np.concatenate([label2_x,label2_y], axis=0)
  label3 = np.concatenate([label3_x,label3_y], axis=0)
  label4 = np.concatenate([label4_x,label4_y], axis=0)
  label5 = np.concatenate([label5_x,label5_y], axis=0)
  np.savez(ifn3,matrices=matrices,label_0=label0,label_1=label1,label_2=label2,label_3=label3,label_4=label4,label_5=label5)

def main():
  npzmerge(sys.argv[1],sys.argv[2],sys.argv[3])

if(__name__ == '__main__'):
  main()

