#!/usr/bin/env python3
import sys

def readlabel(ifn):
  label = []
  with open(ifn,'r') as f:
    tmp = []
    for line in f.readlines():
      line = line.strip()
      if(line == ''):
        label.append(tmp)
        tmp = []
      if('-' in line):
        tmp.append(line)
  return(label)

def get_biome_source(ifn):
  sources = []
  with open(ifn,'r') as f:
    for line in f.readlines():
      line = line.strip()
      if('-' in line):
        sources.append(line)
  return(sources)

def main():
  label = readlabel(sys.argv[1])

if(__name__ == '__main__'):
  main()
