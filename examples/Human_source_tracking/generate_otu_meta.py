import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce
from collections import OrderedDict
import warnings 
warnings.filterwarnings('ignore')


# -----configuration goes here-----

sinks_tsvfolder = 'data/ONN/tsvs/'
sources_tsvfolder = '/home/qiuhao/ONN/ONNdata/'
np.random.seed(4)
possible_pollution_sources = []
random.seed(4)
# -----ends configuration ---------

sink_biomes = [os.path.join(sinks_tsvfolder, i) for i in os.listdir(sinks_tsvfolder)]
if possible_pollution_sources == []:
    source_biomes = [os.path.join(sources_tsvfolder, i) for i in os.listdir(sources_tsvfolder)]
else:
    source_biomes = [os.path.join(sources_tsvfolder, i) for i in os.listdir(sources_tsvfolder)  
                                                        if i in possible_pollution_sources]

sink_samples = [os.path.join(folder, f) for folder in sink_biomes 
                                        for f in os.listdir(folder)]

# sampling samples since SourceTracker and FEAST are slow
if possible_pollution_sources == []:
    source_samples = [os.path.join(folder, f) for folder in source_biomes 
                                              for f in random.sample(os.listdir(folder), min(2, len(os.listdir(folder))))]  
else:
    source_samples = [os.path.join(folder, f) for folder in source_biomes 
                      for f in random.sample(os.listdir(folder), min(int(300/len(possible_pollution_sources), 
						  											 len(os.listdir(folder)))))] 

# read and preprocess samples
filter_cols = lambda x: x[x.columns[1:3]]
rm_empty = lambda x: ';'.join(filter(lambda x: not x.endswith('__'), x.split(';')))
def preprocess(tsv):
    tsv = filter_cols(tsv)
    tsv.loc[:, 'taxonomy'] = tsv.loc[:, 'taxonomy'].str.replace('; ',';').apply(rm_empty).\
        str.extract(pat=r'(p__.*)(?=\b)', expand=False)
    '''tsv.loc[:, 'taxonomy'] = tsv.loc[:, 'taxonomy'].apply(lambda x: ';'.join(x.split(';')[(2 
                                                          if x.startswith('sk') else 1):]))'''
    return tsv

read_tsv = lambda x: (x, preprocess(pd.read_csv(x, sep='\t', header=1)))
sink_tsvs = OrderedDict(map(read_tsv, tqdm(sink_samples)))
source_tsvs = OrderedDict(map(read_tsv, tqdm(source_samples)))

# concatenate samples, to form a otu table
def merge(x, y): 
    x = x.groupby(by='taxonomy').sum()
    y = y.groupby(by='taxonomy').sum()
    return pd.merge(left=x, right=y, on='taxonomy', how='outer')

otu = reduce(merge, tqdm(list(sink_tsvs.values())+list(source_tsvs.values()))).fillna(0) 
print(otu.isna().sum().sum())
#otu.index.name = ''
#print(otu.index == '')
#otu = otu.drop(columns='taxonomy')
#otu.loc[:, 'taxonomy'] = np.arange(otu.shape[0]).astype(str)
otu.astype(int).to_csv('otu.tsv', sep='\t')
# generate meta table from paths
meta_sinks = pd.DataFrame(map(lambda x: (x.split('/')[-2], x.split('/')[-1].split('_')[0]), sink_tsvs.keys()), 
                          columns=['Env', 'SampleID'])
meta_sinks.loc[:, 'id'] = np.arange(1, meta_sinks.shape[0]+1)
meta_sinks.loc[:, 'SourceSink'] = 'Sink'
meta_sinks.loc[:, 'Study'] = 'MGYS00001601'
meta_sinks.loc[:, 'Description'] = 'Human associated'
meta_sinks.loc[:, 'Details'] = 'Undefined'

meta_sources = pd.DataFrame(map(lambda x: (x.split('/')[-2], x.split('/')[-1].\
                                           split('-')[1].split('.')[0]), source_tsvs.keys()),
                            columns=['Env', 'SampleID'])
meta_sources.loc[:, 'id'] = 1
meta_sources.loc[:, 'SourceSink'] = 'Source'
meta_sources.loc[:, 'Study'] = 'Mixed'
meta_sources.loc[:, 'Description'] = 'Possible pollution source'
meta_sources.loc[:, 'Details'] = 'Undefined'
meta = pd.concat([meta_sinks, meta_sources], axis=0).reset_index(drop=True)
meta = meta[['SampleID', 'Study', 'Env', 'Description', 'SourceSink', 'id', 'Details']]
meta.to_csv('meta.tsv', sep='\t', index=False)
print(meta)









