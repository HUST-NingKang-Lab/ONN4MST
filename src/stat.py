#!/bin/env python
import os
import sys
import pandas as pd
from tqdm import tqdm, trange
from functools import reduce
from livingTree import LineageTracker
from pprint import pprint
from collections import OrderedDict  
from joblib import Parallel, delayed

if len(sys.argv) < 3:
	print('Please specify the operation path and name for data set')

#print(sys.argv)
dir_ = sys.argv[2]
dataset_name = sys.argv[3]
avg = False if sys.argv[1] == '0' else True
n_jobs = int(sys.argv[4])
print('stat avg: ', avg)

with open('tmp/error_list', 'r') as f:
	error_list = [i.rstrip('\n') for i in f.readlines()]

folders = pd.Series(os.listdir(dir_)).apply(lambda x: os.path.join(dir_, x))
folders = folders[folders.str.contains(dataset_name, case=False)]
files = [os.path.join(folder, file_) for folder in folders 
									 for file_ in os.listdir(folder)]
cwd = os.getcwd()
files = [os.path.join(cwd, i) for i in files if os.path.join(cwd, i) not in error_list]
print('Number of samples involved:', len(files))
# files = reduce(lambda x,y: x+y, files)
#print(files)
print('Loading data...')
tsvs = [pd.read_csv(i, sep='\t', header=1) for i in tqdm(files)]
tsvs = [tsv[ tsv[tsv.columns[1]]>0 ].reset_index(drop=True) for tsv in tsvs]
#print([i for i in tsvs[0].iloc(1)])
rm_none = lambda x: [i for i in x if i != '']
rm_blank = lambda x: [i for i in x if not i.endswith('__')]
main_ranks = ['sk', 'k', 'p', 'c', 'o', 'f', 'g', 's']

def process(taxas, progress=True):
	#print('Taxas: ', taxas.shape)
	# s__ problem
	#print(taxas)
	taxas = taxas.apply(lambda x: rm_none( rm_blank( x.replace('; ', ';').split(';') )[-1].split('__') )[-1])
	#print(taxas)
	taxas = taxas.apply(lambda x: x.split('#')[0].split()[0].replace('_', ' ').replace('[', '').replace(']', ''))
	#print(taxas)
	# taxas = taxas.apply(lambda x: tracker.ncbi.get_common_names(i).values()[0])
	# preprocess... # just keep the last rank 
	 
	# this will loss repeated categories
	tracker = LineageTracker(ids=[])
	keep = set(tracker.ncbi.get_name_translator(taxas).keys())
	taxas = [i for i in taxas if i in keep] # remove ...
	if progress:
		print('Getting ids...')
		taxas = tqdm(taxas)
	ids = tracker.get_ids_from_names(taxas)
	#print(ids[0:100])
	if progress:
		print('Tracking lineages...')
		ids = tqdm(ids)
	tracker = LineageTracker(ids=ids)
	lineages = tracker.paths_sp
	#print(lineages[0:5])
	if progress:
		print('Using Pseudocount for unclassified categories')

		with tqdm( total=len(lineages) * len(tracker.main_ranks) - sum(map(len, lineages)) ) as pbar:
			class map_itr(map):
				def next(self):
					pbar.update(1)
					return self.__next__()
		
			fake_names = map_itr( lambda x: 'fake_name.{}'.format(x), 
						  		  range(len(lineages) * len(tracker.main_ranks)) )
			lineages = [pd.Series(lineage + [ rank+'__'+fake_names.next() for rank in main_ranks[len(lineage):] ]) 
						for lineage in lineages]
	else:
		class map_itr(map):
			def next(self):
				return self.__next__()
		
		fake_names = map_itr( lambda x: 'fake_name.{}'.format(x), 
					  		  range(len(lineages) * len(tracker.main_ranks)) )
		lineages = [pd.Series(lineage + [ rank+'__'+fake_names.next() for rank in main_ranks[len(lineage):] ]) 
					for lineage in lineages]
	#print(lineages[0:5])
	# stat...and show... 
	# new algorithm
	if progress:
		print('Statistics in progress...')
	#pprint(lineages)
	lineages = [[';'.join(lineage[0:i]) for i in range(1,len(lineage)+1)] for lineage in lineages]
	lineages = [i for i in zip(*lineages)]
	#print('Lineages: ',len(lineages))
	try:
		lineages = [ (rank, set(lineages[index]) ) for index, rank in enumerate(main_ranks) ]
	except IndexError:
		lineages = [ (rank, set([]) ) for index, rank in enumerate(main_ranks) ]
	sets = OrderedDict( sorted(lineages, key=lambda x: main_ranks.index(x[0])) )
	if process:
		print(dataset_name, '\n')
		pprint(sets['p'])
	stat = OrderedDict( [( rank, len(catgrs) ) for rank, catgrs in sets.items()] )
	return stat


print('Preprocessing data...')
if not avg:
	taxas = pd.concat( (map(lambda x: x.iloc(1)[2], tqdm(tsvs))) ).astype(str) # just keep 2nd column
	taxas = taxas.unique()
	#print(taxas)
	stat = process(pd.Series(taxas))
	#print(taxas.shape)
else:
	taxas_itr = map(lambda x: x.iloc(1)[2].astype(str), tqdm(tsvs))
	Par = Parallel(n_jobs=n_jobs, backend='loky')
	table = pd.DataFrame(Par(delayed(process)(taxas, False) 
						 for taxas in taxas_itr ), 
						 columns=main_ranks)
		#table.iloc(0)[index] = process(taxas)
	nsamples = len(files)
	#print(table)
	#print(len(files))
	#print(table.apply(lambda x: x.sum(), axis=0))
	stat = OrderedDict(table.apply(lambda x: x.sum() / len(files), axis=0))


tracker = LineageTracker(ids=[])
long_ranks = tracker.main_ranks
names = {sr: lr for sr, lr in zip(main_ranks, long_ranks)}
print('Result:')
pprint( pd.Series( list(stat.values()), index=stat.keys()).rename(names) )
	
