import os
import sys
import pandas as pd
from tqdm import tqdm, trange
from functools import reduce
from livingTree import LineageTracker
from pprint import pprint
from collections import OrderedDict  

if len(sys.argv) < 3:
	print('Please specify the operation path and name for data set')

dir_ = sys.argv[1]
dataset_name = sys.argv[2]

folders = pd.Series(os.listdir(dir_)).apply(lambda x: os.path.join(dir_, x))
folders = folders[folders.str.contains(dataset_name, case=False)]
files = [os.path.join(folder, file_) for folder in folders 
									 for file_ in os.listdir(folder)]
print('Number of samples involved:', len(files))
# files = reduce(lambda x,y: x+y, files)
#print(files)
print('Loading data...')
tsvs = [pd.read_csv(i, sep='\t', header=1) for i in tqdm(files)]
#print([i for i in tsvs[0].iloc(1)])
print('Preprocessing data...')
taxas = pd.concat(map(lambda x: x.iloc(1)[2], tqdm(tsvs))) # just keep 2nd column
#print(taxas.shape)
rm_none = lambda x: [i for i in x if i != '']
taxas = taxas.apply(lambda x: rm_none( x.split('__') )[-1])
taxas = taxas.apply(lambda x: x.split('#')[0].split()[0])
# preprocess... # just keep the last rank 
taxas = taxas.unique()
#print(taxas.shape)
tracker = LineageTracker(ids=[])

taxas = tracker.ncbi.get_name_translator(taxas).keys() # remove ...
ids = tracker.get_ids_from_names(taxas)
#print(ids[0:100])
tracker = LineageTracker(ids=ids)
lineages = tracker.paths_sp
#print(lineages[0:5])
class map_itr(map):
	def next(self):
		return self.__next__()

print('Using Psuedocount for unclassified categories')
main_ranks = ['sk', 'k', 'p', 'c', 'o', 'f', 'g', 's']
fake_names = map_itr( lambda x: 'fake_name.{}'.format(x), 
					  range(len(lineages) * len(tracker.main_ranks)) )
lineages = [pd.Series(lineage + [ rank+'__'+fake_names.next() for rank in main_ranks[len(lineage):] ]) 
			for lineage in lineages]
#print(lineages[0:5])
# stat...and show... 
# new algorithm
sets = OrderedDict( sorted([ (rank, pd.concat( (taxa[taxa.str.startswith(rank+'__')] 
							   for taxa in lineages) ).unique() )
		for rank in main_ranks ], key=lambda x: main_ranks.index(x[0])) )
stat = OrderedDict( [( rank, len(catgrs) ) for rank, catgrs in sets.items()] )
long_ranks = tracker.main_ranks
names = {sr: lr for sr, lr in zip(main_ranks, long_ranks)}
print('Result:')
pprint( pd.DataFrame( [stat.values()], columns=stat.keys()).rename(columns=names) )
# superkingdom...kingdom...phylum...class...order...family...genus...species
	