import argparse
import os
import joblib
#from dp_utils import SuperTree, DataLoader, IdConverter, Selector, npz_merge
from dp_utils import DataLoader, IdConverter, Selector, npz_merge
from livingTree import SuperTree
import numpy as np
import pandas as pd
from functools import reduce
from tqdm import tqdm, trange
from joblib import Parallel, delayed


des = '''
This is an integrated data preprocessor for Ontology-aware Neural Network.

Work mode:
\tcheck mode: check all of your data files, the error data file are saved in tmp/ folder.
\tbuild mode: de novoly build a species tree using your own data.
\tconvert mode: convert tsv file from EBI MGnify database to model acceptable n-dimensional array.
\tcount mode: count the number of samples in each biome.
\tmerge mode: merge multiple npz files to a single npz.
\tselect mode: do feature selection for merged matrices npz.
'''
parser = argparse.ArgumentParser(description=des, 
								formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("mode", type=str, choices=['check', 'build', 'convert', 'count', 'merge', 'select'], 
					default='convert',
					help = "Work mode of the program. default: convert")
parser.add_argument("-p", "--n_jobs", type=int, default=1, 
					help = 'The number of processors to use. default: 1')
parser.add_argument("-b", "--batch_index", type=int, default=-1, 
					help = 'The batch number to process, -1 means process all at once. default: -1')
parser.add_argument("-s", "--batch_size", type=int, default=1000, 
					help = 'The number of samples in each batch. -1 means process all at once. default: -1')
parser.add_argument("-i", '--input_dir', type=str, default='data/tsvs/',
					help = 'Input directory, must be parent folder of biome folders. default: data/')
parser.add_argument("-o", '--output_dir', type=str, default='data/npzs/',
					help = 'Output directory. default: matrices')					 
parser.add_argument("-t", '--tree', type=str, default='data/trees/',
					help = 'The directory of trees (species_tree.pkl and biome_tree.pkl). default: tree/')

args = parser.parse_args()
if args.mode == 'convert':
	# print('convert mode')
	loader = DataLoader(path = args.input_dir, 
						batch_size = args.batch_size, 
						batch_index=args.batch_index)
else:
	loader = DataLoader(path = args.input_dir)

if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
if not os.path.isdir(args.tree): os.mkdir(args.tree)

if args.mode == 'check':
	# tested
	# check just 1 line in file errors
	loader.check_data(header=1)
	loader.save_error_list()

elif args.mode == 'build':
	# tested
	converter = IdConverter()
	paths = pd.concat(map(lambda x: x.iloc(1)[2], tqdm(loader.get_data(header=1))))
	paths = paths.unique()
	paths = [converter.convert(x, sep=';') for x in  paths]
	# print(list(paths)[0:5])
	stree = SuperTree()
	stree.create_node(identifier='root')
	print('Building tree', flush=True)
	stree.from_paths(tqdm(paths))
	# stree.show()
	stree.to_pickle(file=os.path.join(args.tree, 'species_tree.pkl'))

	biomes = map(lambda x: converter.convert(x, sep='-'), os.listdir(args.input_dir))
	biomes = list(map(lambda x: x[1:], biomes))
	#print(biomes)
	btree = SuperTree()
	btree.create_node(identifier='root')
	btree.from_paths(biomes)
	#btree.show()
	btree.to_pickle(file=os.path.join(args.tree, 'biome_tree.pkl'))
	print('species tree and biome tree are saved in {}'.format(args.tree))


elif args.mode == 'convert':
	print('Loading trees and data......', end='', flush = True)
	tree = SuperTree()
	converter = IdConverter()
	stree = tree.from_pickle(os.path.join(args.tree, 'species_tree.pkl'))
	stree.init_nodes_data(value = 0)    # put this outside
	btree = tree.from_pickle(os.path.join(args.tree, 'biome_tree.pkl'))
	btree.init_nodes_data(value = 0)     # put this outside
	data = []
	for i in map(lambda x: x.iloc(1)[1:], loader.get_data(header=1)): data.append(i.values.tolist())
	print('finished !\nPreprocessing data......', end = '', flush = True)
	data = [{converter.convert(sp[1], sep=';')[-1]: sp[0] for sp in x} for x in data]
	biomes = [converter.convert(os.path.split(x)[0].split('/')[-1], sep='-') 
			  for x in loader.paths_keep]
	print('finished !')
	#biomes = list(biomes)
	#data = list(data)
	print('Total: {} biomes and {} samples'.format(len(biomes), len(data)))
	"""
	define a function(samples, biomes, stree, btree) -> matrices, labels 
	every time you run this program, just specify batch index you want to run eg 0
	this program will do function(samples, biomes, stree, btree) -> matrices, labels
	and save your results.

	** joblib parallel: Parallel(n_jobs=2)(delayed(lambda x: x**2)(i) for i in trange(10**5))
	1. parameters: batch index, n_jobs
	2. function
	3. progress bar
	"""
	@profile
	def convert_to_npzs(sample, biome_layered, species_tree, biome_tree):
		species_tree.fill_with(data = sample)
		species_tree.update_value()  		
		Sum = species_tree['root'].data
		#species_tree.remove_levels(species_tree.depth())   # substitute ? 
		matrix = species_tree.get_matrix() / Sum # relative abundance
		biome_tree.fill_with(data = {biome: 1 for biome in biome_layered})
		bfs_data = biome_tree.get_bfs_data()
		labels = [np.array(bfs_data[i], dtype=np.float32) for i in range(biome_tree.depth() + 1)]
		return matrix, labels
	
	# pre-compute reverse iteration node id order	
	# pre-generate paths to node ids 
	print('Use joblib parallel backend with {} cores'.format(args.n_jobs))
	par = Parallel(n_jobs = args.n_jobs)
	print('Performing conversion')
	res = par(delayed(convert_to_npzs)(data[i], biomes[i], stree.copy(), btree.copy(), ) for i in trange(len(data)))
	#print(res[0:2])
	raw_npzs = list(zip(*res))
	matrices = raw_npzs[0]
	labels	= raw_npzs[1]
	labels = [[np.array(label[i]) for label in labels] for i in range(len(labels[0]))]
	output_dir = os.path.join(args.output_dir, 'matrices.npz')
	np.savez(output_dir, matrices=matrices,
		label_0 = labels[0], label_1 = labels[1], label_2 = labels[2], label_3 = labels[3],
		label_4 = labels[4], label_5 = labels[5])
	print('Results are save in {}'.format(output_dir))

elif args.mode == 'count':
	# tested
	sample_count = loader.get_sample_count()
	res = ['{}:{}'.format(key, value) for key, value in sample_count.items()]
	print('\n'.join(res)) # save needed
	# id_id needed, unique tag conversion needed
	# new algorithm

elif args.mode == 'merge': 
	# tested
	files = map(lambda x: os.path.join(args.input_dir, x), os.listdir(args.input_dir))
	merged_npz = npz_merge(files)
	np.savez(os.path.join(args.output_dir, 'merged_matrices.npz'),
			matrices = merged_npz['matrices'],
			label_0 = merged_npz['label_0'],
			label_1 = merged_npz['label_1'],
			label_2 = merged_npz['label_2'],
			label_3 = merged_npz['label_3'],
			label_4 = merged_npz['label_4'],
			label_5 = merged_npz['label_5'])

elif args.mode == 'select':
	# tested
	if len(os.listdir(args.input_dir)) != 1: print('you need to merge npzs first')
	tmp = np.load(os.path.join(args.input_dir, os.listdir(args.input_dir)[0]))
	matrices = tmp['matrices']
	print(matrices.shape)
	labels_ = {i: tmp[i] for i in ['label_0', 'label_1','label_2','label_3','label_4','label_5']}
	labels = reduce(lambda x,y: np.concatenate((x,y), axis=1), labels_.values())
	selector = Selector(matrices)
	selector.run_basic_select()
	feature_ixs = selector.basic_select__
	tmp_matrices = matrices[:, feature_ixs, :]
	print(tmp_matrices.shape)
	selector = Selector(tmp_matrices)
	selector.cal_feature_importances(label=labels, n_jobs=2)
	selector.run_RF_regression_select()
	feature_ixs = selector.RF_select__
	new_matrices = tmp_matrices[:, feature_ixs, :]
	print(new_matrices.shape)
	np.savez(os.path.join(args.output_dir, 'feature-selected_matrices.npz'), 
			matrices=new_matrices,
			label_0 = labels_['label_0'],
			label_1 = labels_['label_1'],
			label_2 = labels_['label_2'],
			label_3 = labels_['label_3'],
			label_4 = labels_['label_4'],
			label_5 = labels_['label_5'],
			)



