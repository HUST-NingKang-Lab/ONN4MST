from treelib import Node, Tree
import os, sys
# from joblib import *
from pandas import read_csv, DataFrame
import numpy as np
import pickle
import pandas as pd
#from ete3 import NCBITaxa

# new_tree = Tree(tree.subtree(tree.root), deep=True)

class super_tree(Tree):

	def get_bfs_nodes(self, ):
		# tested
		nodes = {}
		for i in range(self.depth()+1):
			nodes[i] = []
			for node in self.expand_tree(mode=2):
				if self.level(node) == i: 
					nodes[i].append(node)
		return nodes

	def get_bfs_data(self, ):
		# tested
		nodes = self.get_bfs_nodes()
		return {i: list(map(lambda x: self[x].data, nodes[i])) for i in range(self.depth() + 1)}

	def get_dfs_nodes(self, ):
		# tested
		return self.paths_to_leaves()

	def get_dfs_data(self, ):
		# tested
		paths = self.get_dfs_nodes() # list
		return [list(map(lambda x: self[x].data, path)) for path in paths]

	def init_nodes_data(self, value = 0):
		# tested
		for id in self.expand_tree(mode=1):
			self[id].data = value

	def from_paths(self, paths):
		# tested
		# check duplicated son-fathur relationship
		for path in paths:
			current_node = self.root
			for nid in path:
				children_ids = [n.identifier for n in self.children(current_node)]
				if nid not in children_ids: self.create_node(identifier=nid, parent=current_node)
				current_node = nid
	'''
	def from_child_father(self, ):
		return None
	'''

	def from_pickle(self, file: str):
		# tested
		with open(file, 'rb') as f: 
			stree = pickle.load(f)
		return stree

	def path_to_node(self, node_id: str):
		# tested
		nid = node_id
		path_r = []
		while nid != 'root':
			path_r.append(nid)
			nid = self[nid].bpointer
		path_r.append('root')
		path_r.reverse()
		return path_r
	
	def fill_with(self, data: dict):
		# tested
		for nid, val in data.items():
			self[nid].data = val

	def update_value(self, ):
		# tested
		all_nodes = [nid for nid in self.expand_tree(mode=2)][::-1]
		for nid in all_nodes:
			d = sum([node.data for node in self.children(nid)])
			self[nid].data = self[nid].data + d

	def to_pickle(self, file: str):
		# tested
		with open(file, 'wb') as f:
			pickle.dump(self, f)

	def get_matrix(self, dtype = np.float32):
		# tested
		paths_to_leaves = self.paths_to_leaves()
		ncol = self.depth() + 1
		nrow = len(paths_to_leaves)
		Matrix = np.zeros(ncol*nrow, dtype=dtype).reshape(nrow, ncol)

		for row, path in enumerate(paths_to_leaves):		
			for col, nid in enumerate(path):
				Matrix[row, col]= self[nid].data
		return Matrix

	def to_matrix_npy(self, file: str, dtype = np.float32):
		# tested
		matrix = self.get_matrix(dtype=dtype)
		np.save(file, matrix)

	def copy(self, ):
		# not working
		return super_tree(self.subtree(self.root), deep=True) 

	def remove_levels(self, level: int):
		# tested
		nids = list(self.expand_tree(mode=1))[::-1]
		for nid in nids:
			if self.level(nid) >= level:
				self.remove_node(nid)  # check

	def save_paths_to_csv(self, file: str, fill_na=True):
		# tested
		paths = self.paths_to_leaves()
		df = pd.DataFrame(paths)
		if fill_na:
			df.fillna('')
		df.to_csv(file, sep = ',')

	'''
	def from_paths_csv(self, file: str):
		# debug needed
		df = read_csv(file, header=0, sep=',')
		def remove_null_str(x): 
			while '' in x: x.remove('')
			return x
		paths = map(remove_null_str, [list(df.iloc(0)[i]) for i in df.axes[0]])
		return self.from_paths(paths)
	
	def from_ete3_species(self, name: str):
		
		# return a subtree of species name, data is retrived from NCBI Taxonomy database
		
		return None
	'''


class data_loader(object):

	def __init__(self, path: str, ftype='.tsv'):
		# tested
		self.ftype = ftype
		self.paths = self.get_file_paths(path)
	
	def get_file_paths(self, path: str):
		# tested
		return [os.path.join(root, file)
		for root, dirs, files in os.walk(path) 
		for file in files 
		if os.path.splitext(file)[1] == self.ftype]
	
	def get_sample_count(self, ):
		self.get_paths_keep()
		split_paths = list(map(lambda x: os.path.split(x)[0].split('/')[-1], self.paths_keep))
		self.sample_count = {i: split_paths.count(i) for i in set(split_paths)}
		return self.sample_count

	def get_paths_keep(self, ):
		self.load_error_list()
		self.paths_keep = list(filter(lambda x: x not in self.error_list, self.paths))
		return self.paths_keep

	def get_data(self, header=1):
		# tested
		self.get_paths_keep()
		self.data = list(map(lambda x: read_csv(x, sep='\t', header=header), self.paths_keep))
		# self.data = map(lambda x: x.iloc(1)[1:], self.data)
		return self.data

	def save_error_list(self, ):
		# tested
		with open('tmp/error_list', 'w') as f:
			f.write('\n'.join(self.error_msg.keys()))
	
	def load_error_list(self, ):
		# tested
		with open('tmp/error_list', 'r') as f:
			self.error_list = f.readlines()

	def check_data(self, header=1):
		# tested
		self.status = {}
		for path in self.paths:
			try: 
				f = read_csv(path, header=header, sep='\t')
				self.status[path] = [self.check_ncols(f), 
				self.check_col_name(f), 
				self.check_values(f)]

			except:
				self.status[path] = ['IOError','IOError','IOError']
		self.error_msg = {path: status 
		for path, status in self.status.items() 
		if list(set(status)) != ['True']}

	def check_ncols(self, File):
		# tested
		if File.shape[1] == 3:
			return 'True'
		else:
			return 'False'

	def check_col_name(self, File):
		# tested
		Colnames = File.columns.tolist()
		if Colnames[0] in ['# OTU ID', '#OTU ID']:
			return 'True'
		else:
			return 'False'

	def check_values(self, File):
		# tested
		Na_status = File.isna().values.any()
		Neg_status = list(set([int(ele) >=0 for ele in File[File.columns.tolist()[1]]]))
		if Na_status == True:
			return 'Na'
		elif len(Neg_status)==0 and Neg_status[0] == False:
			return 'Negtive value error'
		else:
			return 'True'


class id_converter(object):
	def __init__(self, ):
		pass
	
	def convert(self, ids_path: str, sep):
		# tested, use path
		ids = ids_path.split(sep)
		tail = ids[-1].split('__')[-1]
		ids = list(map(lambda x: x+tail if x[-2:] == '__' else x, ids))
		ids = [sep.join(ids[0:i]) for i in range(1, len(ids)+1)]
		self.nid = ids
		return ids
