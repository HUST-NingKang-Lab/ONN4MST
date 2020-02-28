#!/home/chonghui/envs/miniconda3/bin/python

import pickle5 as pickle
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from treelib import Node, Tree
from pandas import read_csv, DataFrame
from numpy import matrix, savez, array, float32, zeros
from utils import updateTree
# from multiprocessing import Pool
from b_buildSpeciesTree import loadErrorList
from a_checkData import findAllFiles
import sys


'''
changes to apply
	更新树的代码需要重写，因为重复id问题解决了，此次更新之后应该能保证数据完整性，记得check
	root.data != Sum of tsv column 1, why? check out!`
'''


def main():
	biome_index = int(sys.argv[1])
	s_tree = loadSpeciesTree()
	b_tree = loadBiomeTree()
	biomes = getBiomes()
	error_list = loadErrorList()
	# sample_count = {}
	# sample_count = loadSampleCount()
	'''
	for index, biome in enumerate(biomes):
		if biome+'.npz' not in os.listdir('matrices'):
			print('processing {}/{}: {}......'.format(index, len(biomes), biome))
			num = generateMatrices_Labels(biome, s_tree, b_tree, error_list)
			# sample_count[biome] = num
			print('done!')
			# print('sample_count of biome:', num)
	'''
	print('processing {}/{}: {}......'.format(biome_index, len(biomes), biomes[biome_index]))
	if biomes[biome_index]+'.npz' not in os.listdir('matrices'):
		num = generateMatrices_Labels(biomes[biome_index], s_tree, b_tree, error_list)
	
	# 保存其他信息
	# saveCountResult(sample_count) # 保存统计结果
	saveLeaves([node.identifier for node in s_tree.leaves('root')]) 
	# 保存叶子信息


def generateMatrices_Labels(biome, s_tree, b_tree, error_list):
	'''
	以一个biome文件夹为单位生成矩阵和标签并进行存储
	返回该biome的统计结果
	'''
	
	directory = '/home/qiuhao/ONN/ONNdata/'+biome
	data = loadDataFromTsvFiles(directory, error_list) # 导入数据
	# 导入时已经去除了坏的数据
	# print('generating matrices and labels......')
	matrices = [''] * len(data) # 更改初始化长度，as list type
	labels = [''] * len(data) # as list type
	'''
	processes = 8   # 进程数
	pool = Pool(processes)
	for index in range(0, len(data), processes):
		for i in range(processes):
			# print(index+i,' ', end='')
			if (index+i)%80 ==0:
				print('    generating matrix and labels {}/{}'.format(index, len(data)))
			if index+i < len(data):
				matrix, label = pool.apply_async(generateMatrix_Label, args=(data[index+i], s_tree, b_tree, biome)).get()
				matrices[index+i] = matrix   # 更改：使用索引来赋值，而不是append
				labels[index+i] = label
			else:
				pass
	pool.close()
	pool.join()
	'''
	for i,tsv in enumerate(data):
		print('    generating matrix and labels {}/{}'.format(i, len(data)))
		matrix, label = generateMatrix_Label(tsv, s_tree, b_tree, biome)
		matrices[i] = matrix
		labels[i] = label
	
	saveMatricesAndLabels(matrices, labels, biome)
	return len(data)


def generateMatrix_Label(tsv, s_tree, b_tree, biome):		
	'''
	以一个tsv文件为单位生成单个矩阵和标签并返回
	'''
	#bug fixes
	print('-'*60)
	s_tree_copy = Tree(s_tree.subtree(s_tree.root), deep=True) # new tree
	# print(s_tree.size() == s_tree_copy.size())
	tsv = manipulateData(tsv)                                  # dataframe to list
	s_tree_copy = fillTree(tsv, s_tree_copy)                   # fill tree
	s_tree_copy = updateTree(s_tree_copy)                      # update data for each node
	Sum = sum([int(i[0]) for i in tsv])
	print('\troot.data == sum(tsv)?......', s_tree_copy['root'].data == Sum)
	s_tree_copy = removeHierarchy(s_tree_copy, 's')            # remove hierarchy 8                                    
	
	matrix = generateMatrix(s_tree_copy)                       # generate matrix
	# matrix = matrix.astype('float32')                          # transform data type
	biome_path = useBiomeTreePath(biome)
	label, nodes = generateLabels(b_tree, biome_path)               # generate label
	# label = [l.astype(float32) for l in label]               # transform data type
	# for key, value in label.items():
		# label[key] = value.astype(float32)
	
	'''
	for i,hierarchy in enumerate(label.values()):
		if i < len(biome):
			print('biome[{}]: {}'.format(i, biome[i]))
			print('nodes[{}]: {}'.format(i, nodes[i]))
			print('labels[{}]: {}'.format(i, hierarchy))
	for i in nodes:
		print('nodes[{}]: {}'.format(i, nodes[i]))
	'''
	return matrix, label


def useBiomeTreePath(biome):
	biome_path = []
	biome = biome.split('-')
	for i,ele in enumerate(biome):
		biome_path.append('-'.join(biome[0:i+1]))
	return biome_path[1:]


def manipulateData(data):
	print('\tmanipulating data......', end='')
	# data = removeDuplicates(data)
	data = data[:].values.tolist()
	for rownum, row in enumerate(data):
        #print(row)
		data[rownum][1] = row[1].split(';')
		# 倒序遍历
		for index, element in enumerate(data[rownum][1][::-1]): 
			# print(element[-2:])
			if element[-2:] == '__':
				data[rownum][1][-index-1] = element + data[rownum][1][-index].split('__')[-1]
		data[rownum][1] = ';'.join(data[rownum][1])
	print('done!')
	return data


def saveCountResult(sample_count_dict):
	with open('sample_count/sample_count.pydata', 'wb') as f:
		pickle.dump(sample_count_dict, f, protocol=5)
	print('sample count data saved !')


def loadSpeciesTree():
	with open('tree/species_tree.pydata', 'rb') as f:
		tree = pickle.load(f)
	return tree


def loadBiomeTree():
	with open('tree/biome_tree.pydata', 'rb') as f: 
		tree = pickle.load(f)
		# tree.show()
	return tree


def getBiomes():
	return os.listdir('/home/qiuhao/ONN/ONNdata')


def removeHierarchy(tree, hierarchy_id):
	print('\tremoving hierarchy......', end='')
	for node in tree.expand_tree(mode=2):
		if node.split(';')[-1].split('__')[0] == hierarchy_id:
			# print('node {} removed!'.format(node))
			tree.remove_node(node)
	print('done')
	return tree


def findNodesInDepth(tree, depth_id):
	nodes = []  
	for node_id in tree.expand_tree(mode = 2):
		if node_id[0] == depth_id:
			nodes.append(node_id)
	return nodes


def loadDataFromTsvFiles(Dir, error_list):
	print('    loading data......', end='')
	all_files = findAllFiles(Dir)
	data = []
	for index, file in enumerate(all_files):
		if file not in error_list :
			# print(file)
			tsv = read_csv(file, sep= '\t', header=1, encoding='utf-8')			   
			# print(tsv.columns)
			try:  # 应付文件之间的差异
				tsv = tsv.drop(columns= ['# OTU ID'])
			except:
				tsv = tsv.drop(columns= ['#OTU ID'])
			data.append(tsv)
		else:
			pass
			# print(file)
	print('done!')
	return data


def fillTree(data, tree):
	print('\tfilling tree......', end='')
	for id in tree.expand_tree(mode=2):
		tree[id].data = 0
	
	for row in data:
			tree[row[1]].data = int(row[0])
			# print('row[0]:{}, row[1]:{}, data:{}'.format(row[0], row[1], tree[row[1]].data))
	
	print('done !')
	return tree


'''
def updateTree(tree):
	
	# 更新树上每一个节点的值
	# 使得删节点并不影响后续数值的计算
	
	leaves = tree.leaves()
	paths_to_leaves = tree.paths_to_leaves()
	tree['root'].data = sum([node.data for node in tree.leaves()])
	# root需要额外更新
	# print(paths_to_leaves)
	for node in tree.expand_tree(mode=2):
		if tree[node].data == 0:
			tree[node].data =sum([node.data for node in tree.leaves(node)]) 
			# print(type(tree[node].data))
	return tree
'''

def generateMatrix(tree):
	paths_to_leaves = tree.paths_to_leaves()
	ncol = max([len(path) for path in paths_to_leaves])
	nrow = len(paths_to_leaves)
	Matrix = zeros(ncol*nrow, dtype=float32).reshape(nrow, ncol)
	# initialize matrix 
	# for each row in matrix
	for row, path in enumerate(paths_to_leaves):		
		# for each element in row
		for col, node in enumerate(path):
			# print(node, tree[node].data)
			Matrix[row, col]= tree[node].data/tree['root'].data
			# 相对丰度计算
	# print('done !')
	return Matrix


def generateLabels(tree, biome):
	# print('generating labels......', end='')
	labels = {}
	nodes = getBiomeHierarchies(tree)
	for index,Nodes in enumerate(nodes.values()):
		labels[index] = zeros(len(Nodes), dtype=float32)
		for Index, node in enumerate(Nodes):
			if index < len(biome):
				# print('\tbiome[index]:', biome[index])
				# print('\tnode:', node)
				if node == biome[index]:
					labels[index][Index] = 1 				
					# print('matched !')
	# print('done !')
	return labels,nodes


def getBiomeHierarchies(tree):
	nodes = {}
	for i in range(tree.depth()):
		nodes[i] = []
		for node in tree.expand_tree(mode=2):
			if tree.level(node) == i+1: 
				nodes[i].append(node)
	return nodes


def saveMatricesAndLabels(matrices, labels, biome):
	# save matrices and labels	
	new_labels= {'label_0': [value[0] for value in labels],\
				'label_1': [value[1] for value in labels],\
				'label_2': [value[2] for value in labels],\
				'label_3': [value[3] for value in labels],\
				'label_4': [value[4] for value in labels],\
				'label_5': [value[5] for value in labels]}
	
	if not os.path.isdir('matrices'): os.mkdir('matrices')
	savez('matrices/'+biome, matrices=matrices, label_0=new_labels['label_0'], \
												label_1=new_labels['label_1'], \
												label_2=new_labels['label_2'], \
												label_3=new_labels['label_3'], \
												label_4=new_labels['label_4'], \
												label_5=new_labels['label_5'])
	print('Matrices saved !')


def saveLeaves(leaves):
	if not os.path.isdir('leaves'): os.mkdir('leaves')
	with open('leaves/leaves_of_tree.txt', 'w') as f:
		f.write('\n'.join(leaves))
	print('Leaves saved !')


if __name__ == '__main__':
	main()





