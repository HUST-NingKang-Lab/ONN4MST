from os import listdir
from pandas import read_csv
from treelib import Node, Tree
from a_checkData import findAllFiles
from multiprocessing import Pool
import pickle5 as pickle
import os

'''
Attention：
	重新编写函数，将路径信息作为节点的id，以消除重复id
	去除重复路径，提高建树的效率
'''


def main():
	directory = '/home/qiuhao/ONN/ONNdata'
	error_list = loadErrorList()
	# print(error_list)
	all_files = findAllFiles(directory)
	tree = Tree()
	tree.create_node('root', 'root')
	batch_size = 1000
	for i in range(0, len(all_files), batch_size):
		print('reading......{}/{}'.format(i,len(all_files)))
		if i+batch_size < len(all_files):
			data = loadDataFromTsvFiles(all_files[i:i+batch_size], error_list)
		else:
			data = loadDataFromTsvFiles(all_files[i:], error_list)
		# print('before removing duplicates, len(data)=', len(data))
		# print(data[0:3])
		data = manipulateData(data)
		# print('after removing duplicates, len(data)=', len(data))
		# print('data[0:3]', data[0:3])
		data = usePathInformation(data)
		# print('data[0:3]', data[0:3])
		tree = buildTree(data, tree)
		del(data)
		# print('tree.size: ',tree.size())
	
	# tree.show()
	if not os.path.isdir('tree'):
		os.mkdir('tree')
	saveTree(tree, 'tree/species_tree.pydata', 'tree/species_tree_visualized.txt')


def usePathInformation(data):
	# data = removeDuplicates(data)
	new_data = []
	for i, row in enumerate(data):
		new_data.append([])
		for j, ele in enumerate(row):
			l = list(row[0:j+1])
			# print(l)
			new_data[i].append(';'.join(l))
			# print(new_data[i][j])
	
	return new_data
	


def loadErrorList():
	with open('tmp/error_list', 'r') as f:
		error_list = f.read()
	return error_list.split('\n')


def loadDataFromTsvFiles(all_files, error_list):
	print('loading data......', end='')
	data = []
	for index, file in enumerate(all_files):
		# if index%1000 == 0: 
			# print('reading......{}/{}'.format(index, len(files)))
		if file not in error_list:
			# print(file)
			tsv = read_csv(file, \
					  	   sep= '\t', \
						   header=1, \
						   encoding='utf-8', \
						   engine='python')
			try:
				tsv = tsv.drop(columns= ['# OTU ID'])
			except:
				tsv = tsv.drop(columns= ['#OTU ID'])
			keep = tsv[:].values.tolist()
			keep = [ele[1] for ele in keep]
			data.extend(keep)
	
	print('\tdone!')
	return data


def removeDuplicates(data):
	new_data = []
	for i in data:
		if i not in new_data:
			new_data.extend(data)
	return new_data


def manipulateData(data):
	print('manipulating data......', end='')
	# data = removeDuplicates(data)
	for rownum, row in enumerate(data):
		#print(row)
		data[rownum] = row.split(';')
		# 倒序遍历
		for index, element in enumerate(data[rownum][::-1]): 
			# print(element[-2:])
			if element[-2:] == '__':
				data[rownum][-index-1] = element + data[rownum][-index].split('__')[-1]
	
	print('\tdone!')
	return data


def buildTree(data, tree):
	print('building tree......', end='')
	for index in range(0, len(data)):
		# if index%100000 ==0 : 
			# print('processing......{}/{}'.format(index, len(data)))
		addNodesToTree(tree, data[index])
	print('\tdone!')
	return tree


def addNodesToTree(tree, nodes):
	
	# 遍历树：按顺序查找，找不到就新建节点
	current_node = 'root'
	for node in nodes:
		childrens_identifier = [n.identifier for n in tree.children(current_node)]  # operation needed 
		if node not in childrens_identifier:
			tree.create_node(node.title(), node, parent=current_node, data=0)
		
		current_node = node

	return tree


def saveTree(tree, tree_filepath, display_filepath):
	# tree.show()
	print('tree.depth: ', tree.depth())
	with open(tree_filepath, 'wb') as handle:
		pickle.dump(tree, handle, protocol=5)
	# tree.save2file(display_filepath)
	print('data saved !')


if __name__=='__main__':
	main()


