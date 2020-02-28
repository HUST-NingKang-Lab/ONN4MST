import os
from treelib import Node, Tree
import pickle

'''
重新建树
消除不同路径上的重复节点
'''

def main():
	tree = loadBiomeTree()
	raw_data = loadRawData()
	tree = fillTree(raw_data, tree)
	# check(tree)
	tree = updateTree(tree)
	# check(tree)
	tree = removeHierarchy(tree, 6)
	tree = removeHierarchy(tree, 5)
	formatExport(tree)


def check(tree):
	for i in range(tree.depth()):
		sum =0
		for id in tree.expand_tree(mode=2):
			if tree.level(id) == i: 
				sum = sum + tree[id].data
		print('level {} in total: {}'.format(i, sum))


def formatExport(tree):
	for path in tree.paths_to_leaves():
		if len(path) == 5:
			data = tree[path[-1]].data
			path = '-'.join(path)
			print('{}:{}'.format(path, data))


def fillTree(raw_data, tree):
	for node in tree.expand_tree(mode=2):
		tree[node].data = 0
	for path in raw_data:
		# tree[path[0][-1]].data = int(path[1])
		tree[path[0][-1]].data = tree[path[0][-1]].data+int(path[1])
		
		# print(tree[path[0][-1]].data)
	return tree


def updateTree(tree):
	# 准备倒序遍历
	print('\tupdating tree......', end='')
	all_nodes = [id for id in tree.expand_tree(mode=2)][::-1]
	# print('all_nodes[-5:]:', all_nodes[-5:])
	for id in all_nodes:
		# print(tree.level(id))
		Data = sum([node.data for node in tree.children(id)])
		tree[id].data = tree[id].data + Data
		# tree[id].data = Data
		# print('tree[{}].data:{}'.format(id, tree[id].data))
	print('done !')
	# print('root.value: ', tree['root'].data)
	return tree


def removeHierarchy(tree, level):
	for id in tree.expand_tree(mode=2):
		if tree.level(id) == level:
			tree.remove_node(id)
			# print('{} removed !'.format(id))
	return tree


def loadBiomeTree():
	with open('tree/biome_tree.pydata', 'rb') as f:
		tree = pickle.load(f)
		# tree.show()
	return tree


def loadRawData():
	with open('data/raw_data.txt', 'r') as f:
		raw_data = f.readlines()
	for index,data in enumerate(raw_data):
		raw_data[index] = data.rstrip('\n').split(':')
		path = raw_data[index][0].split('/')[-1]
		raw_data[index][0] = path.split('-')
	# print(sum([int(ele[1]) for ele in raw_data]))
	return raw_data
	

if __name__ == '__main__':
	main()
