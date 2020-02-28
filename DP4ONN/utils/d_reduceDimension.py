from treelib import Tree, Node
import pickle5 as pickle
import os
from pandas import read_csv
from e_generateMatrices_Labels import loadData
from a_checkData import findAllFiles


'''
	细节待定
'''


def main():
	s_tree = loadSpeciesTree()
	data = loadData()
	data = manipulateData(data)
	tree = fillTree(tree, data)
	tree = updateTree(tree)
	tree = removeHierarchy(tree, 's')

	
