#!/home/chonghui/envs/miniconda3/bin/python

from e_generateMatrix_Label import generateMatrix_Label, loadSpeciesTree, loadBiomeTree, loadErrorList, saveMatricesAndLabels
from a_checkData import findAllFiles
import os
import sys
from pandas import read_csv


def main():
	tsv_index = int(sys.argv[1])
	btree = loadBiomeTree()
	stree = loadSpeciesTree()
	errorlist = loadErrorList()
	biome = 'root-Host-associated-Human-Digestive_system-Large_intestine-Fecal'
	directory = '/home/qiuhao/ONN/ONNdata/'+biome
	print('loading data......')
	data = loadDataFromTsvFiles(directory, errorlist, tsv_index)
	matrices = [' ']*len(data)
	labels = [' ']*len(data)
	
	for i,tsv in enumerate(data):
		print('    generating matrix and labels {}/{}'.format(i, len(data)))
		matrix, label = generateMatrix_Label(tsv, stree, btree, biome)
		matrices[i] = matrix
		labels[i] = label
	
	saveMatricesAndLabels(matrices, labels, biome+'_'+str(tsv_index))


def loadDataFromTsvFiles(Dir, error_list, index):
	print('    loading data......', end='')

	all_files = findAllFiles(Dir)
	
	if index+1000 < len(all_files):
		indexr = index +1000
	else:
		indexr = len(all_files)

	all_files = all_files[index:indexr]
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



if __name__ == '__main__':
	main()
