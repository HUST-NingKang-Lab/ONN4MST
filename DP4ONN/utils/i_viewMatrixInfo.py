#!/home/chonghui/envs/miniconda3/bin/python


from numpy import load, array, float32
import os

def getMatricesFiles():
	return os.listdir('/home/chonghui/project/onn4mdm/matrices')


def loadData():
	# raw = []
	data = {}
	for key in ['matrices','label_0','label_1','label_2','label_3','label_4','label_5']:
		data[key] = []
	for i in getMatricesFiles():
		l = load('/home/chonghui/project/onn4mdm/matrices/'+i, allow_pickle=True)
		# print(l['matrices'])
		if l['matrices'].shape != (0,):
			# raw.append(l)
			for key,value in l.items():
				# print(len(value))
				for ele in value:
					# print('len(key):', len(data[key]))
					data[key].append(ele)
		else:
			# print('null found !')
			pass
	matrices = array(data['matrices'], dtype=float32)
	label_0 = array(data['label_0'], dtype=float32)
	label_1 = array(data['label_1'], dtype=float32)
	label_2 = array(data['label_2'], dtype=float32)
	label_3 = array(data['label_3'], dtype=float32)
	label_4 = array(data['label_4'], dtype=float32)
	label_5 = array(data['label_5'], dtype=float32)
	
	return matrices, label_0, label_1, label_2, label_3, label_4, label_5


if __name__ == "__main__":
	matrices, label_0, label_1, label_2, label_3, label_4, label_5 = loadData()
	
	print('='*67)
	print('{} samples in total'.format(len(matrices)))
	print('Matrix And Label Information......')
	# print('Total : {} matrices, {} labels'.format(len(matrices), len(label_0)))
	print('-'*30+'matrices[0]'+'-'*30)
	print(matrices[0])
	print('shape of matrices', matrices[0].shape)
	print('elements dtype of matrix: ', matrices[0].dtype)
	print('-'*30+'labels[0]'+'-'*30)
	print(' label_0:', label_0[0])
	print(' label_1:', label_1[0])
	print(' label_2:', label_2[0])
	print(' label_3:', label_3[0])
	print(' label_4:', label_4[0])
	print(' label_5:', label_5[0])
	print('elements dtype of label: ', label_0[0].dtype)
	print('-'*30+'Columns'+'-'*30)

	print(' 1st: Sum'.ljust(10, ' '), end='')
	print(' 2nd: Sk'.ljust(10, ' '), end='')
	print(' 3rd: K'.ljust(10, ' '), end='')
	print(' 4th: P'.ljust(10, ' '), end='')
	print(' 5th: C'.ljust(10, ' '), end='')
	print(' 6th: O'.ljust(10, ' '), end='')
	print(' 7th: F'.ljust(10, ' '), end='')
	print(' 8th: G'.ljust(10, ' '))
