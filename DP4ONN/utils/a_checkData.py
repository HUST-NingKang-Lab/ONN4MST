from pandas import read_csv
import os
import pickle
from multiprocessing import Pool


def main():
	Dir = '/home/qiuhao/ONN/ONNdata'
	file_paths = findAllFiles(Dir)
	status = {}
	error_msg = {}
	data ={}
	for path in file_paths:
		# print('processing file:', path, end='')
		status[path] = ['','','']
		try: 
			f = read_csv(path, sep='\t', header=1, encoding='utf-8')
			status = check(path, f, status)
			data[path] = f
			# print('......done !')
		except:
			status[path] = ['IOError','IOError','IOError']
			data[path] = 'null'
			# print('......IOError')	
	for key, value in status.items():
		if list(set(value)) != ['True']:
			print('{} in {}'.format(value, key))
			error_msg[key] = value

	with open('tmp/check_error_msg.pydata', 'wb') as f:
		pickle.dump(error_msg, f, protocol=pickle.HIGHEST_PROTOCOL)
	with open('tmp/all_files.pydata', 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def check(path, f, status):
	p = Pool(3)
	status[path][0] = p.apply_async(checknCols, args=(f, )).get()
	status[path][1] = p.apply_async(checkColName, args=(f, )).get()
	try:
		status[path][2] = p.apply_async(checkValues, args=(f, )).get()
	except:
		status[path][2] = 'Error'
	p.close()
	p.join()
	return status


def findAllFiles(path): return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if os.path.splitext(file)[1] == ftype]
	find_files = []
	for root, dirs, files in os.walk(path):   
		for file in files:
			pathWithFile = os.path.join(root,file)	
			#root当前文件路径本身+	file（查询到的文件名称）组成绝对路径
			# print(pathWithFile)
			if os.path.splitext(pathWithFile)[1] == '.tsv':
				find_files.append(pathWithFile)
			else:
				pass
	return find_files


def loadDataFromDirectory(directory):
	
	return data


def checknCols(File):
	if File.shape[1] == 3:
		return 'True'
	else:
		return 'False'
	

def checkColName(File):
	Colnames = File.columns.tolist()
	if Colnames[0] in ['# OTU ID', '#OTU ID']:
		return 'True'
	else:
		return 'False'


def checkValues(File):
	Na_status = File.isna().values.any()
	Neg_status = list(set([int(ele) >=0 for ele in File[File.columns.tolist()[1]]]))
	if Na_status == True:
		return 'Na'
	elif len(Neg_status)==0 and Neg_status[0] == False:
		return 'Negtive value error'
	else:
		return 'True'


if __name__ == '__main__':
	main()
