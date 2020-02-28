import os
import pandas as pd

# set your dir
directory = '/home/chonghui/project/FEAST_data/out'

def main():
	biomes = [os.path.join(directory, f) for f in os.listdir(directory)]
	table = {sample_id.rstrip('.tsv'): os.path.split(biome)[1] for biome in biomes for sample_id in os.listdir(biome)}
	# iprint([i for i in table.items()][0:50])
	df = pd.DataFrame(columns = ['Sample id','Biome'])
	for i, j in table.items():
		df = df.append({'Sample id': i,'Biome': j}, ignore_index = True)
	print(df.iloc(0)[0:10])
	df.to_csv('sample_biome_table.tsv', sep = '\t')

if __name__ == '__main__':
	main()
















