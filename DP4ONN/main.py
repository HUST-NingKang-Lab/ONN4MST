import argparse
import os
import joblib
from utils import super_tree, data_loader, id_converter
import numpy as np

des = '''
This is an integrated data preprocessor for Ontology-aware Neural Network.

Work mode:
\tcheck mode: check all of your data files, the error data file are saved in tmp/ folder.
\tbuild mode: de novoly build a species tree using your own data.
\tconvert mode: convert tsv file from EBI MGnify database to model acceptable n-dimensional array.
\tcount mode: count the number of samples in each biome.
\tmerge mode: merge multiple npz files to a single npz.
'''
parser = argparse.ArgumentParser(description=des, 
                                formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("mode", type=str, choices=['check', 'build', 'convert', 'count', 'merge'], default='convert',
                    help = "work mode of the program. default: convert")
parser.add_argument("--n_jobs", type=int, default=1, 
                    help = 'the number of processors to use. default: 1')
parser.add_argument('--input_dir', type=str, default='data/',
                    help = 'input directory, must be parent folder of biome folders. default: data/')
parser.add_argument('--output_dir', type=str, default='matrices/',
                    help = 'output directory. default: matrices')                    
parser.add_argument('--tree', type=str, default='tree/',
                    help = 'the directory of trees (species_tree.pkl and biome_tree.pkl). default: tree/')

args = parser.parse_args()
loader = data_loader(path = args.input_dir)
if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
if not os.path.isdir(args.tree): os.mkdir(args.tree)

if args.mode == 'check':
    # tested
    loader.check_data(header=1)
    loader.save_error_list()

elif args.mode == 'build':
    # tested
    converter = id_converter()
    paths = []
    for i in map(lambda x: x.iloc(1)[2], loader.get_data(header=1)): paths.extend(i) 
    paths = map(lambda x: converter.convert(x, sep=';'), paths)
    # print(list(paths)[0:5])
    stree = super_tree()
    stree.create_node(identifier='root')
    stree.from_paths(paths)
    # stree.show()
    stree.to_pickle(file=os.path.join(args.tree, 'species_tree.pkl'))

    biomes = map(lambda x: converter.convert(x, sep='-'), os.listdir(args.input_dir))
    biomes = list(map(lambda x: x[1:], biomes))
    print(biomes)
    btree = super_tree()
    btree.create_node(identifier='root')
    btree.from_paths(biomes)
    btree.show()
    btree.to_pickle(file=os.path.join(args.tree, 'biome_tree.pkl'))


elif args.mode == 'convert':
    tree = super_tree()
    converter = id_converter()
    stree = tree.from_pickle(os.path.join(args.tree, 'species_tree.pkl'))
    biome_tree = tree.from_pickle(os.path.join(args.tree, 'biome_tree.pkl'))
    data = []
    for i in map(lambda x: x.iloc(1)[1:], loader.get_data(header=1)): data.append(i.values.tolist())
    data = map(lambda x: {converter.convert(sp[1], sep=';')[-1]: sp[0] for sp in x}, data)

    biomes = map(lambda x: converter.convert(os.path.split(x)[0].split('/')[-1], 
                                            sep='-'), 
                                            loader.paths_keep)
    biomes = list(biomes)
    matrices = []; labels = []
    for index, sample in enumerate(data):
        species_tree = stree.copy()
        species_tree.init_nodes_data(value = 0)
        species_tree.fill_with(data = sample)
        species_tree.update_value()
        Sum = species_tree['root'].data
        species_tree.remove_levels(species_tree.depth())
        matrices.append(species_tree.get_matrix()/Sum) # relative abundance
        biome_tree.init_nodes_data(value = 0)
        biome_tree.fill_with(data = {biome: 1 for biome in biomes[index]})
        bfs_data = biome_tree.get_bfs_data()
        labels.append([np.array(bfs_data[i], dtype=np.float32) for i in range(biome_tree.depth() + 1)])
    labels = [[np.array(label[i]) for label in labels] for i in range(len(labels[0]))]
    np.savez(os.path.join(args.output_dir, 'matrices.npz'), matrices=matrices,
        label_0 = labels[0], label_1 = labels[1], label_2 = labels[2], label_3 = labels[3],
        label_4 = labels[4], label_5 = labels[5])

elif args.mode == 'count':
    # tested
    sample_count = loader.get_sample_count()
    print(sample_count) # save needed

elif args.mode == 'merge': 
    print('这是一个待添加的功能')
