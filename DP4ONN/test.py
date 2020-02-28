from utils import species_tree; tree = species_tree(); tree.create_node(identifier='root'); tree.from_paths([['aaa','bbb','ccc', 'dd','fff', 'ggg','hhh'], ['ddd','eee']]); tree.init_nodes_data(2); tree.update_value(); tree.to_matrix_npy('ex')


tree.show()
