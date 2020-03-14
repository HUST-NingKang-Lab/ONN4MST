# DP4ONN 
## Data Preprocessor for Ontology-aware Neural Network

This work can make the data preprocessing and sample statistics of the Ontology-aware Neural Network easier. Due to the use of three predefined classes, the simplicity and efficiency of the code are greatly improved. For very time-consuming big data calculations, any minor data or program errors can cost you days or even weeks. Using this program to check the integrity of all data and error values before processing can greatly reduce the probability of your program running errors. 

Good luck:smile:

## Features

- Multiple working modes to meet different needs;

- Generate tree directly from node path;
- Save and read tree, get a deep copy of tree;
- Obtain node ids and values of breadth-first traversal and depth-first traversal in batches;
- Initialize, change, and update node values in batches;
- Batch read, check, and filter large amounts of data;
- NCBI taxonomy database for species ID conversion and species tree generation.

For more features, check out **Getting Started**

## Dependencies

- [Treelib 1.5.5](https://github.com/caesar0301/treelib)
- [Pandas 1.0.1](https://pandas.pydata.org/)
- [Numpy 1.18.1](www.numpy.org)
- [Etetool](etetoolkit.org/)

## Getting Started

- Clone this repo

```shell
git clone https://github.com/AdeBC/DP4ONN && cd DP4ONN
```

- Check data 

```shell
python main.py check --input_dir data
```

- Construct tree de novoly

```shell
python main.py build --input_dir data --tree tree
```

- Convert 'tsv' files to model-acceptable 'npz' file

```shell
python main.py convert --input_dir data --tree tree --output_dir matrices
```

- Count the number of samples in each biome

```shell
python main.py count --input_dir data --output_dir sample_count
```

- Merge multiple 'npz' files to a single 'npz'

```shell
python main.py merge --input_dir matrices
```

## To-do

- Parallelize
- Add module
  - Interact with ete3
  - Merge npz
  - Sample count
- Bug fix

## Maintainer

|   Name    |          Email          |                         Organization                         |
| :-------: | :---------------------: | :----------------------------------------------------------: |
| Hui Chong | ch37915405887@gmail.com | Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology |

