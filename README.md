# ONN4MDM
Ontology-aware Neural Network for Mircobiome Data Mining!

![Training a ONN source tracking model and using it for microbiome samples' source tracking](https://github.com/HUST-NingKang-Lab/ONN4MDM/blob/master/image/Figure2_main.png)
<img src="https://github.com/HUST-NingKang-Lab/ONN4MDM/blob/master/image/Figure2_main.png" width="300" height="450" />

This program is designed to perform fast and accurate biome source tracking. The ontology of biome is organized to a biome tree, which have six layers. From the root nodes to the leaf nodes, each node has only one parent node. The Neural Network is also organized in six layers, which could produce a hierarchical classification result. The input is a species realtive abundance ".tsv" file, which can be produced by Qiime or get from EBI. The output is a biome source ".txt" file, whcih shows you where the input sample comes from.

The preprocessor can make the data preprocessing and sample statistics of the Ontology-aware Neural Network easier. Due to the use of three predefined classes, the simplicity and efficiency of the code are greatly improved. For very time-consuming big data calculations, any minor data or program errors can cost you days or even weeks. Using this program to check the integrity of all data and error values before processing can greatly reduce the probability of your program running errors.

## Install
Download the zip archive from this [repository][1], then unzip the archive.
## Function
The program can be used for biome source tracking.
### tsv convert to npz
The input file format of ONN is the ".npz" file. Before ONN, you need to convert the original input ".tsv" file into ".npz" file. The script "scripts/preprocess.py" could work for it.
### source tracking
If you have successfully converted the ".tsv" file into ".npz" file, then you could run the script "scripts/ONN4MDM.py" for biome source tracking. Besides, you need also indicate a trained model. We have provided a well trained model as the default model.
## Dependencies

- for data preprocessing
  - [python 3.7][6]
  - [Treelib 1.5.5][2]
  - [Pandas 1.0.1][3]
  - [Numpy 1.18.1][4]
- for microbiome source tracking
  - [tensorflow-gpu-1.14.0][5]
  - [python-3.7][6]

## Usage


- Check data 

```shell
python scripts/preprocess.py check --input_dir data/tsvs
```

- Construct tree de novoly

```shell
python scripts/preprocess.py build --input_dir data/tsvs --tree data/trees
```

- Convert 'tsv' files to model-acceptable 'npz' file

```shell
python scripts/preprocess.py convert --input_dir data/tsvs --tree data/trees --output_dir data/npzs
```

- Count the number of samples in each biome

```shell
python scripts/preprocess.py count --input_dir data/tsvs --output_dir tmp/
```

- Merge multiple 'npz' files to a single 'npz'

```shell
python scripts/preprocess.py merge --input_dir data/npzs --output_dir data/npzs
```

- Do feature selection

```shell
python scripts/preprocess.py select --input_dir data/npzs --output_dir data/npzs
```

- **source tracking**

```shell
scripts/ONN4MDM.py [options] -g/--gpus <int> -t/--tree <tree.file> -n/--name <name.file> -m/--model <model.file> -i/--ifn <input.file> -o/--ofn <output.file> -th/--threshold <float>
```
## Todo

1. Optimizing the speed when querying taxa database. https://stackoverflow.com/a/53253110

## Author
   Name   |      Email      |      Organization
:--------:|-----------------|--------------------------------------------------------------------------------------------------------------------------------
Hugo Zha |hugozha@hust.deu.cn|Ph.D. Candidate, School of Life Science and Technology, Huazhong University of Science & Technology
Hui Chong|chonghui@hust.edu.cn ch37915405887@gmail.com|Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology
Kang Ning|ningkang@hust.edu.cn|Professor, School of Life Science and Technology, Huazhong University of Science & Technology

[1]:https://github.com/HUST-NingKang-Lab/ONN4MDM
[2]:https://github.com/caesar0301/treelib
[3]:https://pandas.pydata.org
[4]:www.numpy.org
[5]:https://pypi.org/project/tensorflow-gpu/1.14.0/
[6]:https://www.python.org/downloads/release/python-374/
