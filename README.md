# ONN4MST
<img src="https://github.com/HUST-NingKang-Lab/ONN4MDM/blob/master/image/release.png" width="134" height="20">
Ontology-aware Neural Network for Mircobiome Data Mining!

<img src="https://github.com/HUST-NingKang-Lab/ONN4MDM/blob/master/image/Figure2.png">
This program is designed to perform fast and accurate biome source tracking. The ontology of biome is organized to a biome tree, which have six layers. From the root nodes to the leaf nodes, each node has only one parent node. The Neural Network is also organized in six layers, which could produce a hierarchical classification result. The input is a species realtive abundance ".tsv" file, which can be produced by Qiime or get from EBI. The output is a biome source ".txt" file, whcih shows you where the input sample comes from.

The preprocessor can make the data preprocessing and sample statistics of the Ontology-aware Neural Network easier. Due to the use of three predefined classes, the simplicity and efficiency of the code are greatly improved. For very time-consuming big data calculations, any minor data or program errors can cost you days or even weeks. Using this program to check the integrity of all data and error values before processing can greatly reduce the probability of your program running errors.

## Support
For support using ONN4MST, please email us. Any comments/insights would be greatly appreciated.

## Installation
Download the zip archive from this [repository][1], then unzip the archive on your local computer platform.
## Function
The program could be used for microbiome samples source tracking.

### Before using
Check if the src/searching.py is executable. If this file is not executable, type
```shell
chmod +x src/searching.py
```
### Abundance table convert to the Matrix
The input file format of ONN4MST is the ".npz" file. Before ONN, you need to convert the original input ".tsv" file into ".npz" file. The script "src/preprocess.py" could work for it.
### Microbiome samples source tracking
If you have successfully converted the ".tsv" file into ".npz" file, then you could run the script "src/searching.py" for biome source tracking. Besides, you need also indicate a trained model. We have provided a well trained model as the default model.
## Dependencies

- For data preprocessing
  - [python-3.7][6]
  - [Treelib-1.5.5][2]
  - [Pandas-1.0.1][3]
  - [Numpy-1.16][4]
- For microbiome samples source tracking
  - [tensorflow-gpu-1.14][5]

## Usage


- Check data 

```shell
python src/preprocess.py check --input_dir data/tsvs
```

- Construct tree de novoly

```shell
python src/preprocess.py build --input_dir data/tsvs --tree data/trees
```

- Convert 'tsv' files to model-acceptable 'npz' file

```shell
python src/preprocess.py convert --input_dir data/tsvs --tree data/trees --output_dir data/npzs
```

- Count the number of samples in each biome

```shell
python src/preprocess.py count --input_dir data/tsvs --output_dir tmp/
```

- Merge multiple 'npz' files to a single 'npz'

```shell
python src/preprocess.py merge --input_dir data/npzs --output_dir data/npzs
```

- Do feature selection

```shell
python src/preprocess.py select --input_dir data/npzs --output_dir data/npzs
```

- **Microbiome samples source tracking**

```shell
src/searching.py [-h] [-g {0,1}] [-gid GPU_CORE_ID] [-s {0,1}] [-t TREE] [-m MODEL] [-th THRESHOLD] [-of {1,2,3}] ifn ofn
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
