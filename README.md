# ONN4MDM
Ontology-aware Neural Network for Mircobiome Data Mining!

This program is designed to perform fast and accurate biome source tracking. The ontology of biome is organized to a biome tree, which have six layers. From the root nodes to the leaf nodes, each node has only one parent node. The Neural Network is also organized in six layers, which could produce a hierarchical classification result. The input is a species realtive abundance ".tsv" file, which can be produced by Qiime or get from EBI. The output is a biome source ".txt" file, whcih shows you where the input sample comes from.

## Install
Download the zip archive from this [github website][1], then unzip the archive.
## Function
The program can be used for biome source tracking.
### tsv convert to npz
The input file format of ONN is the ".npz" file. Before ONN, you need to convert the original input ".tsv" file into ".npz" file. The script "/DP4ONN/main.py" could work for it.
### source tracking
If you have successfully converted the ".tsv" file into ".npz" file, then you could run the script "ONN4MDM.py" for biome source tracking. Besides, you need also indicate a trained model. We have provided a well trained model as the default model.
### Dependencies
* tensorflow-gpu-1.14.0
+ python-3.7
- numpy-1.16.4
### Usage
tsv convert to npz: preprocess.py  
source tracking: ONN4MDM.py \[options\] \-g/\-\-gpus &lt;int&gt; \-t/\-\-tree &lt;tree.file&gt; \-n/\-\-name &lt;name.file&gt; \-m/\-\-model &lt;model.file&gt; \-i/\-\-ifn &lt;input.file&gt; \-o/\-\-ofn &lt;output.file&gt; \-th/\-\-threshold &lt;float&gt;
## Author
Name|Email|Organization
:------:|---|---
Hugo Zha|hugozha@hust.deu.cn|Ph.D. Candidate, School of Life Science and Technology, Huazhong University of Science & Technology
Kang Ning|ningkang@hust.edu.cn|Professor, School of Life Science and Technology, Huazhong University of Science & Technology
Hui Chong|chonghui@hust.edu.cn ch37915405887@gmail.com|Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology

[1]:https://github.com/HUST-NingKang-Lab/ONN4MDM
