# ONN4MDM
Ontology-aware Neural Network for Mircobiome Data Mining! This program is designed to perform fast and accurate biome source tracking. The ontology of biome is organized to a biome tree, which have six layers. From the root nodes to the leaf nodes, each node has only one parent node. The Neural Network is also organized in six layers, which could produce a hierarchical classification result. The input is a species realtive abundance ".tsv" file, which can be produced by Qiime or get from EBI. The output is a biome source ".txt" file, whcih shows you where the input sample comes from.

## FUNCTION
The program can be used for biome source tracking.
### tsv convert to npz
The input file format of ONN is the ".npz" file. Before ONN, you need to convert the original input ".tsv" file into ".npz" file. The script "/DP4ONN/main.py" could work for it.

### source tracking
If you have successfully converted the ".tsv" file into ".npz" file, then you could run the script "ONN4MDM.py" for biome source tracking. Besides, you need also indicate a trained model. We have provided a well trained model as the default model.


## Author
Name|Email|Organization
---|---|---
Hugo Zha|hugozha@hust.deu.cn|Ph.D. Candidate, School of Life Science and Technology, Huazhong University of Science & Technology
Kang Ning|ningkang@hust.edu.cn|Professor, School of Life Science and Technology, Huazhong University of Science & Technology
Hui Chong|chonghui@hust.edu.cn ch37915405887@gmail.com|Research Assistant, School of Life Science and Technology, Huazhong University of Science & Technology
