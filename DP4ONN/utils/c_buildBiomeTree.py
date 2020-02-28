#coding=utf-8
import treelib
import pickle
import os
from treelib import Node, Tree
from b_buildTree import saveTree

tree=Tree() #建立树
tree.create_node("root", 'root')
def main():
#   tree=Tree()
    getFilename()
    tree.show()

#获取目录名的结构存储到文件中先进行处理
def getFilename():
    path='/home/qiuhao/ONN/ONNdata/'
    for filename in os.listdir(path):
        pathIn='/home/qiuhao/ONN/ONNdata/'+filename+'/'
        Allsamples =','.join(os.listdir( pathIn ))
        filename=filename.split('-')
    #   print(Allsamples)   

        #   for name in filename:
        tree=buildTree(filename,Allsamples)

#建立树  
def buildTree(filename,Allsamples):
    #遍历文件夹建树
#   tree = Tree()
#   for name in filename:
    all_nodes_identifier = [n.identifier for n in tree.all_nodes() ]            # how to deal with unhashable node Uncul. Bact.
        #if filename[0] not in all_nodes_identifier and filename[0] is not None:
        #   tree.create_node(filename[0], filename[0])  # root node
#       tree.create_node("Root", 'Root')  # root node
#   for i in filename:
    ID='root'
    for name in filename:
        if name==filename[0]:
            a=1
        elif name==filename[1]:
            ID='root'+'-'+filename[1]
            IDbefore='root'
        else:
            ID=ID+'-'+name
        if ID not in all_nodes_identifier:
            tree.create_node(ID, ID, IDbefore)
            all_nodes_identifier = [n.identifier for n in tree.all_nodes() ]
        IDbefore=ID
    return tree
if __name__=='__main__':
	main()
	saveTree(tree, 'tree/biome_tree.pydata', 'tree/biome_tree_visualized.txt')
