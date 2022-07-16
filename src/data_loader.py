import os
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import random
import copy


sampledhedge = [] # sampled ndoes/hyperedges
hedge2node = [] # each row is the set of vertices in that hyperedge
hyperedgelabel = {} # type/label of that hyperedge
hyperedges = []
hypernodes = defaultdict(list)
clique = defaultdict(set)
hedges_n = 0
label_n = 0

def get_nodeEmb(filename):

    f = open(filename,"r")
    nodeEmb = []
    for line in f:
        w = line.split()
        tmp = []
        for a in w:
            val = float(a)
            tmp.append(val)
        nodeEmb.append(tmp)
    return nodeEmb

#def get_EdgeEmb(filename,node_emb,size):
def get_EdgeEmb(filename,size):

    f = open(filename,"r")
    i = 0
    nodeEmb = []
    for line in f:
        w = line.split()
        nn = int(w[0])
        p = int(w[1])
        tmp = [0. for i in range(size)]
        tmp[p] = 1
        #node_emb[i].extend(tmp)
        nodeEmb.append(tmp)
        i+=1
    return nodeEmb

def read_labels(filename):
    f = open(filename,'r')
    i = 0
    labels = []
    r_t = {}
    for line in f:
        w = line.split()
        labels.append(int(w[0]))
        r_t[int(w[0])] = 1
    return labels,len(r_t)




def tensorize(data):
    finalhedge = []
    newhyper = []
    for node in data:
        newn = node[:len(node)-1]
        sampled=np.random.choice(list(newn),size=args.hedge_size,\
                replace=len(newn) < args.hedge_size)
        sampled =np.append(sampled,node[len(node)-1])
        finalhedge.append(sampled)


    return finalhedge


def read_hypergraph(filename):
   node2hedge_list = defaultdict(list) # key = node, val = hyperedges in that node
   global hedges_n, label_n
   hedge2list = [] # each row is set of nodes in a hyperedge
   node2list = [] # each row is set of hedge connected to a node
   file = open(filename)
   i = 0
   maxn = 0
   minn = 10000
   for line in file:
       ids = line.strip().split()
       hyperedge = []
       if len(ids) < 1:
           continue
       for j in range(len(ids)-1):
           id = ids[j]
           hyperedge.append(int(id)) 
           node2hedge_list[int(id)].append(i)
           hypernodes[int(id)].append(hedges_n)
           if int(id) > maxn:
               maxn = int(id)
       t = len(ids) - 1
       hyperedge.append(int(ids[t]))
       hyperedgelabel[int(ids[t])]=1
       hedge2list.append(hyperedge)
       hyperedges.append(hyperedge)
       i += 1
       hedges_n += 1
   j = 0
   for n in range(0,maxn+1):
       if n not in node2hedge_list:
           h = [len(hedge2list)]
           node2list.append(h)
           tmp = []
           tmp.append(n)
           tmp.append(1)
           hedge2list.append(tmp)
       else:
           v = node2hedge_list[n]
           node2list.append(v)
  
   return node2list, hedge2list



def build_hypergraph(train_data_nodes, hedge_data):
    neigh_nodes = []
    nodes =  train_data_nodes
    hedges = hedge_data
    i = 0
    for a in nodes:
        #if len(a) < 1:
        #    continue
        sampled_neighbors=np.random.choice(list(a),size=args.neighbor_samples,\
                replace=len(a) < args.neighbor_samples)
        sampledhedge.append(sampled_neighbors)
        i += 1
    hypere = []
    types = []
    for h in range(len(hedge_data)):
        n = hedge_data[h]
        tmp = []
        for i in range(len(n)-1):
            tmp.append(n[i])
        hypere.append(tmp)
        types.append(n[len(n)-1])
    samplednodes = []
    for a in hypere:
        sampled=np.random.choice(list(a),size=args.hedge_size,\
            replace=len(a) < args.hedge_size)
        samplednodes.append(sampled)
    return samplednodes,types






def load_data(model_args):
    global args
    args = model_args

    directory = '../data/' + args.dataset + '/'
    print('reading hypergraph')
    
    train_data_n, train_data_h = read_hypergraph(directory + 'trainp.txt')
    valid_data_n, valid_data_h = read_hypergraph(directory + 'valid.txt')
    test_data_n, test_data_h = read_hypergraph(directory + 'test.txt')
    #real_labels,r_size = read_labels(directory + 'trainLabel.txt')
    hedge,types = build_hypergraph(train_data_n,train_data_h)
    #node_emb = get_nodeEmb(directory +'bipart.txt')
    #edge_emb = get_EdgeEmb(directory +'nodeparts.txt',node_emb,len(hyperedgelabel)+1)
    edge_emb = get_EdgeEmb(directory +'nodeparts.txt',len(hyperedgelabel)+1)
    print(edge_emb[0])


    print('hypergraph is built')

    triplets = [tensorize(train_data_h), tensorize(valid_data_h), tensorize(test_data_h)]


    neighbor_params = [np.array(sampledhedge), np.array(hedge), np.array(types,dtype=np.int32),np.array(edge_emb)]
    if len(hypernodes) != len(train_data_n):
        print("train node does not include all nodes")
       # return 0

    return triplets, len(hyperedgelabel)+1, neighbor_params



