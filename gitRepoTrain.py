# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:56:00 2015

@author: Timber
"""
from supervisedRWfunc import *

# load the sanpshots of 6-30 and 12-31,
# 6-30 is a graph used as a basis of the learning process
# 12-31 provides both training data and test data
fp = open('repos/1000_repos/snapshot-0630.txt', 'r')
fp_end = open('repos/1000_repos/snapshot-1231.txt', 'r')

nnodes = 1000
edges = []
edges_end = []
features = [[], []]
features_end = [[], []]
for line in fp:
    temp = line.strip().split(',')
    edges.append((int(temp[0]), int(temp[1])))
    features[0].append(float(temp[2]))
    features[1].append(float(temp[3]))

for line in fp_end:
    temp = line.strip().split(',')
    edges_end.append((int(temp[0]), int(temp[1])))
    features_end[0].append(float(temp[2]))
    features_end[1].append(float(temp[3]))

fp.close()
fp_end.close()

# normalize features
u0 = np.mean(features[0])
u1 = np.mean(features[1])
s0 = np.std(features[0])
s1 = np.std(features[1])

features[0] = map(lambda x: (x-u0)/s0, features[0])
features[1] = map(lambda x: (x-u1)/s1, features[1])

edge_feature = []
# features are formed with intercept term
for i in range(len(edges)):
    edge_feature.append([1, features[0][i], features[1][i]])



# assume source node 0, compute the candidate set for future links
source = 0
sNeighbor = []
for e in edges:
    if e[0] == source:
        sNeighbor.append(e[1])
candidates = list(set(list(range(nnodes))) - set([source]) - set(sNeighbor))

sNeighbor_end = []
for e in edges_end:
    if e[0] == source:
        sNeighbor_end.append(e[1])
Dset = list(set(sNeighbor_end) - set(sNeighbor))
Lset = list(set(candidates) - set(Dset))


#######################################
#### Model training phase #############
#######################################

print "Training model..."

# set up parameters
lam = 0
offset = 0
alpha = 0.2
beta_init = [0, 0, 0]
beta_Opt = trainModel(Dset, Lset, offset, lam, nnodes, edges, edge_feature, 
                      source, alpha, beta_init)

print beta_Opt








