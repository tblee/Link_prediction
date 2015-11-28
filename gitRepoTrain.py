# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:56:00 2015

@author: Timber
"""
from supervisedRWfunc import *

print "Reading data..."

# load the sanpshots of 6-30 and 12-31,
# 6-30 is a graph used as a basis of the learning process
# 12-31 provides both training data and test data
fp = open('repos/1000_repos/snapshot-0630.txt', 'r')
fp_end = open('repos/1000_repos/snapshot-1231.txt', 'r')

nnodes = 1000
degrees = [0] * nnodes
edges = []
edges_end = []
features = [[], []]
features_end = [[], []]
for line in fp:
    temp = line.strip().split(',')
    edges.append((int(temp[0]), int(temp[1])))
    features[0].append(float(temp[2]))
    features[1].append(float(temp[3]))
    degrees[int(temp[0])] += 1
    degrees[int(temp[1])] += 1

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
    edge_feature.append([features[0][i], features[1][i]])


#######################################
#### Training set formation ###########
#######################################

print "Forming training set..."

# compute the candidate set for future links according to source node
# train model with a set of source nodes
elig_source = []
for i in range(len(degrees)):
    if degrees[i] > 0:
        elig_source.append(i)
# randomly pick 100 nodes with degree > 0 as the training set
source = np.random.choice(elig_source, size=10, replace=False)
Dset = []
Lset = []
for i in range(len(source)):
    sNeighbor = []
    for e in edges:
        if e[0] == source[i]:
            sNeighbor.append(e[1])
        elif e[1] == source[i]:
            sNeighbor.append(e[0])
    candidates = list(set(list(range(nnodes))) - set([source[i]]) - set(sNeighbor))
    
    sNeighbor_end = []
    for e in edges_end:
        if e[0] == source[i]:
            sNeighbor_end.append(e[1])
        elif e[1] == source[i]:
            sNeighbor_end.append(e[0])
    tempDset = list(set(sNeighbor_end) - set(sNeighbor))
    tempLset = list(set(candidates) - set(tempDset))
    Dset.append(tempDset)
    Lset.append(tempLset)


#######################################
#### Model training phase #############
#######################################

print "Training model..."

# set up parameters
lam = 0
offset = 0
alpha = 0.1
beta_init = [0, 0]

#ff = genFeatures(nnodes, edges, edge_feature)
#trans_p = genTrans_plain(nnodes, edges, 0, 0)
#qqp = diffQ(ff, [0, 0.5, 0.5], trans_p, alpha)
#print qqp
beta_Opt = trainModel(Dset, Lset, offset, lam, nnodes, edges, edge_feature, 
                      source, alpha, beta_init)

print "Training source set:\n", source
print "\nTrained model parameters:\n", beta_Opt

