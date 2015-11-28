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

# pick nodes with number of future links larger than theshold
# these nodes then served as either training node or test node
Dsize_cut = 20

D_source = []
Dset_all = []
Lset_all = []
for i in range(len(elig_source)):
    sNeighbor = []
    for e in edges:
        if e[0] == elig_source[i]:
            sNeighbor.append(e[1])
        elif e[1] == elig_source[i]:
            sNeighbor.append(e[0])
    candidates = list(set(list(range(nnodes))) - set([elig_source[i]]) - set(sNeighbor))
    
    sNeighbor_end = []
    for e in edges_end:
        if e[0] == elig_source[i]:
            sNeighbor_end.append(e[1])
        elif e[1] == elig_source[i]:
            sNeighbor_end.append(e[0])
    tempDset = list(set(sNeighbor_end) - set(sNeighbor))
    if len(tempDset) >= Dsize_cut:
        tempLset = list(set(candidates) - set(tempDset))
        Dset_all.append(tempDset)
        Lset_all.append(tempLset)
        D_source.append(elig_source[i])

# randomly pick nodes with current degree > 0 and number of future 
# links >= Dsize_cut as the training set
trainSize = 10
testSize = 100
# this index is the index of source nodes in D_source list
source_index = np.random.choice(list(range(len(D_source))), 
                                size=trainSize, replace=False)
source = []
Dset = []
Lset = []
for i in source_index:
    source.append(D_source[i])
    Dset.append(Dset_all[i])
    Lset.append(Lset_all[i])

# randomly pick nodes with current degree > 0, number of future links 
# >= Dsize_cut and haven't been picked as training nodes to be test nodes
test_index = np.random.choice(list(set(list(range(len(D_source)))) - 
set(source_index)), size=testSize, replace=False)

testSet = []
Dset_test = []
candidates_test = []
for i in test_index:
    testSet.append(D_source[i])
    Dset_test.append(Dset_all[i])
    candidates_test.append(Dset_all[i] + Lset_all[i])

"""
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
"""

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


#######################################
#### Test model performance ###########
#######################################
"""
# randomly pick test nodes with degree > 0 and haven't been chosen
# as training nodes
testSize = 100
testSet = np.random.choice(list(set(elig_source) - set(source)), 
                           size=testSize, replace=False)

Dset_test = []
for i in range(len(testSet)):
    sNeighbor = []
    for e in edges:
        if e[0] == testSet[i]:
            sNeighbor.append(e[1])
        elif e[1] == testSet[i]:
            sNeighbor.append(e[0])
    candidates = list(set(list(range(nnodes))) - set([testSet[i]]) - set(sNeighbor))
    
    sNeighbor_end = []
    for e in edges_end:
        if e[0] == testSet[i]:
            sNeighbor_end.append(e[1])
        elif e[1] == testSet[i]:
            sNeighbor_end.append(e[0])
    tempDset = list(set(sNeighbor_end) - set(sNeighbor))
    Dset_test.append(tempDset)
"""

# link prediction with transition matrices computed with trained parameters
ff = genFeatures(nnodes, edges, edge_feature)
trans_srw = genTrans(nnodes, edges, ff, testSet, alpha, beta_Opt[0])

# compute personalized PageRank for test nodes to recommend links
pgrank_srw = []
cand_pairs_srw = []
link_hits_srw = []
for i in range(len(testSet)):
    pp = np.repeat(1.0/nnodes, nnodes)
    curpgrank = iterPageRank(pp, trans_srw[i])
    # record the pgrank score
    pgrank_srw.append(curpgrank)
    
    # find the top ranking nodes in candidates set
    cand_pairs = []
    for j in candidates_test[i]:
        cand_pairs.append((j, curpgrank[j]))
    cand_pairs = sorted(cand_pairs, key = lambda x: x[1], reverse=True)
    # record candidate-pagerank pairs
    cand_pairs_srw.append(cand_pairs)
    
    # calculate precision of the top-Dsize_cut predicted links
    link_hits = 0    
    for j in range(Dsize_cut):
        if cand_pairs[j][0] in Dset_test[i]:
            link_hits += 1
    link_hits_srw.append(link_hits)
    








