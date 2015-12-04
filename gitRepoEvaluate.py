# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 18:51:13 2015

@author: Timber
"""

import json
from matplotlib import pyplot as plt
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

print "Reading trained data..."

# read-in trained json file
fjson = open('git_repo_100test_7.json', 'r')
for line in fjson:
    trained_item = json.loads(line)
fjson.close()


# reconstruct future link and non-link sets from the input test set
testSet = trained_item['test set']
Dset_test = []
Lset_test = []
candidates_test = []
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
    tempLset = list(set(candidates) - set(tempDset))
    Dset_test.append(tempDset)
    Lset_test.append(tempLset)
    candidates_test.append(tempDset + tempLset)


# set up parameters
lam = 50
offset = 0.01
alpha = 0.3


#######################################
#### Test model performance ###########
#######################################

# beta value obtained by training with 200 randomly chosen nodes
beta_Opt = trained_item['beta']

print "Evaluating model performance..."

# link prediction with transition matrices computed with trained parameters
ff = genFeatures(nnodes, edges, edge_feature)
trans_srw = genTrans(nnodes, edges, ff, testSet, alpha, beta_Opt[0])

# test link prediction performance up to the first 20 link recommendation
# on repo graph
linkSize = 20
hit_rates_srw = []
hit_rates_uw = []

# compute personalized PageRank for test nodes to recommend links
pgrank_srw = []
cand_pairs_srw = []
link_hits_srw = [0] * linkSize
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
        
    # calculate precision of the top-20 predicted links   
    for j in range(size):
        if cand_pairs[j][0] in Dset_test[i]:
            link_hits_srw[j] += 1
        
for i in range(1, len(link_hits_srw)):
    link_hits_srw[i] += link_hits_srw[i-1]

link_hits_srw = map(lambda x: float(x)/len(testSet), link_hits_srw)
link_hit_rate_srw = link_hits_srw[:]
for i in range(len(link_hit_rate_srw)):
    link_hit_rate_srw[i] = link_hit_rate_srw[i]/(i+1)

print "\nSRW performance: ", link_hits_srw

# evaluate and compared the performance of unweighted random walk
print "Evaluating alternative models..."   


# generate unweighted transition matrices for testSet nodes
trans_uw = genTrans_plain(nnodes, edges, testSet, alpha)

# compute personalized PageRank for test nodes to recommend links
pgrank_uw = []
cand_pairs_uw = []
link_hits_uw = [0] * linkSize
for i in range(len(testSet)):
    pp = np.repeat(1.0/nnodes, nnodes)
    curpgrank = iterPageRank(pp, trans_uw[i])
    # record the pgrank score
    pgrank_uw.append(curpgrank)
    
    # find the top ranking nodes in candidates set
    cand_pairs = []
    for j in candidates_test[i]:
        cand_pairs.append((j, curpgrank[j]))
    cand_pairs = sorted(cand_pairs, key = lambda x: x[1], reverse=True)
    # record candidate-pagerank pairs
    cand_pairs_uw.append(cand_pairs)
    
    # calculate precision of the top-20 predicted links
    link_hits = 0    
    for j in range(linkSize):
        if cand_pairs[j][0] in Dset_test[i]:
            link_hits_uw[j] += 1

for i in range(1, len(link_hits_uw)):
    link_hits_uw[i] += link_hits_uw[i-1]

link_hits_uw = map(lambda x: float(x)/len(testSet), link_hits_uw)
link_hit_rate_uw = link_hits_uw[:]
for i in range(len(link_hit_rate_uw)):
    link_hit_rate_uw[i] = link_hit_rate_uw[i]/(i+1)

print "\nUW performance: ", link_hits_uw


# plot the model performance
plt.plot(range(1, 21), link_hits_srw, '-o', c='blue', label = 'Supervised Random Walk')
plt.plot(range(1, 21), link_hits_uw, '-x', c='red', label = 'Unweighted Random Walk')
plt.legend(loc = 4)
plt.xlabel("Number of predicted links")
plt.ylabel("Average number of hits")
plt.figure(figsize = (400, 200))
plt.show()

# plot the model performance
plt.plot(range(1, 21), link_hit_rate_srw, '-o', c='blue', label = 'Supervised Random Walk')
plt.plot(range(1, 21), link_hit_rate_uw, '-x', c='red', label = 'Unweighted Random Walk')
plt.legend(loc = 1)
plt.xlabel("Number of predicted links")
plt.ylabel("Hit Rate")
plt.figure(figsize = (400, 200))
plt.show()


