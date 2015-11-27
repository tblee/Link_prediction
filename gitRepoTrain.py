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
    edge_feature.append([features[0][i], features[1][i]])



# assume source node 0, compute the candidate set for future links
source = 4
sNeighbor = []
for e in edges:
    if e[0] == source:
        sNeighbor.append(e[1])
    elif e[1] == source:
        sNeighbor.append(e[0])
candidates = list(set(list(range(nnodes))) - set([source]) - set(sNeighbor))

sNeighbor_end = []
for e in edges_end:
    if e[0] == source:
        sNeighbor_end.append(e[1])
    elif e[1] == source:
        sNeighbor_end.append(e[0])
Dset = list(set(sNeighbor_end) - set(sNeighbor))
Lset = list(set(candidates) - set(Dset))


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

print beta_Opt

"""
# first compute the (unnormalized) edge strength matrix and the gradient matrix
sMat = np.zeros((nnodes, nnodes))
for i in range(int(np.shape(trans_p)[0])):
    for j in range(i, int(np.shape(trans_p)[1])):
        if trans_p[i, j] > 0:
            strength = calStrength(ff[i][j], beta)
            sMat[i, j] = strength
            sMat[j, i] = strength


beta = [0, 0.5, 0.5]
gradS = [[ [] for x in range(nnodes) ] for x in range(nnodes) ]
for i in range(int(np.shape(trans_p)[0])):
	for j in range(int(np.shape(trans_p)[1])):
		if trans_p[i, j] > 0:
			gradS[i][j] = strengthDiff(ff[i][j], beta)

qp = []
for i in range(len(beta)):
    qp.append(np.zeros((nnodes, nnodes)))

for i in range(int(np.shape(trans_p)[0])):
    # for each row in the gradient matrix, some common factors can be 
    # computed first
    sumStrength = 0
    sumDiff = [0] * len(beta)
    for j in range(int(np.shape(trans_p)[1])):
        if trans_p[i, j] > 0:
            sumStrength += sMat[i, j]
            for k in range(len(beta)):
                sumDiff[k] += gradS[i][j][k]
    print "in row", i, sumStrength, sumDiff
    
    # individual entries can then be computed
    for j in range(int(np.shape(trans_p)[1])):
        if trans_p[i, j] > 0:
            for k in range(len(beta)):
                qp[k][i, j] = (sumStrength ** -2)*( gradS[i][j][k]*sumStrength -
                sMat[i, j]*sumDiff[k])*(1 - alpha)

"""


