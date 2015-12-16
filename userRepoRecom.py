# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 20:35:53 2015

@author: Timber
"""
import json
from matplotlib import pyplot as plt
from supervisedRWfunc import *

#######################################
######## Read-in Repo Graph ###########
#######################################

# load the sanpshots of 6-30
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

ff = genFeatures(nnodes, edges, edge_feature)


#######################################
######## Read-in User Data ############
#######################################

# read-in user data
fp = open('teleport_sets/teleport_sets-0630.txt', 'r')
fp_end = open('teleport_sets/teleport_sets-1231.txt', 'r')

begin_tele = []
end_tele = []

for line in fp:
    begin_tele.append(json.loads(line))
for line in fp_end:
    end_tele.append(json.loads(line))

fp.close()
fp_end.close()

# read-in trained model parameters
fjson = open('git_repo_100test_7.json', 'r')
for line in fjson:
    trained_item = json.loads(line)
fjson.close()

nUsers = len(begin_tele)
nAddLinks = map(lambda x: len(end_tele[x])-len(begin_tele[x]), range(nUsers))
addedLinks = map(lambda x: list(set(end_tele[x]) - set(begin_tele[x])), range(nUsers))

# pick the users that added more than or equal to 4 links between
# 0630 and 1231 and the initial teleport set > 0
elig_user = filter(lambda x: nAddLinks[x] >= 4 and len(begin_tele[x]) > 0, 
                   range(nUsers))
"""
# pick the users that added more than or equal to 5 links between
# 0630 and 1231
elig_user = filter(lambda x: nAddLinks[x] >= 5, range(nUsers))
"""


#######################################
#### User Repo Recommendation #########
#######################################

### Evaluation using SRW based transition matrix

alpha = 0.3
beta_Opt = trained_item['beta']

chosen_tele = []
for i in range(len(elig_user)):
    user = elig_user[i]
    chosen_tele.append(begin_tele[user])
trans_srw = genTrans_tele(nnodes, edges, ff, chosen_tele, alpha, beta_Opt[0])


numRecom = 10
repo_hits_srw = [0] * numRecom
# calculate personalied pagerank with teleport set and the 
# transition matrix with trained parameter from SRW
for i in range(len(elig_user)):
    user = elig_user[i]
    
    pp = np.repeat(1.0/nnodes, nnodes)
    pgrank = iterPageRank(pp, trans_srw[i])
    
    # choose the nodes that the users had not contributed to by 0630
    candidate_repo = list(set(range(nnodes)) - set(begin_tele[user]))
    
    cand_pg_pair = map(lambda x: (x, pgrank[x]), candidate_repo)
    cand_pg_pair = sorted(cand_pg_pair, key = lambda x: x[1], reverse=True)
    
    # recommend top 3 repo, compute number of hits
    numHits = 0
    for j in xrange(numRecom):
        if cand_pg_pair[j][0] in addedLinks[user]:
            repo_hits_srw[j] += 1
            numHits += 1
    #repo_hits_srw.append(numHits)
for i in range(1, numRecom):
    repo_hits_srw[i] += repo_hits_srw[i-1]

repo_hits_srw = map(lambda x: float(x)/len(elig_user), repo_hits_srw)
    
print "repo recommendation performance (SRW):", repo_hits_srw


### Evaluation using unweighted personalized random walk

trans_uw = genTrans_tele(nnodes, edges, ff, chosen_tele, alpha, [0, 0])

repo_hits_uw = [0] * numRecom
# calculate personalied pagerank with teleport set and the 
# transition matrix with unweighted random walk
for i in range(len(elig_user)):
    user = elig_user[i]
    
    pp = np.repeat(1.0/nnodes, nnodes)
    pgrank = iterPageRank(pp, trans_uw[i])
    
    # choose the nodes that the users had not contributed to by 0630
    candidate_repo = list(set(range(nnodes)) - set(begin_tele[user]))
    
    cand_pg_pair = map(lambda x: (x, pgrank[x]), candidate_repo)
    cand_pg_pair = sorted(cand_pg_pair, key = lambda x: x[1], reverse=True)
    
    # recommend top 3 repo, compute number of hits
    numHits = 0
    for j in xrange(numRecom):
        if cand_pg_pair[j][0] in addedLinks[user]:
            repo_hits_uw[j] += 1
            numHits += 1
    #repo_hits_uw.append(numHits)
for i in range(1, numRecom):
    repo_hits_uw[i] += repo_hits_uw[i-1]

repo_hits_uw = map(lambda x: float(x)/len(elig_user), repo_hits_uw)
    
print "repo recommendation performance (UW):", repo_hits_uw


# plot the model performance
plt.plot(range(1, numRecom+1), repo_hits_srw, '-o', c='blue', label = 'Supervised Random Walk')
plt.plot(range(1, numRecom+1), repo_hits_uw, '-x', c='red', label = 'Unweighted Random Walk')
plt.legend(loc = 4)
plt.xlabel("Number of recommended repos")
plt.ylabel("Average number of hits")
plt.figure(figsize = (800, 600))



