# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 02:21:55 2015

@author: Timber
"""
import json
import numpy as np
import pylab as pyl
from matplotlib import pyplot as plt

fp = open('Crawl data/Snapshot_v1/snapshot-0630.json', 'r')
gitG_raw = []
for line in fp:
    temp = json.loads(line)
    gitG_raw.append(temp)

# convert contribution and interaction array to type numpy array
nnodes = len(gitG_raw)
for i in range(nnodes):
    gitG_raw[i]['contrib'] = np.asarray(gitG_raw[i]['contrib'])
    gitG_raw[i]['interact'] = np.asarray(gitG_raw[i]['interact'])

# read edges and degrees from file
fe = open('Crawl data/Snapshot_v2/interact_2/snapshot-0630.txt', 'r')
#fd = open('snapshot_degrees-0630.json', 'r')

edges = []
degrees = [0]*nnodes
for line in fe:
    temp = line.strip().split(',')
    edges.append((int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])))
    degrees[int(temp[0])] += 1
    degrees[int(temp[1])] += 1

#edges = json.load(fe)
#degrees = json.load(fd)



fp.close()
fe.close()
#fd.close()


"""
# construct edge list and degree list
edges = []
degrees = [0] * nnodes
for i in range(nnodes - 1):
    for j in range(i+1, nnodes):
        if sum(gitG_raw[i]['contrib'] * gitG_raw[j]['contrib']) > 0:
            edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1
""" 
"""
# plot degree histogram
pyl.xlabel('Degree')
pyl.ylabel('Number of nodes')
pyl.title('Degree histogram of GitHub graph')
pyl.hist(degrees, bins=100)
pyl.show()
"""

# create degree distribution
degDist = {}
for i in range(len(degrees)):
    if degrees[i] in degDist.keys():
        degDist[degrees[i]] += 1
    else:
        degDist[degrees[i]] = 1
degItem = sorted(degDist.items(), key = lambda x: x[0])
degKey = []
degAccu = []
for i in range(len(degItem)):
    degKey.append(degItem[i][0])
    degAccu.append(degItem[i][1])

sumDeg = sum(degAccu)
for i in range(1, len(degAccu)):
    degAccu[i] += degAccu[i-1]
#degAccu = map(lambda x: 1 - float(x)/sumDeg, degAccu)
degAccu = map(lambda x: 1 - float(x)/sumDeg, degAccu)


# plot the degree distribution
plt.scatter(degKey, degAccu)
plt.xlabel('Degrees')
plt.ylabel('Cumulative Probability')
plt.axis((-10, 800, -0.01, 0.3))
plt.show()

#fig = plt.figure()
plt.loglog(degKey, degAccu)
plt.xlabel('Degrees')
plt.ylabel('Probability')
ax = plt.gca()
#ax.scatter(degKey[100:], degAccu[100:])
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
#ax.axis((100, 1000, 0.05, 0))












