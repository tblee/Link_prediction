# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:51:54 2015

@author: Timber
"""
from matplotlib import pyplot as plt

fp = open('repos/1000_repos/snapshot-0630.txt', 'r')

nnodes = 1000
edges = []
degrees = [0] * nnodes
for line in fp:
    temp = line.strip().split(',')
    edges.append((int(temp[0]), int(temp[1]), int(temp[2]), int(temp[3])))
    degrees[int(temp[0])] += 1
    degrees[int(temp[1])] += 1

fp.close()


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
plt.axis((-10, 300, -0.01, 0.85))
plt.show()

#fig = plt.figure()
plt.plot(degKey, degAccu)
plt.xlabel('Degrees')
plt.ylabel('Probability')
ax = plt.gca()
#ax.scatter(degKey[100:], degAccu[100:])
#ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
#ax.axis((100, 1000, 0.05, 0))







