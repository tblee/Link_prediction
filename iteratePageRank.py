# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:05:07 2015

@author: Timber
"""
import numpy as np

# ***function: iterPageRank
# this function takes initial node probabilities and a 
# transition matrix then use power iteration to find 
# the PageRank of nodes
def iterPageRank(pp, trans):
    ppnew = np.dot(pp, trans)
    while not(np.allclose(pp, ppnew)):
        pp = ppnew
        ppnew = np.dot(pp, trans)
    return ppnew


nnodes = 5000
pp = np.repeat(1.0/nnodes, nnodes)

# randomly generate ajacency matrix and the transition matrix
edgep = 0.2
trans = np.zeros((nnodes, nnodes))
for i in range(nnodes-1):
    for j in range(i+1, nnodes):
        if np.random.rand() < 0.8:
            trans[i, j] = np.random.rand()
            trans[j, i] = np.random.rand()

# normalize the trans matrix
for i in range(nnodes):
    tempSum = sum(trans[i,])
    trans[i,] = map(lambda x: x/tempSum, trans[i, ])

# obtain steady state
pp = iterPageRank(pp, trans)









