# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:41:54 2015

@author: Timber
"""
import functools
import numpy as np
from scipy.optimize import fmin_bfgs

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


# ***function: genCopyGraph
# generate an undirected random graph with copyig model
# the generated graph has the property of preferencial attachment
# an edge list is returned
def genCopyGraph(nnodes, alpha):
    # since the copy model starts with a triad, the input number of 
    # nodes should be larger than 3
    if nnodes <= 3:
        print "Number of nodes should be larger than 3..."
        return
    # inital setting with a triad
    degrees = np.repeat(2, 3)
    edges = [(0, 1), (0, 2), (1, 2)]
    # growing the graph node by node
    for i in range(3, nnodes):
        # add three edges for the new node
        for nedge in range(3):
            if np.random.rand() < alpha:
                # uniformly choose node to connect
                tar = np.random.choice(i, 1)[0]
                while (tar, i) in edges:
                    tar = np.random.choice(i, 1)[0]
                edges.append((tar, i))
                degrees[tar] += 1
            else:
                # select target to connect according to degree
                sDeg = sum(degrees)
                tar = -1
                firstRound = True
                while (tar, i) in edges or firstRound:
                    firstRound = False
                    randPick = np.random.randint(1, sDeg+1)
                    accu = 0
                    for j in range(len(degrees)):
                        accu += degrees[j]
                        if randPick <= accu:
                            tar = j
                            break
                        
                edges.append((tar, i))
                degrees[tar] += 1
        
        degrees = np.append(degrees, 3)
    return [edges, degrees]


# ***function: calStrength
# return edge strength calculated by exponential function
# the inputs are two vectors, features and the parameters
def calStrength(features, beta):
    return np.exp(np.dot(features, beta))


# ***function: genTrans
# this function takes in a graph (edge list and number of nodes), 
# node features, source node, and alpha/beta parameters to generate
# a random walk transition matrix.
# transition probabilities are determined by edge strength
# beta is the parameter in the edge strength function
# alpha is the teleportation rate back to the source node
def genTrans(nnodes, g, features, s, alpha, beta):
    # feature is supplied as a n*p-matrix
    # edge features are then obtained from the interaction of
    # node features
    # the transition matrix is created with teleportation
    trans = np.zeros((nnodes, nnodes))
    for i in range(len(g)):
        strength = calStrength(np.asarray(features[g[i][0],])*np.asarray(features[g[i][1],])
        , beta)
        trans[g[i][0], g[i][1]] = strength
        trans[g[i][1], g[i][0]] = strength
    
    # normalize the transition matrix
    for i in range(nnodes):
        tempSum = sum(trans[i,])
        trans[i,] = map(lambda x: x/tempSum, trans[i, ])
    
    # create the one matrix
    one = np.zeros((nnodes, nnodes))
    for i in range(nnodes):
        one[i, s] = 1
        
    # combine the regular transition matrix and the one matrix
    trans = (1-alpha)*trans + alpha*one
    
    return trans

# ***function: genTrans_plain
# this function construct transition matrix for random walk
# with unweighted edge strenght, i.e. each eade has strength 1
def genTrans_plain(nnodes, g, s, alpha):
    trans = np.zeros((nnodes, nnodes))
    for i in range(len(g)):
        trans[g[i][0], g[i][1]] = 1
        trans[g[i][1], g[i][0]] = 1
    
    # normalize the transition matrix
    for i in range(nnodes):
        tempSum = sum(trans[i,])
        trans[i,] = map(lambda x: x/tempSum, trans[i, ])
    
    # create the one matrix
    one = np.zeros((nnodes, nnodes))
    for i in range(nnodes):
        one[i, s] = 1
        
    # combine the regular transition matrix and the one matrix
    trans = (1-alpha)*trans + alpha*one
    
    return trans


############################################
############################################
## Below are the functions for learning process

def iterPageDiff(pdiff, p, trans, transdiff):
    pdiffnew = np.dot(pdiff, trans) + np.dot(p, transdiff)
    while not(np.allclose(pdiff, pdiffnew)):
        pdiff = pdiffnew
        pdiffnew = np.dot(pdiff, trans) + np.dot(p, transdiff)
    return pdiffnew[0]


def diffQelem(features, beta, trans, alpha, row, col, k):
    # calculates the element value of transition matrix's differentiation
    # first calculate the denominator part    
    denom = 0
    xdenom = 0
    for j in range(int(np.shape(trans)[1])):
        if trans[row, j] > 0:
            temp = calStrength(np.asarray(features[row,])*np.asarray(features[j,])
            , beta)
            denom += temp
            xdenom += (np.asarray(features[row,])*np.asarray(features[j,]))[k] * temp
    curFeat = np.asarray(features[row,])*np.asarray(features[col,])
    strength = calStrength(curFeat, beta)
    
    elem = (1-alpha)*(curFeat[k]*strength*denom - strength*xdenom) / (denom**2)
    
    return elem


def diffQ(features, beta, trans, alpha, k):
    qp = np.zeros(np.shape(trans))
    for i in range(int(np.shape(trans)[0])):
        for j in range(int(np.shape(trans)[1])):
            if trans[i, j] > 0:
                qp[i, j] = diffQelem(features, beta, trans, alpha, i, j, k)
    
    return qp


def costFunc(pl, pd, offset):
    return (max(0, pl - pd + offset))**2
    
def costDiff(pl, pd, offset):
    return 2.0*(max(0, pl - pd + offset))


def minObj(Dset, Lset, offset, lam, nnodes, g, features, source, alpha, beta):
    # calculate PageRank according to features and beta values
    trans = genTrans(nnodes, g, features, source, alpha, beta)
    pp = np.repeat(1.0/nnodes, nnodes)
    pgrank = iterPageRank(pp, trans)
    
    # compute cost from the generated PageRank value
    cost = 0
    for d in Dset:
        for l in Lset:
            cost += costFunc(pgrank[l], pgrank[d], offset)
    penalty = lam * np.dot(beta, beta)
    
    return (cost + penalty)


def objDiff(Dset, Lset, offset, lam, nnodes, g, features, source, alpha, beta):
    diffVec = []
    # calculate PageRank according to features and beta values
    trans = genTrans(nnodes, g, features, source, alpha, beta)
    pp = np.repeat(1.0/nnodes, nnodes)
    pgrank = iterPageRank(pp, trans)
    for k in range(len(beta)):
        tempObjDiff = 0
        pDiff = np.zeros((1, nnodes))
        transDiff = diffQ(features, beta, trans, alpha, k)
        pDiff = iterPageDiff(pDiff, pgrank, trans, transDiff)
        for d in Dset:
            for l in Lset:
                tempObjDiff += costDiff(pgrank[l], pgrank[d], offset)*(pDiff[l] - pDiff[d])
        # penalty term
        tempObjDiff += 2.0 * lam * beta[k]
        
        diffVec.append(tempObjDiff)
    return np.asarray(diffVec)
        

def trainModel(Dset, Lset, offset, lam, nnodes, g, features, source, alpha, beta_init):
    beta_Opt = fmin_bfgs(functools.partial(minObj, Dset, Lset, 0, 0, nnodes, g, features, 
                            source, alpha), beta_init, fprime = functools.partial(objDiff, 
                            Dset, Lset, 0, 0, nnodes, g, features, source, alpha))
    return beta_Opt


############################################
############################################


