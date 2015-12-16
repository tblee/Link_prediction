# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:21:30 2015

@author: Timber
"""
from supervisedRWfunc import *
import json

# write simulated raw data to json format
fjson = open('Synthesized logs/synthetic_log.json', 'w')
# write error rate to another file
ferror = open('Synthesized logs/synthetic_error_rate.json', 'w')


l_error = []
ex_error = []
uw_error = []
for noise in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    l_error_temp = []
    ex_error_temp = []
    uw_error_temp = []
    for rd in range(1):

        ############################################
        ############################################
        ## synthetic data formation
        
        # synthesize data by generating a random graph and feature for each node
        print "Generating synthesized data..."
        
        nnodes = 1000
        g = genCopyGraph(nnodes, 0.5)
        # assume source node 0, compute the candidate set for future links
        source = 0
        sNeighbor = []
        for e in g[0]:
            if e[0] == source:
                sNeighbor.append(e[1])
        candidates = list(set(list(range(nnodes))) - set([source]) - set(sNeighbor))
        
        # generate features as normal(0, 1) random variables
        features = np.zeros((nnodes, 2))
        for i in range(int(np.shape(features)[0])):
            features[i,0] = np.random.normal(0, 1)
            features[i,1] = np.random.normal(0, 1)
        # compute edge feature
        edge_feature = []
        for i in range(len(g[0])):
            tempfea = [0] * 2
            tempfea[0] = features[g[0][i][0], 0] * features[g[0][i][1], 0]
            tempfea[1] = features[g[0][i][0], 1] * features[g[0][i][1], 1]
            edge_feature.append(tempfea)
        ff = genFeatures(nnodes, g[0], edge_feature)
        
        # use these parameters to simulate future links
        alpha = 0.1
        beta = [1, -1] 
        
        # generate transition matrix
        trans = genTrans(nnodes, g[0], ff, [source], alpha, beta)[0]
        # calculate pageRank
        pp = np.repeat(1.0/nnodes, nnodes)
        pgrank = iterPageRank(pp, trans)
        
        # select the nodes to form future links with source from the candidates
        # in the deterministic scheme, future links are selected deterministically
        # according to PageRank socre from weighted transition matrix
        numFLinks = 100
        candPairs = []
        for i in range(len(candidates)):
            candPairs.append((candidates[i], pgrank[candidates[i]]))
        
        # sort candidates by PageRank score, then construct the D-set and L-set
        candPairs = sorted(candPairs, key = lambda x: x[1], reverse = True)
        Dset = []
        for i in range(numFLinks):
            Dset.append(candPairs[i][0])
        Lset = list(set(candidates) - set(Dset))
        
        
        
        
        ############################################
        ############################################
        ## model training and performace evaluation
        
        # adding noise to features in the training set
        # the training process will take the features with noise
        # simulating the situation of unexplained noise in the features
        if noise > 0:        
            for i in range(nnodes):
                features[i, 0] += np.random.normal(0, noise)
                features[i, 1] += np.random.normal(0, noise)
        # compute edge feature
        edge_feature = []
        for i in range(len(g[0])):
            tempfea = [0] * 2
            tempfea[0] = features[g[0][i][0], 0] * features[g[0][i][1], 0]
            tempfea[1] = features[g[0][i][0], 1] * features[g[0][i][1], 1]
            edge_feature.append(tempfea)
        ff = genFeatures(nnodes, g[0], edge_feature)
        
        
        print "Training model..."
        
        beta_init = [0, 0]
        #beta_Opt = trainModel(Dset, Lset, 0, 0, nnodes, g[0], features, source, alpha,
        #                      beta_init)
        beta_Opt = trainModel([Dset], [Lset], 1, 0, nnodes, g[0], edge_feature, 
                              [source], alpha, beta_init)
        
        
        print "Actual beta", beta
        print "Learned beta", beta_Opt
        
        # compute the error rate of learned beta
        print "Computing error rate..."
        
        # generate transition matrix
        trans = genTrans(nnodes, g[0], ff, [source], alpha, beta_Opt[0])[0]
        # calculate pageRank
        pp = np.repeat(1.0/nnodes, nnodes)
        pgrank_learn = iterPageRank(pp, trans)
        
        # predict future links (future neighbors to the source) according to 
        # PageRank results from the learned parameter
        numFLinks = 100
        candPairs_learn = []
        for i in range(len(candidates)):
            candPairs_learn.append((candidates[i], pgrank_learn[candidates[i]]))
        candPairs_learn = sorted(candPairs_learn, key = lambda x: x[1], reverse = True)
        
        # construct predicted future link set
        Dset_learn = []
        for i in range(numFLinks):
            Dset_learn.append(candPairs_learn[i][0])
        
        # calculate false positive ratio as the error rate, 
        # print the result
        error_learn = 0
        for i in Dset_learn:
            if not(i in Dset):
                error_learn += 1
        error_learn = float(error_learn) / numFLinks
        
        
        
        ############################################
        ############################################
        ## alternative model comparison
        
        print "Running alternative models..."
        
        # link prediction with the exact parameters
        # generate transition matrix
        trans = genTrans(nnodes, g[0], ff, [source], alpha, beta)[0]
        # calculate pageRank
        pp = np.repeat(1.0/nnodes, nnodes)
        pgrank_exact = iterPageRank(pp, trans)
        
        # predict future links (future neighbors to the source) according to 
        # PageRank results from the learned parameter
        numFLinks = 100
        candPairs_exact = []
        for i in range(len(candidates)):
            candPairs_exact.append((candidates[i], pgrank_exact[candidates[i]]))
        candPairs_exact = sorted(candPairs_exact, key = lambda x: x[1], reverse = True)
        
        # construct predicted future link set
        Dset_exact = []
        for i in range(numFLinks):
            Dset_exact.append(candPairs_exact[i][0])
        
        # calculate false positive ratio as the error rate, 
        # print the result
        error_exact = 0
        for i in Dset_exact:
            if not(i in Dset):
                error_exact += 1
        error_exact = float(error_exact) / numFLinks
        
        ############################################
        # link prediction with unweighted PageRank
        # generate transition matrix
        trans = genTrans_plain(nnodes, g[0], [source], alpha)[0]
        # calculate pageRank
        pp = np.repeat(1.0/nnodes, nnodes)
        pgrank_uw = iterPageRank(pp, trans)
        
        # predict future links (future neighbors to the source) according to 
        # PageRank results from the learned parameter
        numFLinks = 100
        candPairs_uw = []
        for i in range(len(candidates)):
            candPairs_uw.append((candidates[i], pgrank_uw[candidates[i]]))
        candPairs_uw = sorted(candPairs_uw, key = lambda x: x[1], reverse = True)
        
        # construct predicted future link set
        Dset_uw = []
        for i in range(numFLinks):
            Dset_uw.append(candPairs_uw[i][0])
        
        # calculate false positive ratio as the error rate, 
        # print the result
        error_uw = 0
        for i in Dset_uw:
            if not(i in Dset):
                error_uw += 1
        error_uw = float(error_uw) / numFLinks
        
        
        l_error_temp.append(error_learn)
        ex_error_temp.append(error_exact)
        uw_error_temp.append(error_uw)
        print "learned model error rate =", error_learn
        print "exact model error rate =", error_exact
        print "unweighted model error rate =", error_uw
        
        tj = json.dumps({'name': 'tr', 'noise': noise, 'data': candPairs}, fjson)
        lj = json.dumps({'name': 'lr', 'noise': noise, 'data': candPairs_learn}, fjson)
        ej = json.dumps({'name': 'ex', 'noise': noise, 'data': candPairs_exact}, fjson)
        uj = json.dumps({'name': 'uw', 'noise': noise, 'data': candPairs_uw}, fjson)
        fjson.write(tj + '\n')
        fjson.write(lj + '\n')
        fjson.write(ej + '\n')
        fjson.write(uj + '\n')
        
        er_log = json.dumps({'noise': noise, 'error': 
            [error_learn, error_exact, error_uw], 'beta': [beta_Opt[0][0], beta_Opt[0][1]]})
        ferror.write(er_log + '\n')

        
    
    l_error.append(l_error_temp)
    ex_error.append(ex_error_temp)
    uw_error.append(uw_error_temp)

fjson.close()
ferror.close()
