# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:50:35 2015

@author: Timber
"""
import json
from matplotlib import pyplot as plt

fe = open('Synthesized logs/synthetic_error_rate.json', 'r')

# extract error data from the json file
errors = {}
for line in fe:
    temp = json.loads(line)
    if temp['noise'] in errors.keys():
        errors[temp['noise']]['learn error'].append(temp['error'][0])
        errors[temp['noise']]['exact error'].append(temp['error'][1])
        errors[temp['noise']]['unweight error'].append(temp['error'][2])
    else:
        errors[temp['noise']] = {}
        errors[temp['noise']]['learn error'] = [temp['error'][0]]
        errors[temp['noise']]['exact error'] = [temp['error'][1]]
        errors[temp['noise']]['unweight error'] = [temp['error'][2]]
fe.close()

# get the aerage error for all noise levels
errorQuad = []
for noise in errors.keys():
    lr_error = float(sum(errors[noise]['learn error']))/ len(errors[noise]['learn error'])
    ex_error = float(sum(errors[noise]['exact error']))/ len(errors[noise]['exact error'])
    uw_error = float(sum(errors[noise]['unweight error']))/ len(errors[noise]['unweight error'])
    errorQuad.append((noise, lr_error, ex_error, uw_error))
errorQuad = sorted(errorQuad, key = lambda x: x[0])

noise = []
lr_error = []
ex_error = []
uw_error = []
for i in range(len(errorQuad)):
    noise.append(errorQuad[i][0])
    lr_error.append(errorQuad[i][1])
    ex_error.append(errorQuad[i][2])
    uw_error.append(errorQuad[i][3])

# plot the errors
plt.plot(noise, lr_error, '-o', c='blue', label = 'Learned weights')
plt.plot(noise, ex_error, '-x', c='green', label = 'Exact weights [1, -1]')
plt.plot(noise, uw_error, '-*', c='red', label = 'Unweighted RW')
plt.legend(loc = 4)
plt.xlabel("Noise Level")
plt.ylabel("Error Rate")
plt.figure(figsize = (40, 20))



