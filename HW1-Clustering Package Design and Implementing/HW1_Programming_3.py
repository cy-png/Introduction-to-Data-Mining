#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:52:33 2020

@author: cyfile
"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def calculate_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

def get_neibor(point, dataset, eps):
    res = []
    for i in range(dataset.shape[0]):
        if calculate_distance(point, dataset[i]) < eps:
            res.append(i)
    return res


def my_DBSCAN(dataset, eps, minPts):
    core_objects = {}
    cluster = {}
    n = dataset.shape[0]
    for i in range(n):
        neibor = get_neibor(dataset[i], dataset, eps)
        if len(neibor) >= minPts:
            core_objects[i] = neibor
    old_core_objects = core_objects.copy()
    k = 0
    unvisit = list(range(n))
    while len(core_objects) > 0:
        old_unvisit = []
        old_unvisit.extend(unvisit)
        core = list(core_objects.keys())
        random_n = random.randint(0, len(core))
        core_point = core[random_n]
        q = []
        q.append(core_point)
        unvisit.remove(core_point)
        while len(q) > 0:
            q_point = q[0]
            del q[0]
            if q_point in old_core_objects.keys():
                delta = [j for j in old_core_objects[q_point] if j in unvisit]
                q.extend(delta)
                unvisit = [h for h in unvisit if h not in delta]
        k += 1
        cluster[k] = [l for l in old_unvisit if l not in unvisit]
        for m in cluster[k]:
            if m in core_objects.keys():
                del core_objects[m]
    return cluster


dataset = pd.read_csv('anthill.csv', sep = ',')
dataset = np.array(dataset)

plt.scatter(dataset[:,0], dataset[:,1])

cluster = my_DBSCAN(dataset, 0.2, 10)

def cluster_result(cluster, dataset):
    cluster_color = ['m','y','c','g','b','r','k','w']
    for i in cluster.keys():
        X = []
        Y = []
        points = cluster[i]
        for j in range(len(points)):
            X.append(dataset[points[j]][0])
            Y.append(dataset[points[j]][1])
        plt.scatter(X, Y, marker='o', color=cluster_color[i % len(cluster_color)], label=i)
    plt.legend(loc='upper right')
    plt.show()

cluster_result(cluster, dataset)
