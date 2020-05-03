#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:00:34 2020

@author: cyfile
"""


#K-means++
from pylab import *
from numpy import *
import codecs
import matplotlib.pyplot as plt
data=[]
labels=[]


 
#caculate distance
def distance(x1,x2):
    return sqrt(sum(power(x1-x2,2)))
 
#for each point, find the nearest centroid for it
def nearest(point, cluster_centers):
    min_dist = inf
    m = np.shape(cluster_centers)[0]  # kmeans++ the number of initialed centroid
    for i in range(m):
        # caculate the distance between each point and centroid
        d = distance(point, cluster_centers[i, ])
        # choose the cloest distance
        if min_dist > d:
            min_dist = d
    return min_dist
#Select the centroid as far as possible
def get_centroids(dataset, k):
    m, n = np.shape(dataset)
    cluster_centers = np.zeros((k , n))
    index = np.random.randint(0, m)
    cluster_centers[0,] = dataset[index, ]
    # 2、Initializes a sequence of distances
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、Find the nearest cluster center for each sample
            d[j] = nearest(dataset[j, ], cluster_centers[0:i, ])
            # 4、Add up all the shortest distances
            sum_all += d[j]
        # 5、Gets a random value between sum_all
        sum_all *= random.rand()
        # 6、The farthest sample point is obtained as the center of the cluster
        for j, di in enumerate(d):
            sum_all=sum_all - di
            if sum_all > 0:
                continue
            cluster_centers[i,] = dataset[j, ]
            break
    return cluster_centers
 
#main program
def Kmeans(dataset,k):
    row_m=shape(dataset)[0]
    cluster_assign=zeros((row_m,2))
    center=get_centroids(dataset,k)
    change=True
    while change:
        change=False
        for i in range(row_m):
            mindist=inf
            min_index=-1
            for j in range(k):
                distance1=distance(center[j,:],dataset[i,:])
                if distance1<mindist:
                    mindist=distance1
                    min_index=j
            if cluster_assign[i,0] != min_index:
                change=True
            cluster_assign[i,:]=min_index,mindist**2
        for cen in range(k):
            cluster_data=dataset[nonzero(cluster_assign[:,0]==cen)]
            center[cen,:]=mean(cluster_data,0)
    return center ,cluster_assign


 
