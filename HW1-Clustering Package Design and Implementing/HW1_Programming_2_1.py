#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:39:51 2020

@author: chuyue
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# import data
def read_dataset():
    dataset = pd.read_csv('three_globs.csv', sep = ',')
    dataset = np.array(dataset)
    return dataset

# random initialization methods
def random_initialization(dataset, k):
    dataset = list(dataset)
    return random.sample(dataset, k)

# k-means++ initialization methods
def kmeans_plus_plus(dataset, k):
    dataset = list(dataset)
    centroid_list = []
    first_point = random.sample(dataset,1)
    centroid_list.append(np.array(first_point).reshape(2,))
    dataset = np.array(dataset)
    # 2. distance list
    dis_list = []
    t = 1
    for number_cluster in range(k-1):
        for i in dataset:
            v1 = i
            min_distance = 1e+16 # set a biggest number
            for j in range(t):
                v2 = centroid_list[j]
                dis = calculate_distance(v1, v2)
                if dis < min_distance:
                    min_distance = dis
            dis_list.append(min_distance)
        sum_distance = sum(dis_list)
        sum_distance *= random.random()
        for j, d in enumerate(dis_list):
            sum_distance -= d
            if sum_distance > 0:
                continue
            centroid_list.append(dataset[j])
            break
    return centroid_list


''' not finished
# global k-means initialization methods
def global_kmeans(dataset, centroid_list, k):
    m = []
    for i in range(k):
        mi = 0
        for j in dataset:
            point = random.sample(dataset,1)
            mi += j
        mi = mi/len(dataset)
        m.append(mi)
'''               
        


# calculate_distance of two points
def calculate_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

#calculate the distance between each point and centroid, choose the minimum
def minimum_distance(dataset, centroid_list, k):
    cluster = dict()
    for i in dataset:
        c = -1
        v1 = i
        min_distance = 1e+16 # set a biggest number
        for j in range(k):
            v2 = centroid_list[j]
            distance = calculate_distance(v1, v2)
            if distance < min_distance:
                min_distance = distance
                c = j
        if c not in cluster.keys():
            cluster.setdefault(c, [])
        cluster[c].append(i)
    return cluster
                
#calculate centroids
def calculate_centroid(cluster):
    centroid_list = []
    for key in cluster:
        centroid = np.mean(cluster[key], axis = 0)
        centroid_list.append(centroid)
    return centroid_list

#calculate total distance within a group
def total_distance(centroid_list, cluster):
    sum_distance = 0
    for key in cluster:
        v1 = centroid_list[key]
        distance = 0
        for i in cluster[key]:
            v2 = i
            distance += calculate_distance(v1, v2)
        sum_distance += distance
    return sum_distance
    

#make visualization to see cluster results
def cluster_result(centroid_list, cluster):
    centroid_color = ['+m','+y','+c','+g','+b','+r','+k','+w']
    dataset_color = ['om','oy','oc','og','ob','or','ok','ow']
    for key in cluster:
        plt.plot(centroid_list[key][0], centroid_list[key][1],centroid_color[key],markersize=30)
        for i in cluster[key]:
            plt.plot(i[0],i[1],dataset_color[key])
    plt.show()
    

#test k-means
def test_kmeans(k, initialization):
    dataset = read_dataset()
    if initialization == 'random':
        centroid_list = random_initialization(dataset,k)
        print("random initialization")
    if initialization == 'k-means++':
        centroid_list = kmeans_plus_plus(dataset, k)
        print("k-means++ initialization")
    cluster = minimum_distance(dataset, centroid_list,k) #1st
    print("initial centroids: ", centroid_list)
    cluster_result(centroid_list, cluster)
    new_total_distance = total_distance(centroid_list, cluster)
    old_total_distance = 0
    t = 1
    
    while abs(new_total_distance - old_total_distance) >= 1e-6:
        centroid_list = calculate_centroid(cluster) #1st
        cluster = minimum_distance(dataset, centroid_list, k) #2nd
        t += 1
        old_total_distance = new_total_distance
        new_total_distance = total_distance(centroid_list, cluster)
        print(t,"times centroids: ",centroid_list)
        cluster_result(centroid_list, cluster)
            
test_kmeans(3, 'k-means++')            

